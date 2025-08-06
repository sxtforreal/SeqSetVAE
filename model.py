import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import AdamW
from torchmetrics.classification import AUROC, AveragePrecision, Accuracy
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup
from modules import (
    SetVAEModule,
    AttentiveBottleneckLayer,
    elbo_loss,
    recon_loss as chamfer_recon_loss,
)


##### SetVAE
class SetVAE(pl.LightningModule):
    """Set representation without historical information."""

    def __init__(
        self, input_dim, reduced_dim, latent_dim, levels, heads, m, beta, lr, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.setvae = SetVAEModule(input_dim, reduced_dim, latent_dim, levels, heads, m)
        self.beta = beta
        self.lr = lr
        self.warmup_steps = kwargs.get("warmup_steps", 10_000)
        self.max_steps = kwargs.get("max_steps", 1_000_000)

    def forward(self, var, val):
        return self.setvae(var, val)

    def training_step(self, batch, batch_idx):
        var, val = batch["var"], batch["val"]
        recon, z_list, _ = self(var, val)
        reduced = (
            self.setvae.dim_reducer(var) if self.setvae.dim_reducer is not None else var
        )
        input_x = reduced * val
        recon_loss, kl = elbo_loss(recon, input_x, z_list)
        loss = recon_loss + self.beta * kl
        self.log_dict(
            {"train_loss": loss, "train_recon": recon_loss, "train_kl": kl},
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        var, val = batch["var"], batch["val"]
        recon, z_list, _ = self(var, val)
        reduced = (
            self.setvae.dim_reducer(var) if self.setvae.dim_reducer is not None else var
        )
        input_x = reduced * val
        recon_loss, kl = elbo_loss(recon, input_x, z_list)  # noqa: F405
        loss = recon_loss + self.beta * kl
        self.log_dict(
            {"val_loss": loss, "val_recon": recon_loss, "val_kl": kl}, prog_bar=True
        )
        return loss

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        sch = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }


##### Decoder
class _SetDecoder(nn.Module):
    """Reconstruct input events from historical information enriched representation."""

    def __init__(self, latent_dim, reduced_dim, levels, heads, m):
        super().__init__()
        self.levels = levels
        self.dir_layers = nn.ModuleList(
            [AttentiveBottleneckLayer(latent_dim, heads, m) for _ in range(levels)]
        )
        self.mag_layers = nn.ModuleList(
            [AttentiveBottleneckLayer(latent_dim, heads, m) for _ in range(levels)]
        )
        self.dir_out = nn.Linear(latent_dim, reduced_dim)
        self.mag_out = nn.Linear(latent_dim, 1)

    def forward(self, h, target_n, noise_std=0.5):
        # Direction branch
        current_dir = h.unsqueeze(1)
        for l in range(self.levels - 1, -1, -1):
            layer_out = self.dir_layers[l](current_dir, target_n, noise_std=noise_std)
            current_dir = layer_out + current_dir.expand_as(layer_out) + h.unsqueeze(1)
        dir_recon = self.dir_out(current_dir)
        dir_norm = F.normalize(dir_recon, dim=-1)

        # Magnitude branch
        current_mag = h.unsqueeze(1)
        for l in range(self.levels - 1, -1, -1):
            layer_out = self.mag_layers[l](current_mag, target_n, noise_std=noise_std)
            current_mag = layer_out + current_mag.expand_as(layer_out)
        mag_recon = self.mag_out(current_mag)

        recon = dir_norm * mag_recon
        return recon


##### Sequential SetVAE
class SeqSetVAE(pl.LightningModule):
    """
    1. Pretrained SetVAE encoder -> z_prim (historical information free)
    2. TransformerEncoder(with causal mask) infuse historical information -> h_t
    3. Reconstruct current event from h_t
    4. Predict sequence level label
    """

    def __init__(
        self,
        input_dim: int,
        reduced_dim: int,
        latent_dim: int,
        levels: int,
        heads: int,
        m: int,
        beta: float,
        lr: float,
        num_classes: int,
        ff_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        pretrained_ckpt: str,
        w: float = 1.0,
        free_bits: float = 0.1,
    ):

        super().__init__()

        # Pretrained SetVAE
        self.setvae = SetVAE(
            input_dim,
            reduced_dim,
            latent_dim,
            levels,
            heads,
            m,
            beta,
            lr,
        )
        if pretrained_ckpt is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(pretrained_ckpt, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            setvae_state = {
                k.replace("setvae.", ""): v
                for k, v in state_dict.items()
                if k.startswith("setvae.")
            }
            self.setvae.load_state_dict(setvae_state, strict=False)
            del ckpt, state_dict, setvae_state

        # Transformer encoder
        enc_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # Rotary positional encoding constant
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, latent_dim, 2).float() / latent_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Decoder & Classifier
        self.decoder = _SetDecoder(
            latent_dim,
            reduced_dim,
            levels,
            heads,
            m,
        )
        self.cls_head = nn.Linear(latent_dim, num_classes)
        torch.nn.init.xavier_uniform_(self.cls_head.weight)
        if self.cls_head.bias is not None:
            nn.init.zeros_(self.cls_head.bias)

        # Metrics
        task_type = "binary" if num_classes == 2 else "multiclass"
        self.val_auc = AUROC(task=task_type, num_classes=num_classes)
        self.val_auprc = AveragePrecision(task=task_type, num_classes=num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.num_classes = num_classes

        # Training hyperparameters
        self.w = w
        self.beta = beta
        self.lr = lr
        self.free_bits = free_bits
        self.latent_dim = latent_dim
        self.save_hyperparameters(ignore=["setvae"])

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor):
        """
        Apply Rotary Positional Embedding to the sequence.

        Args:
            x: Tensor of shape [B, S, D]
            pos: Tensor of shape [B, S] representing each set's minute value.
        Returns:
            Tensor of same shape as x with RoPE applied.
        """
        # Ensure even dimension
        d = x.shape[-1]
        if d % 2 != 0:
            raise ValueError("Embedding dimension must be even for RoPE.")
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq  # [B, S, D/2]
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        # Interleave to match embedding dimension
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated

    def _causal_mask(self, unique_sets, device):
        mask = torch.triu(
            torch.ones(unique_sets, unique_sets, device=device), diagonal=1
        )
        return mask.bool()

    def forward(self, sets):
        """sets : list of dict{[B, N, D],[B, N, 1],[B, N, 1]}"""

        S = len(sets)
        z_prims, kl_total = [], 0.0
        pos_list = []
        for s_dict in sets:
            var, val, time = (
                s_dict["var"],
                s_dict["val"],
                s_dict["minute"],
            )
            assert time.unique().numel() == 1, "Time is not constant in this set"
            minute_val = time.unique()
            pos_list.append(minute_val)
            recon, z_list, _ = self.setvae(var, val)
            # Clamp all logvar in z_list
            for i in range(len(z_list)):
                z_sample, mu, logvar = z_list[i]
                logvar = torch.clamp(logvar, min=-1.0, max=1.0)
                z_list[i] = (z_sample, mu, logvar)
            z_sample, mu, logvar = z_list[-1]  # Choose the deepest layer
            z_prims.append(z_sample.squeeze(1))
            for _, mu, logvar in z_list:
                raw_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                min_kl = self.free_bits * self.latent_dim
                clamped_kl = torch.clamp(raw_kl, min=min_kl)
                kl_total += clamped_kl.mean()
        kl_total /= S * len(z_list)
        z_seq = torch.stack(z_prims, dim=1)
        pos_tensor = torch.stack(pos_list, dim=1)
        z_seq = self._apply_rope(z_seq, pos_tensor)
        h_seq = self.transformer(z_seq, mask=self._causal_mask(S, z_seq.device))

        # Reconstruction
        recon_loss_total = 0.0
        for idx, s_dict in enumerate(sets):
            N_t = s_dict["var"].size(1)
            recon = self.decoder(h_seq[:, idx], N_t)
            if self.setvae.setvae.dim_reducer is not None:
                reduced = self.setvae.setvae.dim_reducer(s_dict["var"])
            else:
                reduced = s_dict["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * s_dict["val"]
            recon_loss_total += chamfer_recon_loss(recon, target_x)
        recon_loss_total /= S

        # Classification
        final_rep = h_seq.mean(dim=1)
        logits = self.cls_head(final_rep)
        return logits, recon_loss_total, kl_total, z_seq

    # Helpers
    def _split_sets(self, var, val, time, set_ids):
        """Split a concatenated patient sequence into list-of-set dicts."""
        sets = []
        s = set_ids[0]
        _, counts = torch.unique_consecutive(s, return_counts=True)
        indices = torch.split(torch.arange(s.size(0), device=s.device), counts.tolist())
        for idx in indices:
            sets.append(
                {
                    "var": var[0, idx].unsqueeze(0),
                    "val": val[0, idx].unsqueeze(0),
                    "minute": time[0, idx].unsqueeze(0),
                }
            )
        return sets

    def _step(self, batch, stage: str):
        var, val, time, set_ids, label = (
            batch["var"],
            batch["val"],
            batch["minute"],
            batch["set_id"],
            batch.get("label"),
        )
        sets = self._split_sets(var, val, time, set_ids)
        logits, recon_loss, kl_loss, z_seq = self(sets)
        pred_loss = F.cross_entropy(logits, label)
        total_loss = pred_loss + self.w * (recon_loss + kl_loss)

        # Metrics for validation stage
        if stage == "val":
            z_mean = z_seq.mean(dim=1)
            if z_mean.size(0) > 1:
                z_var = torch.var(z_mean, dim=0).mean()
            else:
                z_var = 0.0
            self.log("val_z_var", z_var, prog_bar=True)

            if self.num_classes == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.softmax(logits, dim=1)
            self.val_auc.update(probs, label)
            self.val_auprc.update(probs, label)
            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, label)

        self.log_dict(
            {
                f"{stage}_loss": total_loss,
                f"{stage}_recon": recon_loss,
                f"{stage}_kl": kl_loss,
                f"{stage}_pred": pred_loss,
            },
            prog_bar=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        auprc = self.val_auprc.compute()
        acc = self.val_acc.compute()
        self.log_dict(
            {
                "val_auc": auc,
                "val_auprc": auprc,
                "val_accuracy": acc,
            },
            prog_bar=True,
        )
        # reset
        self.val_auc.reset()
        self.val_auprc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        return opt
