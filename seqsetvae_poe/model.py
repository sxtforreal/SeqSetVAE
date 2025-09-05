import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import get_linear_schedule_with_warmup

from .modules import SetVAEModule, AttentiveBottleneckLayer, elbo_loss as base_elbo, recon_loss as chamfer_recon


class _SetDecoder(nn.Module):
    def __init__(self, latent_dim, reduced_dim, levels, heads, m):
        super().__init__()
        self.levels = levels
        self.decoder_layers = nn.ModuleList(
            [AttentiveBottleneckLayer(latent_dim, heads, m) for _ in range(levels)]
        )
        self.out = nn.Linear(latent_dim, reduced_dim)

    def forward(self, h, target_n, noise_std=0.5):
        current = h.unsqueeze(1)
        for l in range(self.levels - 1, -1, -1):
            layer_out = self.decoder_layers[l](current, target_n, noise_std=noise_std)
            current = layer_out + current.expand_as(layer_out)
        return self.out(current)


def _strict_causal_mask(S: int, device: torch.device):
    return torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)


class PoESeqSetVAEPretrain(pl.LightningModule):
    """
    Pretraining with Product-of-Experts:
      - Expert 1 (post): q_x(z_t|x_t) from SetVAE encoder
      - Expert 2 (prior): p(z_t|z_{<t}, \u0394t) from Transformer
      - Merge: q(z_t) \u221d q_x(z_t) * p(z_t)

    Extras:
      - Stale-dropout: randomly drop values whose `carry_mask==1`.
      - Optional Set-MAE masking inside each set (disabled by default here).
      - Keep the existing decoder; remove multi-type decoders and random windows.
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
        ff_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        transformer_dropout: float = 0.15,
        warmup_beta: bool = True,
        max_beta: float = 0.05,
        beta_warmup_steps: int = 8000,
        free_bits: float = 0.03,
        stale_dropout_p: float = 0.2,
        set_mae_ratio: float = 0.0,
    ):
        super().__init__()

        # Per-set encoder
        self.set_encoder = SetVAEModule(
            input_dim=input_dim,
            reduced_dim=reduced_dim,
            latent_dim=latent_dim,
            levels=levels,
            heads=heads,
            m=m,
        )

        # Causal transformer producing p(z_t|history)
        enc_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.post_transformer_norm = nn.LayerNorm(latent_dim, eps=1e-6)

        # Predict prior mean/logvar per step from h_{t-1}
        self.prior_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim * 2)
        )

        # Decoder (same as existing)
        self.decoder = _SetDecoder(latent_dim, reduced_dim if reduced_dim is not None else input_dim, levels, heads, m)

        # Time bucket embedding for \u0394t
        self.num_time_buckets = 64
        edges = torch.logspace(math.log10(0.5), math.log10(24 * 60.0), steps=self.num_time_buckets - 1)
        self.register_buffer("time_bucket_edges", edges, persistent=False)
        self.rel_time_bucket_embed = nn.Embedding(self.num_time_buckets, latent_dim)

        # Training hparams
        self.lr = lr
        self.beta = beta
        self.free_bits = free_bits
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.stale_dropout_p = stale_dropout_p
        self.set_mae_ratio = set_mae_ratio
        self.latent_dim = latent_dim
        self._step = 0
        self.save_hyperparameters()

    # --- helpers ---
    def _relative_time_bucket_embedding(self, minutes: torch.Tensor):
        B, S = minutes.shape
        diffs = (minutes[:, 1:] - minutes[:, :-1]).clamp(min=0.0)
        deltas = torch.cat([torch.zeros(B, 1, device=minutes.device, dtype=minutes.dtype), diffs], dim=1)
        log_delta = torch.log1p(deltas)
        log_edges = torch.log1p(self.time_bucket_edges).to(log_delta.device)
        idx = torch.bucketize(log_delta, log_edges, right=False).clamp(max=self.num_time_buckets - 1)
        return self.rel_time_bucket_embed(idx)

    def _apply_stale_dropout(self, val: torch.Tensor, carry_mask: Optional[torch.Tensor]):
        if carry_mask is None or self.stale_dropout_p <= 0.0 or not self.training:
            return val
        drop = (torch.rand_like(val) < self.stale_dropout_p) & (carry_mask > 0.5)
        return val.masked_fill(drop, 0.0)

    def _poe(self, mu_qx, logvar_qx, mu_p, logvar_p):
        var_qx = logvar_qx.exp()
        var_p = logvar_p.exp()
        var_post = 1.0 / (1.0 / var_qx + 1.0 / var_p)
        mu_post = var_post * (mu_qx / var_qx + mu_p / var_p)
        logvar_post = torch.log(var_post + 1e-8)
        return mu_post, logvar_post

    def _beta(self):
        if not self.warmup_beta:
            return self.max_beta
        if self._step < self.beta_warmup_steps:
            return self.max_beta * (self._step / self.beta_warmup_steps)
        return self.max_beta

    # --- core ---
    def _split_sets(self, var, val, minutes, set_id, padding_mask=None):
        B = var.size(0)
        all_sets = []
        for b in range(B):
            v = var[b]
            x = val[b]
            t = minutes[b]
            s = set_id[b]
            if padding_mask is not None:
                mask = ~padding_mask[b]
                v, x, t, s = v[mask], x[mask], t[mask], s[mask]
            if s.dim() > 1:
                s = s.squeeze(-1)
            uniq, counts = torch.unique_consecutive(s.long(), return_counts=True)
            idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(c) for c in counts])
            patient_sets = []
            for idx in idx_splits:
                patient_sets.append({
                    "var": v[idx].unsqueeze(0),
                    "val": x[idx].unsqueeze(0),
                    "minute": t[idx].unsqueeze(0),
                })
            all_sets.append(patient_sets)
        return all_sets

    def _forward_single(self, sets, carry_mask=None, minutes_seq=None):
        S = len(sets)
        device = self.device
        if S == 0:
            z = torch.zeros(1, 0, self.latent_dim, device=device)
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        z_mu_list, z_logvar_list, pos_list = [], [], []
        z_mu_post_list, z_logvar_post_list = [], []
        kl_total = 0.0

        # encode per set: q_x(z|x)
        for i, s in enumerate(sets):
            var, val, minute = s["var"], s["val"], s["minute"]
            # normalize and stale-dropout already handled upstream; here encode
            _, z_list, _ = self.set_encoder(var, val)
            _, mu_qx, logvar_qx = z_list[-1]
            z_mu_list.append(mu_qx.squeeze(1))
            z_logvar_list.append(logvar_qx.squeeze(1))
            pos_list.append(minute.unique().float())

        z_mu = torch.stack(z_mu_list, dim=1)  # [1,S,D]
        z_logvar = torch.stack(z_logvar_list, dim=1)  # [1,S,D]
        minutes = torch.stack(pos_list, dim=1)
        if minutes.dim() == 1:
            minutes = minutes.unsqueeze(0)

        # history -> prior per step
        attn_mask = _strict_causal_mask(z_mu.size(1), device=z_mu.device)
        # add relative time embedding
        z_mu_with_time = z_mu + self._relative_time_bucket_embedding(minutes)
        h = self.transformer(z_mu_with_time, mask=attn_mask)
        h = self.post_transformer_norm(h)
        prior_params = self.prior_head(h)  # [1,S,2D]
        mu_p, logvar_p = prior_params.chunk(2, dim=-1)

        # PoE q(z) \u221d q_x(z) * p(z)
        mu_post, logvar_post = self._poe(z_mu, z_logvar, mu_p, logvar_p)
        z_mu_post_list.append(mu_post)
        z_logvar_post_list.append(logvar_post)

        # KL between q_x and q_post as regularizer (encourage using prior)
        var_qx, var_post = z_logvar.exp(), logvar_post.exp()
        kl = 0.5 * torch.sum(var_post / var_qx + (mu_post - z_mu) ** 2 / var_qx - 1 + (z_logvar - logvar_post), dim=-1)
        kl_total = kl.mean()

        # reconstruct from h (use h_t)
        recon_total = 0.0
        for idx, s in enumerate(sets):
            N_t = s["var"].size(1)
            recon = self.decoder(h[:, idx], N_t, noise_std=(0.0 if not self.training else 0.3))
            if self.set_encoder.dim_reducer is not None:
                reduced = self.set_encoder.dim_reducer(s["var"])
            else:
                reduced = s["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * s["val"]
            recon_total += chamfer_recon(recon, target_x)
        recon_total = recon_total / max(1, S)
        return recon_total, kl_total

    def forward(self, batch):
        var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        # stale-dropout on values
        val = self._apply_stale_dropout(val, carry_mask)
        # split into sets per patient
        all_sets = self._split_sets(var, val, minutes, set_id, padding_mask)
        total_recon, total_kl, count = 0.0, 0.0, 0
        for sets in all_sets:
            recon, kl = self._forward_single(sets)
            total_recon += recon
            total_kl += kl
            count += 1
        if count == 0:
            device = var.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        return total_recon / count, total_kl / count

    # training hooks
    def training_step(self, batch, batch_idx):
        recon, kl = self.forward(batch)
        beta = self._beta()
        loss = recon + beta * kl
        self.log_dict({"train_loss": loss, "train_recon": recon, "train_kl": kl, "train_beta": beta}, prog_bar=True)
        self._step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        recon, kl = self.forward(batch)
        beta = self._beta()
        loss = recon + beta * kl
        self.log_dict({"val_loss": loss, "val_recon": recon, "val_kl": kl, "val_beta": beta}, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=2_000, num_training_steps=200_000)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

