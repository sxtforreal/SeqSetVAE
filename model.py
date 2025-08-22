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
    elbo_loss,
    AttentiveBottleneckLayer,
    recon_loss as chamfer_recon_loss,
)
from losses import FocalLoss
import math
from typing import Optional


def load_checkpoint_weights(checkpoint_path, device='cpu'):
    """
    Load weights from checkpoint - handles both full PyTorch Lightning checkpoints 
    and direct state dicts (weights only).
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
    
    Returns:
        state_dict: The model state dict containing only weights
    """
    print(f"📦 Loading weights from: {checkpoint_path}")
    print("📦 Loading weights only (no optimizer state or other metadata)")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Full PyTorch Lightning checkpoint
        state_dict = ckpt["state_dict"]
        print(f"✅ Found PyTorch Lightning checkpoint with state_dict")
    else:
        # Direct state dict (weights only)
        state_dict = ckpt
        print(f"✅ Found direct state dict (weights only)")
    
    return state_dict


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
            {"val/loss": loss, "val/recon": recon_loss, "val/kl": kl}, prog_bar=True
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
        self.decoder_layers = nn.ModuleList(
            [AttentiveBottleneckLayer(latent_dim, heads, m) for _ in range(levels)]
        )
        self.out = nn.Linear(latent_dim, reduced_dim)

    def forward(self, h, target_n, noise_std=0.5):
        current = h.unsqueeze(1)  # [B, 1, latent_dim]
        for l in range(self.levels - 1, -1, -1):
            layer_out = self.decoder_layers[l](current, target_n, noise_std=noise_std)
            current = layer_out + current.expand_as(layer_out)
        recon = self.out(current)
        return recon


##### Pretraining-only Sequential SetVAE (no classifier)
class SeqSetVAEPretrain(pl.LightningModule):
    """
    Pretraining stage for SeqSetVAE without classification.
    - Encode each set with SetVAE to obtain z_t (history-free latent)
    - Inject historical info via Transformer with a STRICT causal mask
    - Decode each enriched state h_t back to the original set to reconstruct
    - Optimize reconstruction + KL only (with beta warmup)
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
        warmup_beta: bool = True,
        max_beta: float = 0.1,
        beta_warmup_steps: int = 5000,
        free_bits: float = 0.1,
        transformer_dropout: float = 0.1,
        time_num_frequencies: int = 64,
    ):
        super().__init__()

        # Encoder for per-set latents
        self.set_encoder = SetVAEModule(
            input_dim=input_dim,
            reduced_dim=reduced_dim,
            latent_dim=latent_dim,
            levels=levels,
            heads=heads,
            m=m,
        )

        # Transformer for historical conditioning (STRICT causal mask)
        enc_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.post_transformer_norm = nn.LayerNorm(latent_dim, eps=1e-6)

        # Decoder to reconstruct sets from enriched latents
        self.decoder = _SetDecoder(
            latent_dim=latent_dim,
            reduced_dim=reduced_dim if reduced_dim is not None else input_dim,
            levels=levels,
            heads=heads,
            m=m,
        )

        # Robust time/position encoding (noise-tolerant)
        # - Absolute sinusoidal position (index-based)
        # - Relative time bucket embeddings for Δt between sets
        self.num_time_buckets = 64
        edges = torch.logspace(math.log10(0.5), math.log10(24 * 60.0), steps=self.num_time_buckets - 1)
        self.register_buffer("time_bucket_edges", edges, persistent=False)
        self.rel_time_bucket_embed = nn.Embedding(self.num_time_buckets, latent_dim)
        # ALiBi-like relative time bias parameters (shared across heads)
        self.alibi_slope = nn.Parameter(torch.tensor(1.0))
        self.time_tau = nn.Parameter(torch.tensor(60.0))

        # Training hyperparameters
        self.lr = lr
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.free_bits = free_bits
        self.latent_dim = latent_dim
        self.current_step = 0

        # For collapse monitoring
        self._last_z_list = None

        self.save_hyperparameters()

    # ----------------------- Positional/Time Encoding -----------------------
    def _sinusoidal_positional_encoding(self, seq_len: int, dim: int, device: torch.device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))  # [D/2]
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [S, D]

    def _relative_time_bucket_embedding(self, minutes: torch.Tensor):
        # minutes: [B, S] (float, minutes)
        B, S = minutes.shape
        deltas = torch.cat([
            torch.zeros(B, 1, device=minutes.device, dtype=minutes.dtype),
            torch.diff(minutes, dim=1).clamp(min=0.0)
        ], dim=1)
        log_delta = torch.log1p(deltas)
        log_edges = torch.log1p(self.time_bucket_edges).to(log_delta.device)
        bucket_idx = torch.bucketize(log_delta, log_edges, right=False)
        bucket_idx = bucket_idx.clamp(max=self.num_time_buckets - 1)
        return self.rel_time_bucket_embed(bucket_idx)  # [B, S, D]

    def _apply_positional_encoding(self, x: torch.Tensor, minutes: torch.Tensor):
        # x: [B, S, D]; minutes: [B, S]
        B, S, D = x.shape
        pos = self._sinusoidal_positional_encoding(S, D, x.device).unsqueeze(0).expand(B, -1, -1)
        rel_time_emb = self._relative_time_bucket_embedding(minutes)
        return x + pos + rel_time_emb

    # ----------------------- Masks -----------------------
    def _strict_causal_mask(self, seq_len: int, device: torch.device):
        # Bool mask: True means blocked (no attention)
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _build_causal_time_bias_mask(self, minutes_1d: torch.Tensor):
        # minutes_1d: [S] (float minutes)
        S = minutes_1d.size(0)
        device = minutes_1d.device
        dt = (minutes_1d.unsqueeze(0) - minutes_1d.unsqueeze(1)).clamp(min=0.0)  # [S, S], dt_{i,j} = t_i - t_j
        bias = -self.alibi_slope * torch.log1p(dt / self.time_tau)  # [S, S]
        float_mask = bias.to(dtype=torch.float32)
        future = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        float_mask = float_mask.masked_fill(future, float("-inf"))
        return float_mask

    # ----------------------- Beta schedule -----------------------
    def _current_beta(self) -> float:
        if not self.warmup_beta:
            return self.max_beta
        if self.current_step < self.beta_warmup_steps:
            return self.max_beta * (self.current_step / self.beta_warmup_steps)
        return self.max_beta

    # ----------------------- Core Forward -----------------------
    def _split_sets(self, var, val, time, set_ids, padding_mask=None):
        batch_size = var.size(0)
        all_sets = []
        for b in range(batch_size):
            patient_sets = []
            patient_var = var[b]
            patient_val = val[b]
            patient_time = time[b]
            patient_set_ids = set_ids[b]
            if padding_mask is not None:
                patient_padding_mask = padding_mask[b]
                valid_indices = ~patient_padding_mask
                if not valid_indices.any():
                    patient_sets.append({
                        "var": torch.empty(0, patient_var.size(-1), device=patient_var.device).unsqueeze(0),
                        "val": torch.empty(0, 1, device=patient_val.device).unsqueeze(0),
                        "minute": torch.empty(0, 1, device=patient_time.device).unsqueeze(0),
                    })
                    all_sets.append(patient_sets)
                    continue
                patient_var = patient_var[valid_indices]
                patient_val = patient_val[valid_indices]
                patient_time = patient_time[valid_indices]
                patient_set_ids = patient_set_ids[valid_indices]
            if len(patient_set_ids) == 0:
                patient_sets.append({
                    "var": patient_var.unsqueeze(0),
                    "val": patient_val.unsqueeze(0),
                    "minute": patient_time.unsqueeze(0),
                })
            else:
                if patient_set_ids.dim() > 1:
                    patient_set_ids = patient_set_ids.squeeze(-1)
                if patient_set_ids.dtype != torch.long:
                    patient_set_ids = patient_set_ids.long()
                try:
                    if hasattr(torch, 'unique_consecutive'):
                        unique_set_ids, counts = torch.unique_consecutive(patient_set_ids, return_counts=True)
                    else:
                        unique_set_ids = torch.unique(patient_set_ids)
                        counts = torch.tensor([(patient_set_ids == uid).sum().item() for uid in unique_set_ids])
                    counts_list = [int(c.item()) for c in counts]
                    if not counts_list or any(c <= 0 for c in counts_list):
                        patient_sets.append({
                            "var": patient_var.unsqueeze(0),
                            "val": patient_val.unsqueeze(0),
                            "minute": patient_time.unsqueeze(0),
                        })
                    else:
                        indices = torch.split(torch.arange(len(patient_set_ids), device=patient_set_ids.device), counts_list)
                        for idx in indices:
                            if len(idx) > 0:
                                patient_sets.append({
                                    "var": patient_var[idx].unsqueeze(0),
                                    "val": patient_val[idx].unsqueeze(0),
                                    "minute": patient_time[idx].unsqueeze(0),
                                })
                except Exception:
                    patient_sets.append({
                        "var": patient_var.unsqueeze(0),
                        "val": patient_val.unsqueeze(0),
                        "minute": patient_time.unsqueeze(0),
                    })
            all_sets.append(patient_sets)
        return all_sets

    def _forward_single(self, sets):
        if not isinstance(sets, list):
            raise ValueError(f"Expected sets to be a list, got {type(sets)}")
        S = len(sets)
        if S == 0:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            recon_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
            return recon_loss, kl_loss

        z_prims, kl_total = [], 0.0
        pos_list = []
        all_z_lists = []
        for i, s_dict in enumerate(sets):
            required_keys = ["var", "val", "minute"]
            for key in required_keys:
                if key not in s_dict:
                    raise ValueError(f"Missing required key '{key}' in s_dict at index {i}")
            var, val, time = s_dict["var"], s_dict["val"], s_dict["minute"]
            assert time.unique().numel() == 1, "Time is not constant in this set"
            minute_val = time.unique().float()
            pos_list.append(minute_val)

            # Encode set -> z
            # Deterministic latent features during pretraining metrics: use mu
            _, z_list, _ = self.set_encoder(var, val)
            z_sample, mu, logvar = z_list[-1]
            z_prims.append(mu.squeeze(1))
            all_z_lists.append(z_list)

            kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
            min_kl = self.free_bits * self.latent_dim
            kl_div = torch.clamp(kl_div, min=min_kl)
            var_reg = -0.1 * torch.mean(logvar)
            kl_total += kl_div.mean() + var_reg

        kl_total = kl_total / S
        z_seq = torch.stack(z_prims, dim=1)  # [B, S, D]
        pos_tensor = torch.stack(pos_list, dim=1)
        # normalize shape to [B, S]
        if pos_tensor.dim() == 1:
            pos_tensor = pos_tensor.unsqueeze(0)
        elif pos_tensor.dim() == 3:
            pos_tensor = pos_tensor.squeeze(-1)

        # Positional + relative time bucket encoding
        z_seq = self._apply_positional_encoding(z_seq, pos_tensor)
        z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])

        # Build causal + relative-time bias mask (per patient)
        minutes_1d = pos_tensor.squeeze(0) if pos_tensor.dim() == 2 else pos_tensor.view(-1)
        attn_mask = self._build_causal_time_bias_mask(minutes_1d)
        h_seq = self.transformer(z_seq, mask=attn_mask)
        h_seq = self.post_transformer_norm(h_seq)

        # Reconstruction
        recon_loss_total = 0.0
        valid_sets = 0
        last_recon_list = []
        last_target_list = []
        for idx, s_dict in enumerate(sets):
            N_t = s_dict["var"].size(1)
            recon = self.decoder(h_seq[:, idx], N_t, noise_std=(0.0 if not self.training else 0.3))
            if self.set_encoder.dim_reducer is not None:
                reduced = self.set_encoder.dim_reducer(s_dict["var"])
            else:
                reduced = s_dict["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * s_dict["val"]
            # store for visualization
            try:
                last_recon_list.append(recon.detach())
                last_target_list.append(target_x.detach())
            except Exception:
                pass
            recon_loss_total += chamfer_recon_loss(recon, target_x)
            valid_sets += 1

        if valid_sets > 0:
            recon_loss_total = recon_loss_total / valid_sets

        # Save last reconstructions and targets for visualization
        try:
            self._last_recon_list = locals().get('last_recon_list', [])
            self._last_target_list = locals().get('last_target_list', [])
        except Exception:
            pass

        if all_z_lists:
            self._last_z_list = all_z_lists[0]

        return recon_loss_total, kl_total

    def forward(self, batch_or_sets):
        # Provides compatibility with both direct list-of-sets and dataloader dict
        if isinstance(batch_or_sets, dict):
            var, val, time, set_ids = (
                batch_or_sets["var"],
                batch_or_sets["val"],
                batch_or_sets["minute"],
                batch_or_sets["set_id"],
            )
            padding_mask = batch_or_sets.get("padding_mask", None)
            all_patient_sets = self._split_sets(var, val, time, set_ids, padding_mask)
            total_recon, total_kl, valid = 0.0, 0.0, 0
            for patient_sets in all_patient_sets:
                recon_loss, kl_loss = self._forward_single(patient_sets)
                total_recon += recon_loss
                total_kl += kl_loss
                valid += 1
            if valid == 0:
                device = var.device
                return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            return total_recon / valid, total_kl / valid
        elif isinstance(batch_or_sets, list):
            return self._forward_single(batch_or_sets)
        else:
            raise ValueError("Unsupported input type for forward")

    # ----------------------- Training hooks -----------------------
    def training_step(self, batch, batch_idx):
        recon_loss, kl_loss = self.forward(batch)
        beta = self._current_beta()
        total = recon_loss + beta * kl_loss
        self.log_dict({
            "train_loss": total,
            "train_recon": recon_loss,
            "train_kl": kl_loss,
            "train_beta": beta,
        }, prog_bar=True, on_step=True, on_epoch=True)
        self.logged_metrics = {
            'train_kl': kl_loss,
            'train_recon': recon_loss,
        }
        self.current_step += 1
        return total

    def validation_step(self, batch, batch_idx):
        recon_loss, kl_loss = self.forward(batch)
        beta = self._current_beta()
        total = recon_loss + beta * kl_loss
        self.log_dict({
            "val/loss": total,
            "val/recon": recon_loss,
            "val/kl": kl_loss,
            "val/beta": beta,
        }, prog_bar=True, on_step=False, on_epoch=True)
        return total

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=150, verbose=True, min_lr=self.lr * 0.001)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }


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

        pretrained_ckpt: str = None,
        w: float = 1.0,
        free_bits: float = 0.1,
        warmup_beta: bool = True,
        max_beta: float = 0.1,
        beta_warmup_steps: int = 5000,
        kl_annealing: bool = True,
        skip_pretrained_on_resume: bool = False,  # New parameter: whether to skip pretrained loading when resuming
        focal_gamma: float = 2.0,
        focal_alpha = None,
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
        
        # Load pretrained weights if provided - CRITICAL for good performance
        if pretrained_ckpt is not None:
            try:
                print(f"🔄 Loading pretrained weights from: {pretrained_ckpt}")
                state_dict = load_checkpoint_weights(pretrained_ckpt, device='cpu')
                
                # Load all compatible parameters except classifier head
                loaded_params = {}
                skipped_params = []
                
                for k, v in state_dict.items():
                    # Skip classifier head parameters (will be randomly initialized)
                    if k.startswith('cls_head'):
                        skipped_params.append(k)
                        continue
                    
                    # Try to load all other parameters
                    if k in self.state_dict():
                        if self.state_dict()[k].shape == v.shape:
                            loaded_params[k] = v
                        else:
                            print(f"⚠️ Shape mismatch for {k}: {self.state_dict()[k].shape} vs {v.shape}")
                            skipped_params.append(k)
                    else:
                        # Try to map keys for compatibility
                        mapped_key = None
                        if k.startswith('set_encoder.'):
                            mapped_key = 'setvae.setvae.' + k[len('set_encoder.'):]
                        elif k.startswith('setvae.'):
                            mapped_key = k
                        elif k.startswith('transformer.'):
                            mapped_key = k
                        elif k.startswith('post_transformer_norm.'):
                            mapped_key = k
                        elif k.startswith('decoder.'):
                            mapped_key = k
                        
                        if mapped_key and mapped_key in self.state_dict():
                            if self.state_dict()[mapped_key].shape == v.shape:
                                loaded_params[mapped_key] = v
                            else:
                                skipped_params.append(f"{k} -> {mapped_key}")
                        else:
                            skipped_params.append(k)
                
                # Load the compatible parameters
                missing, unexpected = self.load_state_dict(loaded_params, strict=False)
                
                print(f"✅ Loaded pretrained weights:")
                print(f"   - Successfully loaded: {len(loaded_params)} parameters")
                print(f"   - Missing: {len(missing)} parameters")
                print(f"   - Unexpected: {len(unexpected)} parameters")
                print(f"   - Skipped: {len(skipped_params)} parameters")
                
                if len(loaded_params) == 0:
                    print("❌ WARNING: No parameters were loaded! Check checkpoint compatibility.")
                    
            except Exception as e:
                print(f"❌ Failed to load pretrained weights: {e}")
                print("❌ This will significantly hurt performance!")
                raise e  # Don't continue with random initialization
        else:
            print("⚠️ WARNING: No pretrained checkpoint provided - using random initialization!")
            print("⚠️ This will significantly hurt finetune performance!")

        # Enhanced Transformer encoder with improved configuration for better AUC/AUPRC
        enc_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=0.15,  # Increased dropout for better regularization
            activation='gelu',  # Use GELU activation function
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)
        
        # Add layer normalization after transformer for better feature quality
        self.post_transformer_norm = nn.LayerNorm(latent_dim, eps=1e-6)

        # Robust time/position encoding (noise-tolerant):
        # - Absolute sinusoidal position (index-based)
        # - Relative time bucket embeddings for Δt between sets
        self.num_time_buckets = 64
        edges = torch.logspace(math.log10(0.5), math.log10(24 * 60.0), steps=self.num_time_buckets - 1)
        self.register_buffer("time_bucket_edges", edges, persistent=False)
        self.rel_time_bucket_embed = nn.Embedding(self.num_time_buckets, latent_dim)
        # ALiBi-like relative time bias parameters (shared across heads)
        self.alibi_slope = nn.Parameter(torch.tensor(1.0))
        self.time_tau = nn.Parameter(torch.tensor(60.0))

        # Decoder & Classifier with improved architecture
        self.decoder = _SetDecoder(
            latent_dim,
            reduced_dim,
            levels,
            heads,
            m,
        )
        
        # Simplified single-layer classification head (best practice for finetuning)
        # Research shows that simpler heads work better when backbone is frozen
        self.cls_head = nn.Linear(latent_dim * 2, num_classes)  # *2 for mean+var features
        
        # Initialize with small weights to prevent early saturation
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cls_head.bias)
        
        # Simplified feature processing for efficiency (removed complex fusion modules)
        # Complex feature fusion modules removed to improve training speed and reduce overfitting

        # Metrics
        task_type = "binary" if num_classes == 2 else "multiclass"
        self.val_auc = AUROC(task=task_type, num_classes=num_classes)
        self.val_auprc = AveragePrecision(task=task_type, num_classes=num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.num_classes = num_classes

        # Training hyperparameters
        self.w = w
        self.lr = lr
        self.free_bits = free_bits
        self.latent_dim = latent_dim
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.kl_annealing = kl_annealing
        self.current_step = 0
        # Always use focal loss for classification
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        # Finetune mode: classification only (skip recon/KL) and keep backbone eval
        self.classification_only = False
        self.cls_head_lr = None
        
        self.save_hyperparameters(ignore=["setvae"])
 
    def get_current_beta(self):
        """Calculate current beta value with support for warmup and annealing"""
        if not self.warmup_beta:
            return self.max_beta
            
        if self.current_step < self.beta_warmup_steps:
            # Linear warmup
            return self.max_beta * (self.current_step / self.beta_warmup_steps)
        else:
            return self.max_beta

    def _sinusoidal_positional_encoding(self, seq_len: int, dim: int, device: torch.device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _relative_time_bucket_embedding(self, minutes: torch.Tensor):
        # minutes: [B, S] (float, minutes)
        B, S = minutes.shape
        deltas = torch.cat([
            torch.zeros(B, 1, device=minutes.device, dtype=minutes.dtype),
            torch.diff(minutes, dim=1).clamp(min=0.0)
        ], dim=1)
        log_delta = torch.log1p(deltas)
        log_edges = torch.log1p(self.time_bucket_edges).to(log_delta.device)
        bucket_idx = torch.bucketize(log_delta, log_edges, right=False)
        bucket_idx = bucket_idx.clamp(max=self.num_time_buckets - 1)
        return self.rel_time_bucket_embed(bucket_idx)

    def _apply_positional_encoding(self, x: torch.Tensor, pos: torch.Tensor, padding_mask: torch.Tensor = None):
        B, S, D = x.shape
        pos_enc = self._sinusoidal_positional_encoding(S, D, x.device).unsqueeze(0).expand(B, -1, -1)
        rel_time = self._relative_time_bucket_embedding(pos)
        encoded = x + pos_enc + rel_time
        if padding_mask is not None:
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return encoded

    def _create_causal_mask(self, seq_len, device, padding_mask=None):
        """Create causal mask for variable length sequences with padding support"""
        # Create basic causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        
        # Avoid randomness in classification-only finetune to keep features stable
        if self.training and not self.classification_only:
            random_mask = torch.rand(seq_len, seq_len, device=device) < 0.1
            mask = mask & ~random_mask
        
        # Handle padding: padded positions should not attend to anything
        if padding_mask is not None:
            # padding_mask: [B, S] -> expand to [B, S, S]
            batch_size = padding_mask.size(0)
            expanded_padding = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            # If source is padded, it shouldn't attend to anything
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            mask = mask | expanded_padding
            # If target is padded, nothing should attend to it
            mask = mask | expanded_padding.transpose(-1, -2)
        
        return mask

    def _build_causal_time_bias_mask(self, minutes_1d: torch.Tensor):
        # minutes_1d: [S] (float minutes)
        S = minutes_1d.size(0)
        device = minutes_1d.device
        dt = (minutes_1d.unsqueeze(0) - minutes_1d.unsqueeze(1)).clamp(min=0.0)
        bias = -self.alibi_slope * torch.log1p(dt / self.time_tau)
        float_mask = bias.to(dtype=torch.float32)
        future = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        float_mask = float_mask.masked_fill(future, float("-inf"))
        return float_mask

    def forward(self, sets, padding_mask=None):
        """
        Optimized forward pass with improved batch processing efficiency.
        
        Args:
            sets: Either:
                  - List of patient sets, where each patient has a list of set dictionaries
                  - List of set dictionaries for single patient
                  - Dictionary with keys: 'var', 'val', 'minute', 'set_id', 'label', 'padding_mask'
            padding_mask: [B, S] boolean mask where True indicates padding (optional)
        
        Returns:
            logits: [B, num_classes] Classification logits
            recon_loss: Scalar reconstruction loss
            kl_loss: Scalar KL divergence loss
        """
        # Handle dictionary input (from dataloader) - OPTIMIZED VERSION
        if isinstance(sets, dict):
            var, val, time, set_ids, label = (
                sets["var"],
                sets["val"],
                sets["minute"],
                sets["set_id"],
                sets.get("label"),
            )
            padding_mask = sets.get("padding_mask", None)
            
            # OPTIMIZATION: Fast batch processing without patient-by-patient splitting
            batch_size = var.size(0)
            
            # Quick validation check (only in debug mode)
            if not self.training and hasattr(self, '_debug_mode'):
                all_patient_sets = self._split_sets(var, val, time, set_ids, padding_mask)
                return self._forward_batch_detailed(all_patient_sets)
            
            # FAST PATH: Direct batch processing for efficiency
            all_logits = []
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            
            for b in range(batch_size):
                # Extract patient data quickly
                patient_var = var[b]
                patient_val = val[b]
                patient_time = time[b]
                
                # Apply padding mask efficiently
                if padding_mask is not None:
                    valid_mask = ~padding_mask[b]
                    if valid_mask.any():
                        patient_var = patient_var[valid_mask]
                        patient_val = patient_val[valid_mask]
                        patient_time = patient_time[valid_mask]
                
                # Fast VAE encoding (skip set splitting for efficiency)
                if len(patient_var) > 0:
                    try:
                        # Direct encoding without set splitting
                        _, z_list, _ = self.setvae.setvae(patient_var.unsqueeze(0), patient_val.unsqueeze(0))
                        z_sample, mu, logvar = z_list[-1]
                        patient_features = mu.squeeze(1)  # [1, latent_dim]
                        
                        # Fast classification
                        patient_logits = self.cls_head(patient_features)
                        all_logits.append(patient_logits)
                        
                        # Skip reconstruction and KL in classification-only mode
                        if not self.classification_only:
                            kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
                            total_kl_loss += kl_div.mean()
                        
                    except Exception as e:
                        # Fallback to zero logits
                        device = var.device
                        all_logits.append(torch.zeros(1, self.num_classes, device=device))
                else:
                    # Empty patient
                    device = var.device
                    all_logits.append(torch.zeros(1, self.num_classes, device=device))
            
            # Combine results
            logits = torch.cat(all_logits, dim=0)  # [B, num_classes]
            recon_loss = torch.tensor(0.0, device=var.device)  # Skip in classification mode
            kl_loss = torch.tensor(0.0, device=var.device) if self.classification_only else total_kl_loss / batch_size
            
            return logits, recon_loss, kl_loss
        
        # Handle list input (original format)
        if isinstance(sets, list) and len(sets) > 0 and isinstance(sets[0], list):
            # Multi-patient batch case
            return self._forward_batch(sets, padding_mask)
        else:
            # Single patient case (backward compatibility)
            return self._forward_single(sets)

    def _forward_batch_detailed(self, all_patient_sets):
        """
        Detailed batch processing (fallback for debugging).
        This is the original complex logic, kept for compatibility.
        """
        all_logits = []
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        valid_patients = 0
        
        for patient_idx, patient_sets in enumerate(all_patient_sets):
            if len(patient_sets) == 0:
                continue
                
            try:
                logits, recon_loss, kl_loss = self._forward_single(patient_sets)
                all_logits.append(logits)
                total_recon_loss += recon_loss
                total_kl_loss += kl_loss
                valid_patients += 1
            except Exception as e:
                print(f"Error processing patient {patient_idx}: {e}")
                continue
        
        if valid_patients == 0:
            device = next(self.parameters()).device
            batch_size = len(all_patient_sets)
            logits = torch.zeros(batch_size, self.num_classes, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
        else:
            logits = torch.cat(all_logits, dim=0)
            recon_loss = total_recon_loss / valid_patients
            kl_loss = total_kl_loss / valid_patients
        
        return logits, recon_loss, kl_loss

    def _forward_single(self, sets):
        """
        Forward pass for single patient (original implementation).
        
        Args:
            sets: List of set dictionaries for one patient
            
        Returns:
            logits: [1, num_classes] Classification logits
            recon_loss: Scalar reconstruction loss
            kl_loss: Scalar KL divergence loss
        """
        # Ensure sets is a list
        if not isinstance(sets, list):
            raise ValueError(f"Expected sets to be a list, got {type(sets)}")
        
        S = len(sets)
        if S == 0:
            # Handle empty sets case
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            logits = torch.zeros(1, self.num_classes, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
            return logits, recon_loss, kl_loss
        
        z_prims, kl_total = [], 0.0
        pos_list = []
        
        # Get current beta value
        current_beta = self.get_current_beta()
        
        # Store latent variable information (for collapse detector)
        all_z_lists = []
        
        for i, s_dict in enumerate(sets):
            # Ensure s_dict is a dictionary
            if not isinstance(s_dict, dict):
                raise ValueError(f"Expected s_dict to be a dictionary, got {type(s_dict)} at index {i}")
            
            # Check if required keys exist
            required_keys = ["var", "val", "minute"]
            for key in required_keys:
                if key not in s_dict:
                    raise ValueError(f"Missing required key '{key}' in s_dict at index {i}")
            
            var, val, time = (
                s_dict["var"],
                s_dict["val"],
                s_dict["minute"],
            )
            
            # Ensure tensors are the right type
            if not isinstance(var, torch.Tensor):
                raise ValueError(f"Expected var to be a torch.Tensor, got {type(var)} at index {i}")
            if not isinstance(val, torch.Tensor):
                raise ValueError(f"Expected val to be a torch.Tensor, got {type(val)} at index {i}")
            if not isinstance(time, torch.Tensor):
                raise ValueError(f"Expected time to be a torch.Tensor, got {type(time)} at index {i}")
            
            assert time.unique().numel() == 1, "Time is not constant in this set"
            minute_val = time.unique().float()  # Ensure float type
            pos_list.append(minute_val)
            
            # Modern VAE feature extraction: use both mean and variance for richer representation
            _, z_list, _ = self.setvae.setvae(var, val)
            z_sample, mu, logvar = z_list[-1]
            
            # Extract and process mean and variance features
            mu_feat = mu.squeeze(1)  # [B, latent_dim]
            logvar_feat = logvar.squeeze(1)  # [B, latent_dim] - keep as logvar for numerical stability
            
            # Apply learnable VAE feature fusion
            combined_feat = self._fuse_vae_features(mu_feat, logvar_feat)
            
            z_prims.append(combined_feat)
            
            # Collect latent variable information (for collapse detector)
            all_z_lists.append(z_list)
            
            # KL calculation (skip in classification-only mode for speed)
            if not self.classification_only:
                kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
                min_kl = self.free_bits * self.latent_dim
                kl_div = torch.clamp(kl_div, min=min_kl)
                var_reg = -0.1 * torch.mean(logvar)
                kl_total += kl_div.mean() + var_reg
            
        kl_total = kl_total / S
        z_seq = torch.stack(z_prims, dim=1)  # [B, S, latent]
        pos_tensor = torch.stack(pos_list, dim=1)  # [B, S]
        
        # Apply positional and time encoding
        z_seq = self._apply_positional_encoding(z_seq, pos_tensor, None)
        
        # Add layer normalization
        z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])
        
        # Build causal + relative-time bias mask
        minutes_1d = pos_tensor.squeeze(0) if pos_tensor.dim() == 2 else pos_tensor.view(-1)
        attn_mask = self._build_causal_time_bias_mask(minutes_1d)
        h_seq = self.transformer(z_seq, mask=attn_mask)
        
        # Apply post-transformer normalization for better feature quality
        h_seq = self.post_transformer_norm(h_seq)
        
        # Classification-only finetune mode skips reconstruction entirely
        if self.classification_only:
            recon_loss_total = torch.tensor(0.0, device=h_seq.device)
        else:
            recon_loss_total = 0.0
            valid_sets = 0
            last_recon_list = []
            last_target_list = []
            for idx, s_dict in enumerate(sets):
                N_t = s_dict["var"].size(1)
                recon = self.decoder(h_seq[:, idx], N_t, noise_std=(0.0 if not self.training else 0.3))
                if self.setvae.setvae.dim_reducer is not None:
                    reduced = self.setvae.setvae.dim_reducer(s_dict["var"]) 
                else:
                    reduced = s_dict["var"]
                norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
                reduced_normalized = reduced / (norms + 1e-8)
                target_x = reduced_normalized * s_dict["val"]
                # store for visualization
                try:
                    last_recon_list.append(recon.detach())
                    last_target_list.append(target_x.detach())
                except Exception:
                    pass
                recon_loss_total += chamfer_recon_loss(recon, target_x)
                valid_sets += 1
            if valid_sets > 0:
                recon_loss_total /= valid_sets
            # expose concatenated tensors if possible
            try:
                self._last_recon_list = last_recon_list
                self._last_target_list = last_target_list
                self._last_recon_cat = torch.cat(last_recon_list, dim=1) if len(last_recon_list) > 0 else None
                self._last_target_cat = torch.cat(last_target_list, dim=1) if len(last_target_list) > 0 else None
            except Exception:
                self._last_recon_cat = None
                self._last_target_cat = None
        
        # Optimized feature extraction for finetune mode
        if self.classification_only:
            # Use VAE features directly for better performance in finetune mode
            # Aggregate VAE features from all timesteps using last timestep (most informative)
            if all_z_lists:
                # Get the last timestep's VAE features (most recent and informative)
                last_z_list = all_z_lists[-1]  # Features from last timestep
                if last_z_list:
                    _, mu, logvar = last_z_list[-1]  # Get the final layer's mu, logvar
                    mu_feat = mu.squeeze(1)  # [B, latent_dim]
                    logvar_feat = logvar.squeeze(1)  # [B, latent_dim]
                    enhanced_features = self._fuse_vae_features(mu_feat, logvar_feat)  # [B, latent_dim * 2]
                else:
                    # Fallback: use transformer output
                    enhanced_features = h_seq[:, -1, :]  # [B, latent_dim]
            else:
                # Fallback: use transformer output
                enhanced_features = h_seq[:, -1, :]  # [B, latent_dim]
        else:
            # Full training mode: use enhanced features
            enhanced_features = self._extract_enhanced_features(h_seq)
            # Fallback if needed
            if enhanced_features is None:
                attn_weights = F.softmax(torch.sum(h_seq * z_seq, dim=-1), dim=1)
                enhanced_features = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)
        
        # Classification with enhanced features
        logits = self.cls_head(enhanced_features)
        
        # Save latent variable information for collapse detector
        if all_z_lists:
            # Merge latent variable information from all sets (use first set as representative)
            self._last_z_list = all_z_lists[0] if all_z_lists else None
        
        return logits, recon_loss_total, (torch.tensor(0.0, device=h_seq.device) if self.classification_only else kl_total * current_beta)

    def _forward_batch(self, all_patient_sets, padding_mask=None):
        """
        Forward pass for multi-patient batch.
        
        Args:
            all_patient_sets: List of patient sets, where each patient has a list of set dictionaries
            padding_mask: [B, S] boolean mask where True indicates padding (optional)
            
        Returns:
            logits: [B, num_classes] Classification logits
            recon_loss: Scalar reconstruction loss
            kl_loss: Scalar KL divergence loss
        """
        batch_size = len(all_patient_sets)
        all_logits = []
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        valid_patients = 0
        
        for patient_idx, patient_sets in enumerate(all_patient_sets):
            if len(patient_sets) == 0:
                # Skip empty patients
                continue
                
            # Process this patient's sets
            patient_logits, patient_recon_loss, patient_kl_loss = self._forward_single(patient_sets)
            all_logits.append(patient_logits)
            total_recon_loss += patient_recon_loss
            total_kl_loss += patient_kl_loss
            valid_patients += 1
        
        if valid_patients == 0:
            # Handle case where all patients are empty
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            logits = torch.zeros(batch_size, self.num_classes, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
        else:
            # Concatenate logits and average losses
            logits = torch.cat(all_logits, dim=0)  # [B, num_classes]
            recon_loss = total_recon_loss / valid_patients
            kl_loss = total_kl_loss / valid_patients
        
        return logits, recon_loss, kl_loss

    def _extract_enhanced_features(self, h_t):
        """
        Extract enhanced features using multi-scale pooling and fusion
        for better AUC/AUPRC performance
        """
        B, S, D = h_t.shape
        
        # For classification-only mode, use simpler and more stable feature extraction
        if self.classification_only:
            # Use last token (most recent) with attention-weighted pooling
            last_token = h_t[:, -1, :]  # [B, D] - most recent representation
            
            # Attention-weighted pooling over all tokens
            attn_weights = F.softmax(
                torch.matmul(h_t, last_token.unsqueeze(-1)).squeeze(-1), 
                dim=1
            )  # [B, S]
            attn_pooled = torch.sum(h_t * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
            
            # Combine last token and attention pooling
            enhanced_features = 0.7 * last_token + 0.3 * attn_pooled
            return enhanced_features
        
        # For full training mode, use multi-scale pooling
        # Global average pooling
        global_avg = self.feature_fusion['global_pool'](h_t.transpose(1, 2)).squeeze(-1)  # [B, D]
        
        # Global max pooling
        global_max = self.feature_fusion['max_pool'](h_t.transpose(1, 2)).squeeze(-1)  # [B, D]
        
        # Attention-based pooling
        # Use mean pooling as query for attention
        query = h_t.mean(dim=1, keepdim=True)  # [B, 1, D]
        attn_output, _ = self.feature_fusion['attention_pool'](
            query, h_t, h_t, 
            attn_mask=None
        )
        attention_pool = attn_output.squeeze(1)  # [B, D]
        
        # Concatenate all pooling results
        combined_features = torch.cat([global_avg, global_max, attention_pool], dim=1)  # [B, 3*D]
        
        # Project to original dimension
        enhanced_features = self.feature_projection(combined_features)  # [B, D]
        
        return enhanced_features

    def _fuse_vae_features(self, mu, logvar):
        """
        Optimized VAE feature fusion for sequence-level prediction.
        Based on recent research: concatenating mean and variance features 
        works better than complex fusion for classification tasks.
        
        Args:
            mu: [B, latent_dim] - posterior mean
            logvar: [B, latent_dim] - posterior log variance
            
        Returns:
            fused_features: [B, latent_dim * 2] - concatenated mean and variance features
        """
        # Convert logvar to standard deviation for better numerical stability
        std = torch.exp(0.5 * logvar.clamp(-10, 10))  # Clamp for numerical stability
        
        # Simple concatenation works best for frozen backbone + simple classifier
        # This allows the classifier to learn optimal combination of mean and uncertainty
        fused_features = torch.cat([mu, std], dim=-1)  # [B, latent_dim * 2]
        
        return fused_features

    # Helpers
    def _split_sets(self, var, val, time, set_ids, padding_mask=None):
        """
        Split a concatenated patient sequence into list-of-set dicts.
        Supports both single-patient and multi-patient batches.
        
        Args:
            var: [B, N, D] Variable embeddings
            val: [B, N, 1] Values
            time: [B, N, 1] Time stamps
            set_ids: [B, N, 1] Set identifiers
            padding_mask: [B, N] Boolean mask where True indicates padding
            
        Returns:
            List of set dictionaries for each patient in the batch
        """
        batch_size = var.size(0)
        all_sets = []
        
        for b in range(batch_size):
            patient_sets = []
            
            # Get data for this patient
            patient_var = var[b]  # [N, D]
            patient_val = val[b]  # [N, 1]
            patient_time = time[b]  # [N, 1]
            patient_set_ids = set_ids[b]  # [N, 1]
            
            # Apply padding mask if provided
            if padding_mask is not None:
                patient_padding_mask = padding_mask[b]  # [N]
                # Only keep non-padded positions
                valid_indices = ~patient_padding_mask
                if not valid_indices.any():
                    # All positions are padded, create empty set
                    patient_sets.append({
                        "var": torch.empty(0, patient_var.size(-1), device=patient_var.device).unsqueeze(0),
                        "val": torch.empty(0, 1, device=patient_val.device).unsqueeze(0),
                        "minute": torch.empty(0, 1, device=patient_time.device).unsqueeze(0),
                    })
                    all_sets.append(patient_sets)
                    continue
                
                patient_var = patient_var[valid_indices]
                patient_val = patient_val[valid_indices]
                patient_time = patient_time[valid_indices]
                patient_set_ids = patient_set_ids[valid_indices]
            
            # Split into sets based on set_ids
            if len(patient_set_ids) == 0:
                # Empty patient, create empty set
                patient_sets.append({
                    "var": patient_var.unsqueeze(0),  # [1, N, D]
                    "val": patient_val.unsqueeze(0),  # [1, N, 1]
                    "minute": patient_time.unsqueeze(0),  # [1, N, 1]
                })
            else:
                # Ensure set_ids is the right shape and type
                if patient_set_ids.dim() > 1:
                    patient_set_ids = patient_set_ids.squeeze(-1)
                
                # Convert to long if needed
                if patient_set_ids.dtype != torch.long:
                    patient_set_ids = patient_set_ids.long()
                
                # Find unique consecutive set IDs
                try:
                    # Check if torch.unique_consecutive is available
                    if hasattr(torch, 'unique_consecutive'):
                        unique_set_ids, counts = torch.unique_consecutive(patient_set_ids, return_counts=True)
                    else:
                        # Fallback implementation for older PyTorch versions
                        unique_set_ids = torch.unique(patient_set_ids)
                        counts = torch.tensor([(patient_set_ids == uid).sum().item() for uid in unique_set_ids])
                    
                    # Convert counts to list of integers
                    counts_list = [int(c.item()) for c in counts]
                    
                    # Ensure counts_list is not empty and all values are positive
                    if not counts_list or any(c <= 0 for c in counts_list):
                        print(f"Warning: Invalid counts_list for patient {b}: {counts_list}. Treating as single set.")
                        patient_sets.append({
                            "var": patient_var.unsqueeze(0),  # [1, N, D]
                            "val": patient_val.unsqueeze(0),  # [1, N, 1]
                            "minute": patient_time.unsqueeze(0),  # [1, N, 1]
                        })
                    else:
                        indices = torch.split(torch.arange(len(patient_set_ids), device=patient_set_ids.device), counts_list)
                        
                        for idx in indices:
                            if len(idx) > 0:  # Only add non-empty sets
                                patient_sets.append({
                                    "var": patient_var[idx].unsqueeze(0),  # [1, set_size, D]
                                    "val": patient_val[idx].unsqueeze(0),  # [1, set_size, 1]
                                    "minute": patient_time[idx].unsqueeze(0),  # [1, set_size, 1]
                                })
                except Exception as e:
                    # Fallback: treat all data as one set
                    print(f"Warning: Error in set splitting for patient {b}: {e}. Treating as single set.")
                    patient_sets.append({
                        "var": patient_var.unsqueeze(0),  # [1, N, D]
                        "val": patient_val.unsqueeze(0),  # [1, N, 1]
                        "minute": patient_time.unsqueeze(0),  # [1, N, 1]
                    })
            
            all_sets.append(patient_sets)
        
        return all_sets

    def _step(self, batch, stage: str):
        """Optimized training/validation step using efficient forward pass"""
        label = batch.get("label")
        
        # OPTIMIZED: Use the efficient forward pass directly
        logits, recon_loss, kl_loss = self(batch)
        
        # Loss calculation: always use focal loss for classification
        pred_loss = self.focal_loss_fn(logits, label)

        if self.classification_only:
            # OPTIMIZED: Pure focal loss for finetune mode - no reconstruction or KL loss
            total_loss = pred_loss
            # Set other losses to zero for logging (but don't compute them)
            recon_loss = torch.tensor(0.0, device=pred_loss.device, requires_grad=False)
            kl_loss = torch.tensor(0.0, device=pred_loss.device, requires_grad=False)
            pred_weight = torch.tensor(1.0, device=pred_loss.device, requires_grad=False)
            recon_weight = torch.tensor(0.0, device=pred_loss.device, requires_grad=False)
        else:
            # Full training mode with reconstruction
            if stage == "train":
                progress = min(1.0, self.current_step / 5000)
                recon_weight = 0.8 * (1.0 - progress) + 0.3 * progress
                pred_weight = self.w * (0.5 + 0.5 * progress)
            else:
                recon_weight = 0.5
                pred_weight = self.w
            current_beta = self.get_current_beta()
            total_loss = pred_weight * pred_loss + recon_weight * recon_loss + current_beta * kl_loss

        # Metrics for validation stage
        if stage == "val":
            if self.num_classes == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.softmax(logits, dim=1)
            self.val_auc.update(probs, label)
            self.val_auprc.update(probs, label)
            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, label)

        # Log more useful metrics
        current_beta = self.get_current_beta()
        # Compute latent statistics if available
        mean_variance = None
        active_units_ratio = None
        if hasattr(self, '_last_z_list') and self._last_z_list:
            variances = []
            active_ratios = []
            for z_sample, mu, logvar in self._last_z_list:
                var = torch.exp(logvar)
                variances.append(var.mean())
                active = (var > 0.01).float().mean()
                active_ratios.append(active)
            if variances:
                mean_variance = torch.stack(variances).mean()
            if active_ratios:
                active_units_ratio = torch.stack(active_ratios).mean()
        # Simple logging: two modes only
        if self.classification_only:
            # Finetune mode: only focal loss and total loss
            log_payload = {
                f"{stage}/focal_loss": pred_loss,
                f"{stage}/total_loss": total_loss,
            }
            # Add VAE feature statistics for monitoring
            if hasattr(self, '_last_z_list') and self._last_z_list:
                try:
                    _, mu, logvar = self._last_z_list[-1]
                    mu_norm = torch.norm(mu, dim=-1).mean()
                    std_mean = torch.exp(0.5 * logvar).mean()
                    log_payload.update({
                        f"{stage}/mu_norm": mu_norm,
                        f"{stage}/std_mean": std_mean,
                    })
                except:
                    pass
        else:
            # Pretraining mode: log all losses including reconstruction and KL
            log_payload = {
                f"{stage}/focal_loss": pred_loss,
                f"{stage}/recon_loss": recon_loss,
                f"{stage}/kl_loss": kl_loss,
                f"{stage}/total_loss": total_loss,
                f"{stage}/pred_weight": pred_weight,
                f"{stage}/recon_weight": recon_weight,
                f"{stage}/beta": current_beta,
            }
            if mean_variance is not None:
                log_payload[f"{stage}/variance"] = mean_variance
            if active_units_ratio is not None:
                log_payload[f"{stage}/active_units"] = active_units_ratio
        self.log_dict(
            log_payload,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        
        # Store logged metrics for collapse detector (only in full training mode)
        if stage == "train" and not self.classification_only:
            self.logged_metrics = {
                'train_kl': kl_loss,
                'train_recon': recon_loss,
                'train_variance': mean_variance if mean_variance is not None else torch.tensor(0.0, device=kl_loss.device),
                'train_active_units': active_units_ratio if active_units_ratio is not None else torch.tensor(0.0, device=kl_loss.device),
            }
        
        # Always update current step
        if stage == "train":
            self.current_step += 1
            
        # Expose latent variables for collapse detector
        if hasattr(self, 'setvae') and hasattr(self.setvae, '_last_z_list'):
            self._last_z_list = self.setvae._last_z_list
            
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

    def on_after_backward(self):
        # Log global gradient norm after backward
        total_norm = None
        parameters = [p for p in self.parameters() if p.grad is not None]
        if parameters:
            device = parameters[0].grad.device
            norms = [torch.norm(p.grad.detach(), 2) for p in parameters]
            if norms:
                total_norm = torch.norm(torch.stack(norms), 2)
                self.log('grad_norm', total_norm, on_step=True, prog_bar=False)

    def configure_optimizers(self):
        if self.classification_only:
            # Optimize classifier head only with more conservative LR
            cls_params = [p for p in self.cls_head.parameters() if p.requires_grad]
            # Use more conservative learning rate for better stability
            cls_lr = self.cls_head_lr or (self.lr * 3.0)  # Reduced from 10x to 3x
            optimizer = AdamW(
                [{'params': cls_params, 'lr': cls_lr, 'name': 'cls_head'}],
                lr=cls_lr,
                betas=(0.9, 0.98),  # Slightly adjusted beta2 for better stability
                eps=1e-8,
                weight_decay=0.01,  # Increased weight decay for better regularization
            )
        else:
            # Set different learning rates for different parts
            setvae_params = list(self.setvae.parameters())
            transformer_params = list(self.transformer.parameters())
            other_params = [
                p for n, p in self.named_parameters() 
                if not any(n.startswith(prefix) for prefix in ['setvae', 'transformer'])
            ]
            param_groups = [
                {'params': setvae_params, 'lr': self.lr * 0.05, 'name': 'setvae'},
                {'params': transformer_params, 'lr': self.lr, 'name': 'transformer'},
                {'params': other_params, 'lr': self.lr, 'name': 'others'}
            ]
            optimizer = AdamW(
                param_groups,
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.02,
            )
        
        # Fixed learning rate scheduler to monitor AUC for better performance
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if self.classification_only:
            # For finetune mode, monitor AUC
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='max',  # Monitor AUC (higher is better)
                factor=0.8,  # More gentle reduction
                patience=4,  # Reduced patience for faster adaptation
                verbose=True,
                min_lr=1e-6  # Minimum learning rate
            )
            monitor_metric = "val_auc"
        else:
            # For full training mode, monitor loss
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',  # Monitor training loss (lower is better)
                factor=0.7,  # Reduce LR by 30% when plateau
                patience=200,  # Wait 200 steps before reducing LR
                verbose=True,
                min_lr=self.lr * 0.001  # Minimum learning rate
            )
            monitor_metric = "val_loss"
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Check every epoch instead of every step
                "frequency": 1,
                "monitor": monitor_metric,  # Monitor appropriate metric
            },
        }

    # -------- Finetune helpers --------
    def enable_classification_only_mode(self, cls_head_lr: Optional[float] = None):
        self.classification_only = True
        if cls_head_lr is not None:
            self.cls_head_lr = cls_head_lr
        self.set_backbone_eval()

    def set_backbone_eval(self):
        self.setvae.eval()
        self.transformer.eval()
        self.post_transformer_norm.eval()
        self.decoder.eval()
        # Simplified backbone eval (removed complex feature fusion modules)
 
    def on_train_start(self):
        if self.classification_only:
            self.set_backbone_eval()
 
    def init_classifier_head_xavier(self):
        """Initialize classifier head with Xavier uniform and zero biases."""
        if isinstance(self.cls_head, nn.Linear):
            # Single linear layer
            nn.init.xavier_uniform_(self.cls_head.weight, gain=1.0)
            if self.cls_head.bias is not None:
                nn.init.zeros_(self.cls_head.bias)
        else:
            # Sequential or other module types
            for module in self.cls_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
