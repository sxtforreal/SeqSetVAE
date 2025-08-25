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
        diffs = (minutes[:, 1:] - minutes[:, :-1]).clamp(min=0.0)
        deltas = torch.cat([
            torch.zeros(B, 1, device=minutes.device, dtype=minutes.dtype),
            diffs
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
        # PRETRAIN MODE: Only reconstruction + KL loss, NO focal loss
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
        # PRETRAIN MODE: Only reconstruction + KL loss, NO focal loss
        total = recon_loss + beta * kl_loss
        self.log_dict({
            "val_loss": total,
            "val_recon": recon_loss,
            "val_kl": kl_loss,
            "val_beta": beta,
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
                "monitor": "val_loss",
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
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha = None,
        medical_scenario: str = "multi_condition_screening",  # SOTA: Medical scenario for loss strategy
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
                
                # Enhanced parameter mapping with intelligent prefix/suffix handling
                loaded_params = {}
                skipped_params = []
                mapping_stats = {
                    'exact_match': 0,
                    'prefix_mapped': 0,
                    'shape_mismatch': 0,
                    'not_found': 0
                }
                
                for k, v in state_dict.items():
                    # Skip classifier head parameters (will be randomly initialized)
                    if k.startswith('cls_head'):
                        skipped_params.append(f"{k} (classifier head - will be reinitialized)")
                        continue
                    
                    # Try exact match first
                    if k in self.state_dict():
                        if self.state_dict()[k].shape == v.shape:
                            loaded_params[k] = v
                            mapping_stats['exact_match'] += 1
                            continue
                        else:
                            print(f"⚠️ Shape mismatch for {k}: {self.state_dict()[k].shape} vs {v.shape}")
                            skipped_params.append(f"{k} (shape mismatch)")
                            mapping_stats['shape_mismatch'] += 1
                            continue
                    
                    # Try intelligent parameter name mapping for different checkpoint formats
                    mapped_key = None
                    mapping_type = None
                    
                    # Handle different prefixes from different training modes
                    if k.startswith('set_encoder.'):
                        mapped_key = 'setvae.setvae.' + k[len('set_encoder.'):]
                        mapping_type = "set_encoder -> setvae.setvae"
                    elif k.startswith('setvae.setvae.') and not k.startswith('setvae.setvae.setvae.'):
                        mapped_key = k  # Already correctly prefixed
                        mapping_type = "setvae.setvae (correct prefix)"
                    elif k.startswith('setvae.') and not k.startswith('setvae.setvae.'):
                        mapped_key = k  # setvae.* parameters
                        mapping_type = "setvae prefix"
                    elif k.startswith('transformer.'):
                        mapped_key = k
                        mapping_type = "transformer"
                    elif k.startswith('post_transformer_norm.'):
                        mapped_key = k
                        mapping_type = "post_transformer_norm"
                    elif k.startswith('decoder.'):
                        mapped_key = k
                        mapping_type = "decoder"
                    else:
                        # Try removing common prefixes that might exist in some checkpoints
                        for prefix in ['model.', 'module.', 'net.']:
                            if k.startswith(prefix):
                                candidate_key = k[len(prefix):]
                                if candidate_key in self.state_dict():
                                    mapped_key = candidate_key
                                    mapping_type = f"removed {prefix} prefix"
                                    break
                    
                    # Try to load the mapped parameter
                    if mapped_key and mapped_key in self.state_dict():
                        if self.state_dict()[mapped_key].shape == v.shape:
                            loaded_params[mapped_key] = v
                            mapping_stats['prefix_mapped'] += 1
                            print(f"✅ Mapped: {k} -> {mapped_key} ({mapping_type})")
                        else:
                            skipped_params.append(f"{k} -> {mapped_key} (shape mismatch: {self.state_dict()[mapped_key].shape} vs {v.shape})")
                            mapping_stats['shape_mismatch'] += 1
                    else:
                        skipped_params.append(f"{k} (no compatible parameter found)")
                        mapping_stats['not_found'] += 1
                
                # Load the compatible parameters
                missing, unexpected = self.load_state_dict(loaded_params, strict=False)
                
                print(f"✅ Pretrained weight loading summary:")
                print(f"   📊 Parameter mapping statistics:")
                print(f"      - Exact matches: {mapping_stats['exact_match']}")
                print(f"      - Prefix mapped: {mapping_stats['prefix_mapped']}")
                print(f"      - Shape mismatches: {mapping_stats['shape_mismatch']}")
                print(f"      - Not found: {mapping_stats['not_found']}")
                print(f"   📥 Loading results:")
                print(f"      - Successfully loaded: {len(loaded_params)} parameters")
                print(f"      - Missing in checkpoint: {len(missing)} parameters")
                print(f"      - Unexpected in checkpoint: {len(unexpected)} parameters")
                print(f"      - Skipped: {len(skipped_params)} parameters")
                
                # Show some examples of skipped parameters for debugging
                if skipped_params:
                    print(f"   ⚠️ Examples of skipped parameters:")
                    for param in skipped_params[:5]:  # Show first 5
                        print(f"      - {param}")
                    if len(skipped_params) > 5:
                        print(f"      ... and {len(skipped_params) - 5} more")
                
                if len(loaded_params) == 0:
                    print("❌ WARNING: No parameters were loaded! Check checkpoint compatibility.")
                    print("💡 Possible issues:")
                    print("   - Checkpoint format doesn't match expected parameter names")
                    print("   - Checkpoint is from a different model architecture")
                    print("   - Parameter name prefixes have changed")
                    raise ValueError("No pretrained parameters were loaded - this will hurt performance!")
                elif len(loaded_params) < 10:  # Arbitrary threshold for "too few" parameters
                    print(f"⚠️ Warning: Only {len(loaded_params)} parameters loaded. This might indicate compatibility issues.")
                else:
                    print(f"🎉 Successfully loaded {len(loaded_params)} pretrained parameters!")
                    
            except Exception as e:
                print(f"❌ Failed to load pretrained weights: {e}")
                print("❌ This will significantly hurt performance!")
                print("💡 Debugging tips:")
                print(f"   - Check if checkpoint file exists: {pretrained_ckpt}")
                print("   - Verify checkpoint format compatibility")
                print("   - Check if checkpoint contains expected parameter names")
                raise e  # Don't continue with random initialization
        else:
            raise ValueError("❌ Pretrained checkpoint is required for finetune mode!")

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
        
        # Advanced classification head for better AUC/AUPRC performance
        self.cls_head = self._build_advanced_classifier(latent_dim, num_classes)
        
        # Learnable gate for pooling between last_token and attention pooling
        self.pooling_gate = nn.Parameter(torch.tensor(0.7))
        
        # Simplified feature extraction - no complex fusion modules for better stability
        # Use simple but effective pooling strategy

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
        self.use_focal_loss = use_focal_loss
        self.focal_loss_fn = (
            FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
            if use_focal_loss
            else None
        )
        # Finetune mode: classification only (skip recon/KL) and keep backbone eval
        self.classification_only = False
        self.cls_head_lr = None
        self.medical_scenario = medical_scenario  # Store for SOTA loss strategy
        
        self.save_hyperparameters(ignore=["setvae"])
 
    def _build_advanced_classifier(self, latent_dim: int, num_classes: int):
        """
        Build an advanced classification head with attention mechanisms and residual connections
        for improved AUC/AUPRC performance on medical data.
        """
        hidden_dim = latent_dim
        intermediate_dim = latent_dim // 2
        
        return nn.ModuleDict({
            # Feature enhancement layers
            'feature_enhancer': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
            ),
            
            # Self-attention for feature refinement
            'self_attention': nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            
            # Residual connections and normalization
            'attention_norm': nn.LayerNorm(hidden_dim),
            'attention_dropout': nn.Dropout(0.1),
            
            # Advanced feature fusion with multiple pathways
            'pathway1': nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            ),
            
            'pathway2': nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ),
            
            # Feature gate for pathway combination
            'gate': nn.Sequential(
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            ),
            
            # Final classification layers with residual connection
            'pre_classifier': nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim // 2),
                nn.LayerNorm(intermediate_dim // 2),
                nn.GELU(),
                nn.Dropout(0.15),
            ),
            
            'classifier': nn.Linear(intermediate_dim // 2, num_classes),
            
            # Auxiliary prediction head for better gradient flow
            'aux_classifier': nn.Linear(hidden_dim, num_classes),
        })
    
    def _forward_advanced_classifier(self, features):
        """Forward pass through the advanced classifier"""
        # Feature enhancement
        enhanced_features = self.cls_head['feature_enhancer'](features)
        
        # Self-attention for feature refinement
        # Add batch and sequence dimensions for attention
        if enhanced_features.dim() == 2:
            attn_input = enhanced_features.unsqueeze(1)  # [B, 1, D]
        else:
            attn_input = enhanced_features
            
        attn_output, _ = self.cls_head['self_attention'](
            attn_input, attn_input, attn_input
        )
        
        # Residual connection + normalization
        attn_output = attn_output.squeeze(1) if attn_output.dim() == 3 else attn_output
        enhanced_features = self.cls_head['attention_norm'](
            enhanced_features + self.cls_head['attention_dropout'](attn_output)
        )
        
        # Multi-pathway feature processing
        pathway1_out = self.cls_head['pathway1'](enhanced_features)
        pathway2_out = self.cls_head['pathway2'](enhanced_features)
        
        # Gated feature combination
        gate_weights = self.cls_head['gate'](enhanced_features)
        combined_features = (
            gate_weights[:, 0:1] * pathway1_out + 
            gate_weights[:, 1:2] * pathway2_out
        )
        
        # Final classification
        pre_logits = self.cls_head['pre_classifier'](combined_features)
        main_logits = self.cls_head['classifier'](pre_logits)
        
        # Auxiliary prediction for better gradient flow during training
        aux_logits = self.cls_head['aux_classifier'](enhanced_features)
        
        return main_logits, aux_logits

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
        diffs = (minutes[:, 1:] - minutes[:, :-1]).clamp(min=0.0)
        deltas = torch.cat([
            torch.zeros(B, 1, device=minutes.device, dtype=minutes.dtype),
            diffs
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
        Forward pass with support for variable length sequences and batch processing.
        
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
        # Handle dictionary input (from dataloader)
        if isinstance(sets, dict):
            var, val, time, set_ids, label = (
                sets["var"],
                sets["val"],
                sets["minute"],
                sets["set_id"],
                sets.get("label"),
            )
            padding_mask = sets.get("padding_mask", None)
            
            # Split sets for each patient in the batch
            all_patient_sets = self._split_sets(var, val, time, set_ids, padding_mask)
            
            # Process each patient separately and collect results
            all_logits = []
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            valid_patients = 0
            
            for patient_idx, patient_sets in enumerate(all_patient_sets):
                if len(patient_sets) == 0:
                    # Skip empty patients
                    continue
                
                # Ensure patient_sets is a list of dictionaries
                if not isinstance(patient_sets, list):
                    print(f"Warning: patient_sets is not a list for patient {patient_idx}, got {type(patient_sets)}")
                    continue
                
                # Check if all elements are dictionaries
                for i, s_dict in enumerate(patient_sets):
                    if not isinstance(s_dict, dict):
                        print(f"Warning: s_dict at index {i} is not a dictionary for patient {patient_idx}, got {type(s_dict)}")
                        continue
                    
                    # Check if required keys exist
                    required_keys = ["var", "val", "minute"]
                    for key in required_keys:
                        if key not in s_dict:
                            print(f"Warning: Missing key '{key}' in s_dict at index {i} for patient {patient_idx}")
                            continue
                    
                    # Check if values are tensors
                    for key in required_keys:
                        if not isinstance(s_dict[key], torch.Tensor):
                            print(f"Warning: {key} is not a tensor in s_dict at index {i} for patient {patient_idx}, got {type(s_dict[key])}")
                            continue
                
                # Process this patient's sets
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
                # Handle case where all patients are empty
                batch_size = var.size(0)
                device = var.device
                logits = torch.zeros(batch_size, self.num_classes, device=device)
                recon_loss = torch.tensor(0.0, device=device)
                kl_loss = torch.tensor(0.0, device=device)
            else:
                # Average losses across patients
                logits = torch.cat(all_logits, dim=0)  # [B, num_classes]
                recon_loss = total_recon_loss / valid_patients
                kl_loss = total_kl_loss / valid_patients
            
            return logits, recon_loss, kl_loss
        
        # Handle list input (original format)
        if isinstance(sets, list) and len(sets) > 0 and isinstance(sets[0], list):
            # Multi-patient batch case
            return self._forward_batch(sets, padding_mask)
        else:
            # Single patient case (backward compatibility)
            return self._forward_single(sets)

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
            if self.classification_only:
                setvae_inner = self.setvae.setvae
                if hasattr(setvae_inner, "encode_from_var_val"):
                    z_list, _ = setvae_inner.encode_from_var_val(var, val)
                else:
                    # Fallback for older checkpoints/modules without encode_from_var_val
                    _, z_list, _ = setvae_inner(var, val)
            else:
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
            
            # KL calculation (skip entirely in classification-only finetune)
            if not self.classification_only:
                kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
                min_kl = self.free_bits * self.latent_dim
                kl_div = torch.clamp(kl_div, min=min_kl)
                var_reg = -0.1 * torch.mean(logvar)
                kl_total += kl_div.mean() + var_reg
        
        if not self.classification_only:
            kl_total = kl_total / S
        else:
            kl_total = torch.tensor(0.0, device=z_prims[0].device)
        
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
        
        # Enhanced feature extraction using multi-scale pooling
        enhanced_features = self._extract_enhanced_features(h_seq)
        
        # Alternative: Use attention-based pooling as fallback (deterministic)
        if enhanced_features is None:
            attn_weights = F.softmax(torch.sum(h_seq * z_seq, dim=-1), dim=1)
            enhanced_features = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)
        
        # Classification with advanced classifier
        main_logits, aux_logits = self._forward_advanced_classifier(enhanced_features)
        # Use main logits for primary predictions
        logits = main_logits
        
        # Save latent variable information for collapse detector
        if all_z_lists:
            # Merge latent variable information from all sets (use first set as representative)
            self._last_z_list = all_z_lists[0] if all_z_lists else None
        
        return logits, recon_loss_total, (torch.tensor(0.0, device=h_seq.device) if self.classification_only else kl_total * current_beta), aux_logits

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
        Simple and effective feature extraction for sequence-level classification.
        Uses attention-weighted pooling for better representation.
        """
        B, S, D = h_t.shape
        
        # Use last token (most recent) as primary representation
        last_token = h_t[:, -1, :]  # [B, D] - most recent representation
        
        # Attention-weighted pooling over all tokens for context
        # Compute attention weights using last token as query
        attn_weights = F.softmax(
            torch.matmul(h_t, last_token.unsqueeze(-1)).squeeze(-1), 
            dim=1
        )  # [B, S]
        
        # Compute attention-pooled representation
        attn_pooled = torch.sum(h_t * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        # Learnable gate g in [0,1]
        g = torch.sigmoid(self.pooling_gate)
        enhanced_features = g * last_token + (1.0 - g) * attn_pooled
        
        return enhanced_features

    def _fuse_vae_features(self, mu, logvar):
        """
        Simple and effective VAE feature fusion for sequence-level classification.
        Combines mean and variance information for better representation learning.
        
        Args:
            mu: [B, latent_dim] - posterior mean
            logvar: [B, latent_dim] - posterior log variance
            
        Returns:
            fused_features: [B, latent_dim] - enhanced features combining mean and uncertainty
        """
        # Convert logvar to variance for interpretability
        var = torch.exp(logvar)  # [B, latent_dim]
        
        # Simple but effective fusion strategy:
        # 1. Use mean as the primary representation
        # 2. Modulate with variance information for uncertainty awareness
        
        # Compute uncertainty score (higher variance = higher uncertainty)
        uncertainty = torch.mean(var, dim=-1, keepdim=True)  # [B, 1]
        
        # Normalize uncertainty to [0, 1] range for stable training
        uncertainty_normalized = torch.sigmoid(uncertainty - 1.0)
        
        # Create variance-aware features
        # High uncertainty -> rely more on mean, low uncertainty -> allow variance modulation
        variance_scale = 1.0 + 0.1 * (1.0 - uncertainty_normalized) * torch.tanh(var)
        
        # Final fused features: mean modulated by variance information
        fused_features = mu * variance_scale
        
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
        var, val, time, set_ids, label = (
            batch["var"],
            batch["val"],
            batch["minute"],
            batch["set_id"],
            batch.get("label"),
        )
        
        # Get padding mask if available
        padding_mask = batch.get("padding_mask", None)
        
        # Split sets for each patient in the batch
        all_patient_sets = self._split_sets(var, val, time, set_ids, padding_mask)
        
        # Process each patient separately and collect results
        all_logits = []
        all_aux_logits = []
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        valid_patients = 0
        
        for patient_sets in all_patient_sets:
            if len(patient_sets) == 0:
                # Skip empty patients
                continue
                
            # Process this patient's sets
            logits, recon_loss, kl_loss, aux_logits = self(patient_sets)
            all_logits.append(logits)
            all_aux_logits.append(aux_logits)
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            valid_patients += 1
        
        if valid_patients == 0:
            # Handle case where all patients are empty
            batch_size = var.size(0)
            device = var.device
            logits = torch.zeros(batch_size, self.num_classes, device=device)
            aux_logits = torch.zeros(batch_size, self.num_classes, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
        else:
            # Average losses across patients
            logits = torch.cat(all_logits, dim=0)  # [B, num_classes]
            aux_logits = torch.cat(all_aux_logits, dim=0)  # [B, num_classes]
            recon_loss = total_recon_loss / valid_patients
            kl_loss = total_kl_loss / valid_patients
        
        # 🏆 SOTA Loss Strategy: 基于2024年最新学术研究
        # 集成多项前沿技术：SoftAdapt, Asymmetric Loss, Self-Distillation, Gradient Adaptation
        
        # 初始化SOTA损失策略 (懒加载，仅在第一次调用时创建)
        if not hasattr(self, '_sota_loss_strategy'):
            from sota_loss_strategies import get_sota_loss_strategy
            # 根据医疗场景选择最优策略
            medical_scenario = getattr(self, 'medical_scenario', 'multi_condition_screening')
            self._sota_loss_strategy = get_sota_loss_strategy(
                medical_scenario=medical_scenario,
                num_classes=self.num_classes
            )
            print(f"🔬 Initialized SOTA loss strategy for: {medical_scenario}")
        
        # 获取当前训练步骤作为epoch近似
        current_epoch = getattr(self, 'current_step', 0) // 100  # 假设每100步为一个epoch
        
        # 收集主头和辅助头的参数用于梯度分析
        main_params = list(self.cls_head['classifier'].parameters()) + list(self.cls_head['pre_classifier'].parameters())
        aux_params = list(self.cls_head['aux_classifier'].parameters())
        
        try:
            # 🚀 使用SOTA损失策略计算损失
            pred_loss, loss_breakdown = self._sota_loss_strategy.compute_loss(
                main_logits=logits,
                aux_logits=aux_logits,
                labels=label,
                epoch=current_epoch,
                main_params=main_params,
                aux_params=aux_params
            )
            
            # 存储详细的损失分解用于监控
            if stage == "train":
                for key, value in loss_breakdown.items():
                    if isinstance(value, torch.Tensor):
                        setattr(self, f'_last_{key}', value.detach())
            
        except Exception as e:
            # 降级到简化策略（确保训练不会中断）
            print(f"⚠️ SOTA loss computation failed, falling back to simplified strategy: {e}")
            
            # 主损失：Focal Loss
            if self.focal_loss_fn is not None:
                main_pred_loss = self.focal_loss_fn(logits, label)
            else:
                main_pred_loss = F.cross_entropy(logits, label, label_smoothing=0.1)
                
            # 辅助损失：不对称损失 (处理极端不平衡)
            try:
                from sota_loss_strategies import AsymmetricLoss
                if not hasattr(self, '_asymmetric_loss'):
                    self._asymmetric_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
                aux_pred_loss = self._asymmetric_loss(aux_logits, label)
            except:
                aux_pred_loss = F.cross_entropy(aux_logits, label, label_smoothing=0.15)
            
            # EMA自蒸馏损失
            if not hasattr(self, '_ema_teacher'):
                self._ema_teacher = F.softmax(logits.detach(), dim=1)
            else:
                self._ema_teacher = 0.999 * self._ema_teacher + 0.001 * F.softmax(logits.detach(), dim=1)
            
            distill_loss = F.kl_div(
                F.log_softmax(aux_logits / 3.0, dim=1),
                self._ema_teacher / 3.0,
                reduction='batchmean'
            ) * 9.0  # temperature^2 scaling
            
            # 动态权重组合
            main_weight = 0.6
            aux_weight = 0.3  
            distill_weight = 0.1
            
            pred_loss = main_weight * main_pred_loss + aux_weight * aux_pred_loss + distill_weight * distill_loss

        # FINETUNE MODE: Only classification loss, no recon/KL loss
        total_loss = pred_loss
        pred_weight = torch.tensor(1.0, device=pred_loss.device)
        recon_weight = torch.tensor(0.0, device=pred_loss.device)
        
        # Zero out recon and KL losses for finetune mode
        recon_loss = torch.tensor(0.0, device=pred_loss.device)
        kl_loss = torch.tensor(0.0, device=pred_loss.device)

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
        log_payload = {
                f"{stage}_loss": total_loss,
                f"{stage}_recon": recon_loss,
                f"{stage}_kl": kl_loss,
                f"{stage}_pred": pred_loss,
                f"{stage}_class_loss": pred_loss,
                f"{stage}_beta": current_beta,
                f"{stage}_recon_weight": recon_weight,
                f"{stage}_pred_weight": pred_weight,
            }
        
        # Add detailed loss breakdown for better monitoring
        if hasattr(self, 'classification_only') and self.classification_only:
            # Extract individual loss components for monitoring
            if 'main_pred_loss' in locals():
                log_payload[f"{stage}_main_loss"] = main_pred_loss
            if 'aux_pred_loss' in locals():
                log_payload[f"{stage}_aux_loss"] = aux_pred_loss  
            if 'consistency_loss' in locals():
                log_payload[f"{stage}_consistency_loss"] = consistency_loss
        if mean_variance is not None:
            log_payload[f"{stage}_variance"] = mean_variance
        if active_units_ratio is not None:
            log_payload[f"{stage}_active_units"] = active_units_ratio
        self.log_dict(
            log_payload,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        
        # Store logged metrics for collapse detector
        if stage == "train":
            self.logged_metrics = {
                'train_kl': kl_loss,
                'train_recon': recon_loss,
                'train_variance': mean_variance if mean_variance is not None else torch.tensor(0.0, device=kl_loss.device),
                'train_active_units': active_units_ratio if active_units_ratio is not None else torch.tensor(0.0, device=kl_loss.device),
            }
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
            # Enhanced optimizer for advanced classifier architecture
            cls_params = []
            
            # Collect parameters from advanced classifier with different LR groups
            attention_params = []
            classifier_params = []
            
            for name, module in self.cls_head.items():
                if 'attention' in name:
                    attention_params.extend(list(module.parameters()))
                else:
                    classifier_params.extend(list(module.parameters()))
            
            # Use differentiated learning rates for different components
            cls_lr = self.cls_head_lr or (self.lr * 4.0)  # Higher LR for complex architecture
            
            param_groups = [
                {
                    'params': attention_params, 
                    'lr': cls_lr * 0.8,  # Slightly lower LR for attention layers
                    'weight_decay': 0.005,
                    'name': 'attention'
                },
                {
                    'params': classifier_params, 
                    'lr': cls_lr,
                    'weight_decay': 0.01,
                    'name': 'classifier'
                }
            ]
            
            optimizer = AdamW(
                param_groups,
                betas=(0.9, 0.999),  # Standard AdamW betas
                eps=1e-8,
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
        
        # Enhanced learning rate scheduler for advanced architecture
        if self.classification_only:
            # Use cosine annealing with warm restarts for better convergence
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=500,  # Initial restart period
                T_mult=2,  # Multiply restart period by this factor
                eta_min=cls_lr * 0.001,  # Minimum learning rate
                verbose=True
            )
        else:
            # Standard scheduler for pretraining
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.7,
                patience=200,
                verbose=True,
                min_lr=self.lr * 0.001
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" if self.classification_only else "epoch",  # Step-wise for advanced training
                "frequency": 1,
                "monitor": "val_auc" if self.classification_only else "val_loss",  # Monitor AUC for finetune
            },
        }

    # -------- Finetune helpers --------
    def enable_classification_only_mode(self, cls_head_lr: Optional[float] = None):
        self.classification_only = True
        if cls_head_lr is not None:
            self.cls_head_lr = cls_head_lr
        self.set_backbone_eval()

    def set_backbone_eval(self):
        """Set all backbone components to eval mode for finetune"""
        self.setvae.eval()
        self.transformer.eval()
        self.post_transformer_norm.eval()
        self.decoder.eval()
 
    def on_train_start(self):
        if self.classification_only:
            self.set_backbone_eval()
 
    def init_classifier_head_xavier(self):
        """Initialize classifier head Linear layers with Xavier uniform and zero biases."""
        def init_linear_layers(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize all linear layers in the advanced classifier
        for key, module in self.cls_head.items():
            if isinstance(module, nn.Sequential):
                for sub_module in module:
                    init_linear_layers(sub_module)
            elif isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                init_linear_layers(module)
            elif hasattr(module, 'modules'):
                for sub_module in module.modules():
                    init_linear_layers(sub_module)
