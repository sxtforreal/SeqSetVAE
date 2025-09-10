import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import get_linear_schedule_with_warmup
from typing import Optional

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
        # Task C: next-step distributional forecast (here: next set has any new event?)
        enable_next_change: bool = True,
        next_change_weight: float = 0.3,
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
            enable_posterior_std_augmentation=False,
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
        self.enable_next_change = enable_next_change
        self.next_change_weight = next_change_weight
        # Manual LR scheduler stepping to guarantee order: optimizer.step() -> scheduler.step()
        self._manual_scheduler = None

        # Task C head: predict whether next set contains any new (non-carry) event
        if enable_next_change:
            self.next_change_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim, 1),
            )
            self._bce = nn.BCEWithLogitsLoss(reduction="mean")
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

    def _apply_set_mae(self, val: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Randomly mask a fraction of tokens within a set by zeroing their values.
        Expects `val` shape [1, N, 1] for a single set. No-op if not training or ratio<=0.
        """
        if mask_ratio <= 0.0 or not self.training or val.numel() == 0:
            return val
        token_probs = torch.rand(val.shape[:2], device=val.device)
        mask = (token_probs < mask_ratio).unsqueeze(-1)  # [1,N,1]
        return val.masked_fill(mask, 0.0)

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
    def _split_sets(self, var, val, minutes, set_id, padding_mask=None, carry_mask=None):
        B = var.size(0)
        all_sets = []
        for b in range(B):
            v = var[b]
            x = val[b]
            t = minutes[b]
            s = set_id[b]
            c = carry_mask[b] if carry_mask is not None else None
            if padding_mask is not None:
                mask = ~padding_mask[b]
                v, x, t, s = v[mask], x[mask], t[mask], s[mask]
                if c is not None:
                    c = c[mask]
            if s.dim() > 1:
                s = s.squeeze(-1)
            uniq, counts = torch.unique_consecutive(s.long(), return_counts=True)
            idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(c) for c in counts])
            patient_sets = []
            for idx in idx_splits:
                # has_change: any token in this set that is not carried (is_carry==0)
                if c is not None and len(idx) > 0:
                    has_change = (c[idx].squeeze(-1) < 0.5).any().float().view(1)
                else:
                    has_change = torch.tensor(0.0, device=v.device).view(1)
                patient_sets.append({
                    "var": v[idx].unsqueeze(0),
                    "val": x[idx].unsqueeze(0),
                    "minute": t[idx].unsqueeze(0),
                    "set_time": t[idx].max().view(1),
                    "has_change": has_change,  # 1 if any real event exists in this set
                })
            all_sets.append(patient_sets)
        return all_sets

    def _forward_single(self, sets, carry_mask=None, minutes_seq=None):
        S = len(sets)
        device = self.device
        if S == 0:
            z = torch.zeros(1, 0, self.latent_dim, device=device)
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        z_mu_list, z_logvar_list, pos_list = [], [], []
        z_mu_post_list, z_logvar_post_list = [], []
        kl_total = 0.0
        next_change_loss = torch.tensor(0.0, device=device)
        next_change_targets = []

        # encode per set: q_x(z|x)
        for i, s in enumerate(sets):
            var, val, minute = s["var"], s["val"], s["minute"]
            # Optional Set-MAE masking on input values
            val_inp = self._apply_set_mae(val, self.set_mae_ratio)
            # normalize and stale-dropout already handled upstream; here encode
            _, z_list, _ = self.set_encoder(var, val_inp)
            _, mu_qx, logvar_qx = z_list[-1]
            z_mu_list.append(mu_qx.squeeze(1))
            z_logvar_list.append(logvar_qx.squeeze(1))
            # use a single scalar time per set (e.g., last timestamp)
            pos_list.append(s["set_time"].float())

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
        kl = 0.5 * torch.sum(
            var_post / var_qx + (mu_post - z_mu) ** 2 / var_qx - 1 + (z_logvar - logvar_post),
            dim=-1,
        )  # [1,S]
        # Apply free-bits threshold per step
        min_kl = self.free_bits * self.latent_dim
        kl = torch.clamp(kl, min=min_kl)
        kl_total = kl.mean()

        # Task C: next-step change prediction using h_t
        if self.enable_next_change and S > 1:
            # Build targets from set t+1 has_change for t in [0..S-2]
            for idx in range(1, S):
                next_change_targets.append(sets[idx]["has_change"])  # shape [1]
            targets = torch.stack(next_change_targets, dim=1).to(h.dtype)  # [1, S-1]
            logits = self.next_change_head(h[:, :-1, :]).squeeze(-1)  # [1, S-1]
            next_change_loss = self._bce(logits, targets)

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
        return recon_total, kl_total, next_change_loss

    def forward(self, batch):
        var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        # stale-dropout on values
        val = self._apply_stale_dropout(val, carry_mask)
        # split into sets per patient
        all_sets = self._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
        total_recon, total_kl, total_c, count = 0.0, 0.0, 0.0, 0
        for sets in all_sets:
            recon, kl, next_c = self._forward_single(sets)
            total_recon += recon
            total_kl += kl
            total_c += next_c
            count += 1
        if count == 0:
            device = var.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero
        return total_recon / count, total_kl / count, total_c / count

    # training hooks
    def training_step(self, batch, batch_idx):
        recon, kl, next_c = self.forward(batch)
        beta = self._beta()
        total = recon + beta * kl + (self.next_change_weight * next_c if self.enable_next_change else 0.0)
        log_dict = {
            "train_loss": total,
            "train_recon": recon,
            "train_kl": kl,
            "train_beta": beta,
        }
        if self.enable_next_change:
            log_dict["train_next_change"] = next_c
        self.log_dict(log_dict, prog_bar=True)
        self._step += 1
        return total

    def validation_step(self, batch, batch_idx):
        recon, kl, next_c = self.forward(batch)
        beta = self._beta()
        total = recon + beta * kl + (self.next_change_weight * next_c if self.enable_next_change else 0.0)
        log_dict = {"val_loss": total, "val_recon": recon, "val_kl": kl, "val_beta": beta}
        if self.enable_next_change:
            log_dict["val_next_change"] = next_c
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)
        return total

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        # Create step-based scheduler but step it manually after actual optimizer steps
        self._manual_scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=2_000, num_training_steps=200_000
        )
        # Return only the optimizer to Lightning to avoid it stepping the scheduler early
        return opt

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Step scheduler ONLY when an optimizer step occurred (respect gradient accumulation)
        if self._manual_scheduler is None or self.trainer is None:
            return
        try:
            accumulate = int(getattr(self.trainer, "accumulate_grad_batches", 1))
        except Exception:
            accumulate = 1
        is_last_batch = False
        try:
            num_batches = int(getattr(self.trainer, "num_training_batches", 0) or 0)
            is_last_batch = (batch_idx + 1) >= num_batches and num_batches > 0
        except Exception:
            pass
        should_step = ((batch_idx + 1) % max(1, accumulate) == 0) or is_last_batch
        if should_step:
            self._manual_scheduler.step()


class SetVAEOnlyPretrain(pl.LightningModule):
    """
    SetVAE-only pretraining over LVCF-expanded sets.

    Objective:
      - Reconstruction: permutation-invariant Chamfer on x_target = normalize(var) * val
      - KL: q(z|x) to N(0, I), with free-bits and beta warmup

    Input perturbations (train only):
      - Value dropout: stronger on carry tokens (stale) p_stale; light on live tokens p_live
      - Token masking (Set-MAE): mask at most a small number of tokens per set
      - Additive value noise (Gaussian)
      - Small directional jitter on normalized variable vectors, then renormalize
      - Decoder noise during training
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
        warmup_beta: bool = True,
        max_beta: float = 0.2,
        beta_warmup_steps: int = 8000,
        free_bits: float = 0.05,
        # perturbation params
        p_stale: float = 0.5,
        p_live: float = 0.05,
        set_mae_ratio: float = 0.15,
        small_set_mask_prob: float = 0.4,
        small_set_threshold: int = 5,
        max_masks_per_set: int = 2,
        val_noise_std: float = 0.07,
        dir_noise_std: float = 0.01,
        train_decoder_noise_std: float = 0.3,
        eval_decoder_noise_std: float = 0.05,
    ):
        super().__init__()
        self.set_encoder = SetVAEModule(
            input_dim=input_dim,
            reduced_dim=reduced_dim,
            latent_dim=latent_dim,
            levels=levels,
            heads=heads,
            m=m,
        )

        # Training hparams
        self.lr = lr
        self.beta = beta
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.free_bits = free_bits
        self.latent_dim = latent_dim
        self._step = 0

        # Perturbation hparams
        self.p_stale = p_stale
        self.p_live = p_live
        self.set_mae_ratio = set_mae_ratio
        self.small_set_mask_prob = small_set_mask_prob
        self.small_set_threshold = small_set_threshold
        self.max_masks_per_set = max_masks_per_set
        self.val_noise_std = val_noise_std
        self.dir_noise_std = dir_noise_std
        self.train_decoder_noise_std = train_decoder_noise_std
        self.eval_decoder_noise_std = eval_decoder_noise_std

        # Manual LR scheduler stepping to guarantee order: optimizer.step() -> scheduler.step()
        self._manual_scheduler = None
        self.save_hyperparameters()

    # --- helpers ---
    def _beta(self):
        if not self.warmup_beta:
            return self.max_beta
        if self._step < self.beta_warmup_steps:
            return self.max_beta * (self._step / self.beta_warmup_steps)
        return self.max_beta

    def _split_sets(self, var, val, minutes, set_id, padding_mask=None, carry_mask=None):
        B = var.size(0)
        all_sets = []
        for b in range(B):
            v = var[b]
            x = val[b]
            t = minutes[b]
            s = set_id[b]
            c = carry_mask[b] if carry_mask is not None else None
            if padding_mask is not None:
                mask = ~padding_mask[b]
                v, x, t, s = v[mask], x[mask], t[mask], s[mask]
                if c is not None:
                    c = c[mask]
            if s.dim() > 1:
                s = s.squeeze(-1)
            uniq, counts = torch.unique_consecutive(s.long(), return_counts=True)
            idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(cn) for cn in counts])
            patient_sets = []
            for idx in idx_splits:
                patient_sets.append({
                    "var": v[idx].unsqueeze(0),
                    "val": x[idx].unsqueeze(0),
                    "minute": t[idx].unsqueeze(0),
                    "carry": (c[idx].unsqueeze(0) if c is not None else torch.zeros(1, len(idx), 1, device=v.device)),
                })
            all_sets.append(patient_sets)
        return all_sets

    def _compute_target(self, s):
        if self.set_encoder.dim_reducer is not None:
            reduced = self.set_encoder.dim_reducer(s["var"])  # [1,N,R]
        else:
            reduced = s["var"]
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        reduced_normalized = reduced / (norms + 1e-8)
        target_x = reduced_normalized * s["val"]
        return reduced_normalized, target_x

    def _apply_value_dropout(self, val: torch.Tensor, carry: torch.Tensor):
        if not self.training:
            return val
        # Bernoulli masks, independent per token
        stale_mask = (torch.rand_like(val) < self.p_stale) & (carry > 0.5)
        live_mask = (torch.rand_like(val) < self.p_live) & (carry <= 0.5)
        out = val.clone()
        out[stale_mask] = 0.0
        out[live_mask] = 0.0
        return out

    def _apply_set_mae_inplace(self, val: torch.Tensor, carry: torch.Tensor):
        """
        Mask a small number of tokens in-place.
        - For N <= small_set_threshold: with probability small_set_mask_prob, mask exactly 1 token
        - For N >  small_set_threshold: mask ceil(set_mae_ratio * N), capped by max_masks_per_set
        Ensure at least one non-carry token remains unmasked if possible.
        """
        if not self.training:
            return
        N = val.size(1)
        if N == 0:
            return
        device = val.device
        max_masks = self.max_masks_per_set
        if N <= self.small_set_threshold:
            if torch.rand((), device=device) < self.small_set_mask_prob and N >= 1:
                # choose 1 index
                idx = torch.randint(0, N, (1,), device=device)
                val[:, idx, :] = 0.0
        else:
            k = int(math.ceil(self.set_mae_ratio * float(N)))
            k = max(0, min(k, max_masks))
            if k > 0:
                idx = torch.randperm(N, device=device)[:k]
                val[:, idx, :] = 0.0
        # guarantee at least one non-carry token not zeroed if possible
        non_carry = (carry <= 0.5).squeeze(-1)
        if non_carry.any():
            # if all non-carry got zeroed, restore one
            mask_zero = (val.abs() <= 1e-8).squeeze(-1)
            if torch.all(mask_zero | (~non_carry)):
                indices = torch.nonzero(non_carry, as_tuple=False).squeeze(-1)
                pick = indices[torch.randint(0, len(indices), (1,), device=device)]
                # restore original magnitude to 1.0 scale (keep original val if available)
                val[:, pick, :] = 1.0

    def _forward_single(self, s):
        # target (clean)
        v_norm, x_target = self._compute_target(s)  # [1,N,D], [1,N,D]

        # build noisy input
        val_in = s["val"].clone()
        carry = s.get("carry", torch.zeros_like(val_in))
        # value dropout by stale/live
        val_in = self._apply_value_dropout(val_in, carry)
        # token masking (Set-MAE)
        self._apply_set_mae_inplace(val_in, carry)
        # additive value noise
        if self.training and self.val_noise_std > 0:
            val_in = val_in + self.val_noise_std * torch.randn_like(val_in)
        # directional jitter on variable vectors
        if self.training and self.dir_noise_std > 0:
            v_noisy = v_norm + self.dir_noise_std * torch.randn_like(v_norm)
            v_noisy = v_noisy / (torch.norm(v_noisy, p=2, dim=-1, keepdim=True) + 1e-8)
        else:
            v_noisy = v_norm
        x_input = v_noisy * val_in

        # encode/decode
        z_list, _ = self.set_encoder.encode(x_input)
        noise_std = self.train_decoder_noise_std if self.training else self.eval_decoder_noise_std
        recon = self.set_encoder.decode(z_list, target_n=x_target.size(1), noise_std=noise_std)

        # losses
        r_loss = chamfer_recon(recon, x_target)
        # KL to standard normal with free-bits
        # Reuse base_elbo to compute KL terms (we discard its recon to avoid double compute)
        _, kl = base_elbo(recon, x_target, z_list, free_bits=self.free_bits)
        return r_loss, kl

    def forward(self, batch):
        var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        all_sets = self._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
        total_recon, total_kl, count = 0.0, 0.0, 0
        for sets in all_sets:
            for s in sets:
                recon, kl = self._forward_single(s)
                total_recon += recon
                total_kl += kl
                count += 1
        if count == 0:
            device = var.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero
        return total_recon / count, total_kl / count

    # training hooks
    def training_step(self, batch, batch_idx):
        recon, kl = self.forward(batch)
        beta = self._beta()
        total = recon + beta * kl
        self.log_dict({"train_loss": total, "train_recon": recon, "train_kl": kl, "train_beta": beta}, prog_bar=True)
        self._step += 1
        return total

    def validation_step(self, batch, batch_idx):
        recon, kl = self.forward(batch)
        beta = self._beta()
        total = recon + beta * kl
        self.log_dict({"val_loss": total, "val_recon": recon, "val_kl": kl, "val_beta": beta}, prog_bar=True, on_epoch=True)
        return total

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        self._manual_scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=2_000, num_training_steps=200_000
        )
        return opt

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._manual_scheduler is None or self.trainer is None:
            return
        try:
            accumulate = int(getattr(self.trainer, "accumulate_grad_batches", 1))
        except Exception:
            accumulate = 1
        is_last_batch = False
        try:
            num_batches = int(getattr(self.trainer, "num_training_batches", 0) or 0)
            is_last_batch = (batch_idx + 1) >= num_batches and num_batches > 0
        except Exception:
            pass
        should_step = ((batch_idx + 1) % max(1, accumulate) == 0) or is_last_batch
        if should_step:
            self._manual_scheduler.step()
