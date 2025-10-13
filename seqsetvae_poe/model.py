import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Optional

from modules import SetVAEModule, AttentiveBottleneckLayer, elbo_loss as base_elbo, recon_loss as chamfer_recon


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
        # KL capacity schedule (Burgess et al. 2018)
        use_kl_capacity: bool = True,
        capacity_per_dim_end: float = 0.03,
        capacity_warmup_steps: int = 20000,
        stale_dropout_p: float = 0.2,
        set_mae_ratio: float = 0.0,
        # Task C: next-step distributional forecast (here: next set has any new event?)
        enable_next_change: bool = True,
        next_change_weight: float = 0.3,
        # Adaptive PoE gate (observation weight)
        use_adaptive_poe: bool = True,
        poe_beta_min: float = 0.1,
        poe_beta_max: float = 3.0,
        # Freeze SetVAE encoder during dynamics pretraining (two-stage training)
        freeze_set_encoder: bool = True,
        # PoE mode: 'conditional' uses a learned likelihood expert; 'naive' uses set encoder posterior
        poe_mode: str = "conditional",
        # Reconstruction weighting (perm-invariant Chamfer components)
        recon_alpha: float = 1.0,
        recon_beta: float = 1.0,
        recon_gamma: float = 3.0,
        recon_scale_calib: float = 0.0,
        recon_beta_var: float = 0.1,
        # Partial unfreezing options (when freeze_set_encoder=True)
        partial_unfreeze: bool = False,
        unfreeze_dim_reducer: bool = False,
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
        if freeze_set_encoder:
            for p in self.set_encoder.parameters():
                p.requires_grad = False
            if partial_unfreeze:
                # Unfreeze posterior head (mu_logvar)
                try:
                    for p in self.set_encoder.mu_logvar.parameters():
                        p.requires_grad = True
                except Exception:
                    pass
                # Unfreeze last encoder layer to adapt features slightly
                try:
                    for p in self.set_encoder.encoder_layers[-1].parameters():
                        p.requires_grad = True
                except Exception:
                    pass
                # Optionally unfreeze dimension reducer to calibrate scale
                if unfreeze_dim_reducer and getattr(self.set_encoder, "dim_reducer", None) is not None:
                    try:
                        for p in self.set_encoder.dim_reducer.parameters():
                            p.requires_grad = True
                    except Exception:
                        pass

        # PoE mode
        assert poe_mode in {"conditional", "naive"}
        self.poe_mode = poe_mode

        # GRU-based dynamics producing p(z_t | history)
        self.gru_cell = nn.GRUCell(latent_dim, latent_dim)

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

        # Adaptive observation gate for PoE (scalar per time step)
        self.use_adaptive_poe = use_adaptive_poe
        self.poe_beta_min = float(poe_beta_min)
        self.poe_beta_max = float(poe_beta_max)
        if self.use_adaptive_poe:
            # Inputs: [d_mahal, delta_H, log_dt1p] -> beta in (poe_beta_min, poe_beta_max)
            self.obs_gate = nn.Sequential(
                nn.Linear(3, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
            )

        # Conditional likelihood expert head (natural parameters)
        # Input at time t: [set_agg_t, mu_p_t, logvar_p_t] (concat over last dim)
        # Outputs: [log_prec_like_t, h_like_t]
        if self.poe_mode == "conditional":
            self.like_head = nn.Sequential(
                nn.Linear(latent_dim * 3, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim * 2),
            )

        # Training hparams
        self.lr = lr
        self.beta = beta
        self.free_bits = free_bits
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        # Capacity schedule
        self.use_kl_capacity = use_kl_capacity
        self.capacity_per_dim_end = capacity_per_dim_end
        self.capacity_warmup_steps = capacity_warmup_steps
        self.stale_dropout_p = stale_dropout_p
        self.set_mae_ratio = set_mae_ratio
        self.latent_dim = latent_dim
        self._step = 0
        self.enable_next_change = enable_next_change
        self.next_change_weight = next_change_weight
        # Recon weights
        self.recon_alpha = float(recon_alpha)
        self.recon_beta = float(recon_beta)
        self.recon_gamma = float(recon_gamma)
        self.recon_scale_calib = float(recon_scale_calib)
        self.recon_beta_var = float(recon_beta_var)
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

    def _poe(self, mu_qx, logvar_qx, mu_p, logvar_p, beta_scale: Optional[torch.Tensor] = None):
        var_qx = logvar_qx.exp()
        var_p = logvar_p.exp()
        if beta_scale is not None:
            # Scale observation precision by beta: Lambda_qx' = beta * Lambda_qx -> var_qx' = var_qx / beta
            var_qx = var_qx / beta_scale
        var_post = 1.0 / (1.0 / var_qx + 1.0 / var_p)
        mu_post = var_post * (mu_qx / var_qx + mu_p / var_p)
        logvar_post = torch.log(var_post + 1e-8)
        return mu_post, logvar_post

    @staticmethod
    def _kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p):
        # KL(q||p) for diagonal Gaussians
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        term_trace = (var_q / (var_p + 1e-8))
        term_quad = (mu_p - mu_q) ** 2 / (var_p + 1e-8)
        kld = 0.5 * (logvar_p - logvar_q + term_trace + term_quad - 1.0)
        # sum over dim -> [B,S]
        return kld.sum(dim=-1)

    def _beta(self):
        if not self.warmup_beta:
            return self.max_beta
        try:
            step = int(getattr(self, "global_step", 0))
        except Exception:
            step = 0
        if step < self.beta_warmup_steps:
            return self.max_beta * (step / self.beta_warmup_steps)
        return self.max_beta

    def _capacity(self):
        if not self.use_kl_capacity:
            return 0.0
        try:
            step = int(getattr(self, "global_step", 0))
        except Exception:
            step = 0
        cap_end = float(self.capacity_per_dim_end) * float(self.latent_dim)
        if step >= self.capacity_warmup_steps:
            return cap_end
        return cap_end * (float(step) / float(max(1, self.capacity_warmup_steps)))

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
        h_states: list[torch.Tensor] = []
        kl_list: list[torch.Tensor] = []
        beta_gate_list: list[torch.Tensor] = []
        next_change_loss = torch.tensor(0.0, device=device)
        next_change_targets = []

        # encode per set: q_x(z|x)
        for i, s in enumerate(sets):
            var, val, minute = s["var"], s["val"], s["minute"]
            # Optional Set-MAE masking on input values
            val_inp = self._apply_set_mae(val, self.set_mae_ratio)
            # Encode q_x(z|x)
            z_list_enc, enc_tokens = self.set_encoder.encode_from_var_val(var, val_inp)
            _z, mu_qx, logvar_qx = z_list_enc[-1]
            z_mu_list.append(mu_qx.squeeze(1))
            z_logvar_list.append(logvar_qx.squeeze(1))
            # use a single scalar time per set (e.g., last timestamp)
            pos_list.append(s["set_time"].float())
            # aggregate token features for conditional likelihood expert
            if self.poe_mode == "conditional":
                # enc_tokens shape [1,N,D]; mean pool to [1,D]
                agg = enc_tokens.mean(dim=1) if enc_tokens is not None else mu_qx.squeeze(1)
                z_mu_post_list.append(agg)  # temporarily stash set agg; will be replaced later

        z_mu = torch.stack(z_mu_list, dim=1)  # [1,S,D]
        z_logvar = torch.stack(z_logvar_list, dim=1)  # [1,S,D]
        minutes = torch.stack(pos_list, dim=1)
        if minutes.dim() == 1:
            minutes = minutes.unsqueeze(0)

        # Step through time with GRUCell; prior uses h_{t-1}, update with posterior mean
        B = 1
        h = torch.zeros(B, self.latent_dim, device=device)
        time_emb = self._relative_time_bucket_embedding(minutes)  # [1,S,D]
        mu_p_seq: list[torch.Tensor] = []
        logvar_p_seq: list[torch.Tensor] = []
        # Pre-extract aggregated set features if conditional mode
        if self.poe_mode == "conditional":
            set_aggs = torch.stack([z_mu_post_list[t] for t in range(S)], dim=1)  # [B,S,D]
            z_mu_post_list = []  # clear
        for t in range(S):
            # Prior from previous hidden
            prior_params_t = self.prior_head(h)  # [B, 2D]
            mu_p_t, logvar_p_t = prior_params_t.chunk(2, dim=-1)
            mu_p_seq.append(mu_p_t)
            logvar_p_seq.append(logvar_p_t)
            # Adaptive PoE beta (scalar per step)
            if self.use_adaptive_poe:
                mu_qx_t = z_mu[:, t, :]
                logvar_qx_t = z_logvar[:, t, :]
                var_sum = (logvar_p_t.exp() + logvar_qx_t.exp()).clamp(min=1e-8)
                d2 = ((mu_qx_t - mu_p_t) ** 2 / var_sum).mean(dim=-1, keepdim=True)  # [B,1]
                delta_H = (logvar_p_t - logvar_qx_t).mean(dim=-1, keepdim=True)  # [B,1]
                dt = minutes[:, t : t + 1] - (minutes[:, t - 1 : t] if t > 0 else minutes[:, t : t + 1])
                log_dt1p = torch.log1p(dt.clamp(min=0.0))  # [B,1]
                gate_inp = torch.cat([d2, delta_H, log_dt1p], dim=-1)
                gate_raw = self.obs_gate(gate_inp)
                gate_sig = torch.sigmoid(gate_raw)
                beta_t = self.poe_beta_min + (self.poe_beta_max - self.poe_beta_min) * gate_sig  # [B,1]
            else:
                beta_t = None
            # PoE merge
            if self.poe_mode == "conditional":
                # Build likelihood natural params from set agg and prior
                set_agg_t = set_aggs[:, t, :]
                like_inp = torch.cat([set_agg_t, mu_p_t, logvar_p_t], dim=-1)
                like_out = self.like_head(like_inp)
                log_prec_like_t, h_like_t = like_out.chunk(2, dim=-1)
                # precision must be positive; stabilize scale
                prec_like_t = F.softplus(log_prec_like_t) + 1e-4  # [B,D]
                Lambda_p_t = torch.exp(-logvar_p_t)  # [B,D]
                if beta_t is not None:
                    # scale likelihood contribution
                    beta_val = beta_t  # [B,1]
                    prec_like_t = prec_like_t * beta_val
                    h_like_t = h_like_t * beta_val
                Lambda_post_t = Lambda_p_t + prec_like_t
                h_post_t = Lambda_p_t * mu_p_t + h_like_t
                mu_post_t = h_post_t / (Lambda_post_t + 1e-8)
                logvar_post_t = -torch.log(Lambda_post_t + 1e-8)
            else:
                mu_post_t, logvar_post_t = self._poe(
                    z_mu[:, t, :], z_logvar[:, t, :], mu_p_t, logvar_p_t, beta_t
                )
            z_mu_post_list.append(mu_post_t.unsqueeze(1))
            z_logvar_post_list.append(logvar_post_t.unsqueeze(1))
            # KL(q_post || p_prior)
            kl_t = self._kl_diag_gauss(mu_post_t, logvar_post_t, mu_p_t, logvar_p_t)  # [B]
            # Free bits per step
            min_kl = self.free_bits * self.latent_dim
            kl_t = torch.clamp(kl_t, min=min_kl)
            kl_list.append(kl_t)
            if beta_t is not None:
                beta_gate_list.append(beta_t.squeeze(-1))
            # Update GRU state with posterior mean and time embedding
            inp_t = mu_post_t + time_emb[:, t, :]
            h = self.gru_cell(inp_t, h)
            h_states.append(h.unsqueeze(1))  # [B,1,D]

        mu_p = torch.cat(mu_p_seq, dim=0).view(B, S, -1) if len(mu_p_seq) > 0 else torch.zeros(B, S, self.latent_dim, device=device)
        logvar_p = torch.cat(logvar_p_seq, dim=0).view(B, S, -1) if len(logvar_p_seq) > 0 else torch.zeros(B, S, self.latent_dim, device=device)
        mu_post = torch.cat(z_mu_post_list, dim=1) if len(z_mu_post_list) > 0 else torch.zeros(B, S, self.latent_dim, device=device)
        logvar_post = torch.cat(z_logvar_post_list, dim=1) if len(z_logvar_post_list) > 0 else torch.zeros(B, S, self.latent_dim, device=device)
        h_seq = torch.cat(h_states, dim=1) if len(h_states) > 0 else torch.zeros(B, S, self.latent_dim, device=device)
        kl_total = torch.stack(kl_list, dim=1).mean() if len(kl_list) > 0 else torch.tensor(0.0, device=device)

        # Task C: next-step change prediction using h_t
        if self.enable_next_change and S > 1:
            # Build targets from set t+1 has_change for t in [0..S-2]
            for idx in range(1, S):
                next_change_targets.append(sets[idx]["has_change"])  # shape [1]
            targets = torch.stack(next_change_targets, dim=1).to(h_seq.dtype)  # [1, S-1]
            logits = self.next_change_head(h_seq[:, :-1, :]).squeeze(-1)  # [1, S-1]
            next_change_loss = self._bce(logits, targets)

        # reconstruct from h (use h_t)
        recon_total = 0.0
        for idx, s in enumerate(sets):
            N_t = s["var"].size(1)
            recon = self.decoder(h_seq[:, idx], N_t, noise_std=(0.0 if not self.training else 0.3))
            if self.set_encoder.dim_reducer is not None:
                reduced = self.set_encoder.dim_reducer(s["var"])
            else:
                reduced = s["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * s["val"]
            recon_total += chamfer_recon(
                recon,
                target_x,
                alpha=self.recon_alpha,
                beta=self.recon_beta,
                gamma=self.recon_gamma,
                beta_var=self.recon_beta_var,
                scale_calib_weight=self.recon_scale_calib,
            )
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
        # Capacity annealing: encourage KL to reach target capacity
        cap = torch.tensor(self._capacity(), device=kl.device, dtype=kl.dtype)
        kl_objective = torch.clamp(kl - cap, min=0.0)
        total = recon + beta * kl_objective + (self.next_change_weight * next_c if self.enable_next_change else 0.0)
        log_dict = {
            "train_loss": total,
            "train_recon": recon,
            "train_kl": kl,
            "train_kl_obj": kl_objective,
            "train_beta": beta,
            "train_capacity": cap,
        }
        if self.enable_next_change:
            log_dict["train_next_change"] = next_c
        self.log_dict(log_dict, prog_bar=True)
        self._step += 1
        return total

    def validation_step(self, batch, batch_idx):
        recon, kl, next_c = self.forward(batch)
        beta = self._beta()
        cap = torch.tensor(self._capacity(), device=kl.device, dtype=kl.dtype)
        kl_objective = torch.clamp(kl - cap, min=0.0)
        total = recon + beta * kl_objective + (self.next_change_weight * next_c if self.enable_next_change else 0.0)
        log_dict = {"val_loss": total, "val_recon": recon, "val_kl": kl, "val_kl_obj": kl_objective, "val_beta": beta, "val_capacity": cap}
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
        max_beta: float = 1.0,
        beta_warmup_steps: int = 20000,
        free_bits: float = 0.03,
        # KL capacity schedule (Burgess et al. 2018)
        use_kl_capacity: bool = True,
        capacity_per_dim_end: float = 0.10,
        capacity_warmup_steps: int = 30000,
        # New: fairness & stability controls
        kl_fairness_weight: float = 0.1,
        kl_spread_tol: float = 1.0,
        kl_over_weight: float = 1.0,
        var_stability_weight: float = 0.01,
        per_dim_free_bits: float = 0.002,
        # Reconstruction weighting (perm-invariant Chamfer components)
        recon_alpha: float = 1.0,
        recon_beta: float = 2.0,
        recon_gamma: float = 3.0,
        recon_scale_calib: float = 0.5,
        recon_beta_var: float = 0.01,
        # perturbation params
        p_stale: float = 0.1,
        p_live: float = 0.02,
        set_mae_ratio: float = 0.02,
        small_set_mask_prob: float = 0.1,
        small_set_threshold: int = 5,
        max_masks_per_set: int = 2,
        val_noise_std: float = 0.02,
        dir_noise_std: float = 0.01,
        train_decoder_noise_std: float = 0.05,
        eval_decoder_noise_std: float = 0.02,
        # InfoVAE / Flow options
        use_flows: bool = False,
        num_flows: int = 0,
        mmd_weight: float = 0.0,
        mmd_scales: tuple = (1.0, 2.0, 4.0, 8.0),
        # Posterior stability in encoder
        posterior_logvar_min: float = -2.5,
        posterior_logvar_max: float = 2.5,
        enable_posterior_std_augmentation: bool = False,
        posterior_std_aug_sigma: float = 0.0,
        # Auto-tune and monitoring
        auto_tune_kl: bool = True,
        kl_target_nats: float = 10.0,
        kl_patience_epochs: int = 2,
        beta_step: float = 0.2,
        capacity_per_dim_step: float = 0.02,
        max_beta_ceiling: float = 2.0,
        capacity_per_dim_max: float = 0.20,
        active_ratio_warn_threshold: float = 0.5,
    ):
        super().__init__()
        self.set_encoder = SetVAEModule(
            input_dim=input_dim,
            reduced_dim=reduced_dim,
            latent_dim=latent_dim,
            levels=levels,
            heads=heads,
            m=m,
            use_flows=use_flows,
            num_flows=num_flows,
            posterior_logvar_min=posterior_logvar_min,
            posterior_logvar_max=posterior_logvar_max,
            enable_posterior_std_augmentation=enable_posterior_std_augmentation,
            posterior_std_aug_sigma=posterior_std_aug_sigma,
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
        # Fairness & stability
        self.kl_fairness_weight = float(kl_fairness_weight)
        self.kl_spread_tol = float(kl_spread_tol)
        self.kl_over_weight = float(kl_over_weight)
        self.var_stability_weight = float(var_stability_weight)
        self.per_dim_free_bits = float(per_dim_free_bits)
        # Recon weights
        self.recon_alpha = float(recon_alpha)
        self.recon_beta = float(recon_beta)
        self.recon_gamma = float(recon_gamma)
        self.recon_scale_calib = float(recon_scale_calib)
        self.recon_beta_var = float(recon_beta_var)
        # Capacity schedule
        self.use_kl_capacity = use_kl_capacity
        self.capacity_per_dim_end = capacity_per_dim_end
        self.capacity_warmup_steps = capacity_warmup_steps

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
        # InfoVAE/MMD
        self.mmd_weight = float(mmd_weight)
        try:
            # ensure tuple of floats
            self.mmd_scales = tuple(float(s) for s in (mmd_scales if isinstance(mmd_scales, (list, tuple)) else (mmd_scales,)))
        except Exception:
            self.mmd_scales = (1.0, 2.0, 4.0, 8.0)

        # Auto-tune & monitors
        self.auto_tune_kl = bool(auto_tune_kl)
        self.kl_target_nats = float(kl_target_nats)
        self.kl_patience_epochs = int(kl_patience_epochs)
        self.beta_step = float(beta_step)
        self.capacity_per_dim_step = float(capacity_per_dim_step)
        self.max_beta_ceiling = float(max_beta_ceiling)
        self.capacity_per_dim_max = float(capacity_per_dim_max)
        self.active_ratio_warn_threshold = float(active_ratio_warn_threshold)
        self._kl_not_met_epochs = 0
        self._num_adjustments = 0
        self._val_kl_sum = 0.0
        self._val_kl_count = 0
        self._active_warnings = []

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

    def _capacity(self):
        if not self.use_kl_capacity:
            return 0.0
        cap_end = float(self.capacity_per_dim_end) * float(self.latent_dim)
        if self._step >= self.capacity_warmup_steps:
            return cap_end
        return cap_end * (float(self._step) / float(max(1, self.capacity_warmup_steps)))

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

    def _apply_set_mae_inplace(self, val: torch.Tensor, carry: torch.Tensor, original_val: torch.Tensor):
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
        # guarantee at least one non-carry token not zeroed if possible (restore original value)
        non_carry = (carry <= 0.5).squeeze(-1)
        if non_carry.any():
            # if all non-carry got zeroed, restore one
            mask_zero = (val.abs() <= 1e-8).squeeze(-1)
            if torch.all(mask_zero | (~non_carry)):
                indices = torch.nonzero(non_carry, as_tuple=False).squeeze(-1)
                pick = indices[torch.randint(0, len(indices), (1,), device=device)]
                val[:, pick, :] = original_val[:, pick, :]

    def _forward_single(self, s):
        # target (clean)
        v_norm, x_target = self._compute_target(s)  # [1,N,D], [1,N,D]

        # build noisy input
        val_in = s["val"].clone()
        original_val = val_in.clone()
        carry = s.get("carry", torch.zeros_like(val_in))
        # value dropout by stale/live
        val_in = self._apply_value_dropout(val_in, carry)
        # token masking (Set-MAE)
        self._apply_set_mae_inplace(val_in, carry, original_val)
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
        r_loss = chamfer_recon(
            recon,
            x_target,
            alpha=self.recon_alpha,
            beta=self.recon_beta,
            gamma=self.recon_gamma,
            beta_var=self.recon_beta_var,
            scale_calib_weight=self.recon_scale_calib,
        )
        # Last-layer raw KL to N(0,I) (no extra regularizers)
        _z_last, mu_last, logvar_last = z_list[-1]
        per_dim_kl_last = 0.5 * (logvar_last.exp() + mu_last.pow(2) - 1.0 - logvar_last)  # [1,1,D]
        # Per-dim free bits clamp to avoid punishing low-use dimensions and preserve capacity per dim
        if self.per_dim_free_bits is not None and self.per_dim_free_bits > 0.0:
            per_dim_kl_last = torch.clamp(per_dim_kl_last, min=self.per_dim_free_bits)
        kl_last = per_dim_kl_last.sum(dim=-1).mean()  # scalar
        # Variance stability: penalize posterior variance deviating from 1
        var_dev_last = (logvar_last.exp() - 1.0).pow(2).mean()
        # collect last-layer latent sample (after optional flows) for MMD
        z_last = z_list[-1][0]  # [1,1,D]
        try:
            if hasattr(self.set_encoder, 'apply_flows') and len(getattr(self.set_encoder, 'flows', [])) > 0:
                z_last, _ = self.set_encoder.apply_flows(z_last)
        except Exception:
            pass
        z_flat = z_last.squeeze(1)  # [1,D]
        # Return per-dim KL for fairness aggregation and var deviation
        return r_loss, kl_last, z_flat, per_dim_kl_last.squeeze(0).squeeze(0), var_dev_last

    def forward(self, batch):
        var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        all_sets = self._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
        total_recon, total_kl, count = 0.0, 0.0, 0
        zs = []
        kl_dim_sum = None
        var_dev_sum = 0.0
        for sets in all_sets:
            for s in sets:
                recon, kl, z_flat, kl_dim_vec, var_dev = self._forward_single(s)
                total_recon += recon
                total_kl += kl
                zs.append(z_flat)
                if kl_dim_sum is None:
                    kl_dim_sum = kl_dim_vec
                else:
                    kl_dim_sum = kl_dim_sum + kl_dim_vec
                var_dev_sum = var_dev_sum + var_dev
                count += 1
        if count == 0:
            device = var.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, torch.zeros(0, self.latent_dim, device=device), torch.zeros(self.latent_dim, device=device), torch.tensor(0.0, device=device)
        z_samples = torch.cat(zs, dim=0) if len(zs) > 0 else torch.zeros(0, self.latent_dim, device=var.device)
        kl_dim_mean = kl_dim_sum / float(max(1, count)) if kl_dim_sum is not None else torch.zeros(self.latent_dim, device=var.device)
        var_dev_mean = var_dev_sum / float(max(1, count))
        return total_recon / count, total_kl / count, z_samples, kl_dim_mean, var_dev_mean

    # --- InfoVAE / MMD ---
    @staticmethod
    def _pdists(x: torch.Tensor, y: torch.Tensor):
        # x: [N,D], y: [M,D] -> [N,M]
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1, keepdim=True).t()
        xy = x @ y.t()
        d2 = (x2 + y2 - 2.0 * xy).clamp_min(0.0)
        return d2

    def _rbf_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or y.numel() == 0:
            return torch.tensor(0.0, device=x.device if x.numel() > 0 else y.device)
        dxx = self._pdists(x, x)
        dyy = self._pdists(y, y)
        dxy = self._pdists(x, y)
        kxx = 0.0
        kyy = 0.0
        kxy = 0.0
        for s in self.mmd_scales:
            gamma = 1.0 / (2.0 * (float(s) ** 2))
            kxx = kxx + torch.exp(-gamma * dxx)
            kyy = kyy + torch.exp(-gamma * dyy)
            kxy = kxy + torch.exp(-gamma * dxy)
        n = x.shape[0]
        m = y.shape[0]
        # Use biased estimate for stability
        mmd2 = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1) + 1e-8) \
             + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1) + 1e-8) \
             - 2.0 * kxy.mean()
        return mmd2

    # training hooks
    def training_step(self, batch, batch_idx):
        recon, kl, z_samples, kl_dim, var_dev = self.forward(batch)
        beta = self._beta()
        cap = torch.tensor(self._capacity(), device=kl.device, dtype=kl.dtype)
        # Lower-bound capacity objective: penalize only when below capacity
        kl_objective = F.relu(cap - kl)
        # MMD regularizer (InfoVAE-style): match aggregated q(z) to p(z)
        mmd = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
        if self.mmd_weight > 0.0 and z_samples.numel() > 0:
            prior = torch.randn_like(z_samples)
            mmd = self._rbf_mmd(z_samples, prior)
        # KL fairness: encourage high-entropy per-dimension KL allocation, limit over-concentration
        fairness = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
        if kl_dim is not None and kl_dim.numel() > 0:
            eps = 1e-8
            D = float(self.latent_dim)
            p = kl_dim / (kl_dim.sum() + eps)
            uniform_log = math.log(1.0 / max(1.0, D))
            kl_to_uniform = (p * (torch.log(p + eps) - uniform_log)).sum()
            over = F.relu(p - (1.0 / D) * (1.0 + self.kl_spread_tol))
            l2_over = (over ** 2).sum()
            fairness = self.kl_fairness_weight * (kl_to_uniform + self.kl_over_weight * l2_over)
        # Variance stability penalty
        var_pen = self.var_stability_weight * var_dev
        total = recon + beta * kl_objective + (self.mmd_weight * mmd) + fairness + var_pen
        self.log_dict({"train_loss": total, "train_recon": recon, "train_kl_last": kl, "train_kl_obj_lb": kl_objective, "train_beta": beta, "train_capacity": cap, "train_mmd": mmd, "train_kl_fair": fairness, "train_var_pen": var_pen}, prog_bar=True)
        self._step += 1
        return total

    def validation_step(self, batch, batch_idx):
        recon, kl, z_samples, kl_dim, var_dev = self.forward(batch)
        beta = self._beta()
        cap = torch.tensor(self._capacity(), device=kl.device, dtype=kl.dtype)
        kl_objective = F.relu(cap - kl)
        mmd = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
        if self.mmd_weight > 0.0 and z_samples.numel() > 0:
            prior = torch.randn_like(z_samples)
            mmd = self._rbf_mmd(z_samples, prior)
        # KL fairness and variance penalties (eval-time)
        fairness = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
        if kl_dim is not None and kl_dim.numel() > 0:
            eps = 1e-8
            D = float(self.latent_dim)
            p = kl_dim / (kl_dim.sum() + eps)
            uniform_log = math.log(1.0 / max(1.0, D))
            kl_to_uniform = (p * (torch.log(p + eps) - uniform_log)).sum()
            over = F.relu(p - (1.0 / D) * (1.0 + self.kl_spread_tol))
            l2_over = (over ** 2).sum()
            fairness = self.kl_fairness_weight * (kl_to_uniform + self.kl_over_weight * l2_over)
        var_pen = self.var_stability_weight * var_dev
        total = recon + beta * kl_objective + (self.mmd_weight * mmd) + fairness + var_pen
        self.log_dict({"val_loss": total, "val_recon": recon, "val_kl_last": kl, "val_kl_obj_lb": kl_objective, "val_beta": beta, "val_capacity": cap, "val_mmd": mmd, "val_kl_fair": fairness, "val_var_pen": var_pen}, prog_bar=True, on_epoch=True)
        # accumulate kl for epoch-level auto-tune
        self._val_kl_sum += float(kl.detach().cpu().item())
        self._val_kl_count += 1
        # Per-dim KL stats on clean inputs (no perturbations)
        try:
            kl_dim = self._compute_kl_dim_stats(batch)  # [D]
            # Active ratios
            for thr in [0.005, 0.01, 0.02]:
                active = float((kl_dim > thr).float().mean().item())
                self.log(f"val_active_ratio@{thr}", active, prog_bar=(thr == 0.01), on_epoch=True)
                if thr == 0.01 and active < self.active_ratio_warn_threshold:
                    msg = f"Active ratio@0.01 low: {active:.3f} (<{self.active_ratio_warn_threshold})"
                    self._active_warnings.append({"epoch": int(getattr(self, "current_epoch", -1)), "message": msg})
                    try:
                        print(f"⚠️  {msg}")
                    except Exception:
                        pass
            # coverage metrics
            kd = kl_dim.detach()
            total_kl = float(kd.sum().item() + 1e-8)
            if total_kl > 0:
                sorted_vals, _ = torch.sort(kd, descending=True)
                cumsum = torch.cumsum(sorted_vals, dim=0) / sorted_vals.sum()
                d90 = int((cumsum < 0.90).sum().item() + 1)
                d95 = int((cumsum < 0.95).sum().item() + 1)
                self.log("val_kl_dim90", d90, prog_bar=False, on_epoch=True)
                self.log("val_kl_dim95", d95, prog_bar=False, on_epoch=True)
        except Exception:
            pass
        return total

    def on_validation_epoch_start(self):
        self._val_kl_sum = 0.0
        self._val_kl_count = 0

    def on_validation_epoch_end(self):
        if self._val_kl_count <= 0:
            return
        kl_epoch = self._val_kl_sum / max(1, self._val_kl_count)
        self.log("val_kl_last_epoch", kl_epoch, prog_bar=True)
        # Auto-tune if KL below target
        if self.auto_tune_kl and kl_epoch < self.kl_target_nats:
            self._kl_not_met_epochs += 1
            if self._kl_not_met_epochs >= self.kl_patience_epochs:
                did_adjust = False
                # Increase capacity end within limit
                if self.capacity_per_dim_end < self.capacity_per_dim_max:
                    new_cap = min(self.capacity_per_dim_end + self.capacity_per_dim_step, self.capacity_per_dim_max)
                    try:
                        print(f"🔧 Increasing capacity_per_dim_end: {self.capacity_per_dim_end:.4f} -> {new_cap:.4f}")
                    except Exception:
                        pass
                    self.capacity_per_dim_end = new_cap
                    did_adjust = True
                # Increase max_beta within ceiling
                if self.max_beta < self.max_beta_ceiling:
                    new_beta = min(self.max_beta + self.beta_step, self.max_beta_ceiling)
                    try:
                        print(f"🔧 Increasing max_beta: {self.max_beta:.3f} -> {new_beta:.3f}")
                    except Exception:
                        pass
                    self.max_beta = new_beta
                    did_adjust = True
                if did_adjust:
                    self._num_adjustments += 1
                self._kl_not_met_epochs = 0
                # If too many adjustments, trigger early stop
                if self._num_adjustments >= 5:
                    try:
                        print("⛔ KL target not met after multiple adjustments; requesting early stop.")
                    except Exception:
                        pass
                    try:
                        if getattr(self, 'trainer', None) is not None:
                            self.trainer.should_stop = True
                    except Exception:
                        pass

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

    def _compute_kl_dim_stats(self, batch: dict) -> torch.Tensor:
        var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        all_sets = self._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
        kl_dim_sum = None
        count = 0
        for sets in all_sets:
            for s in sets:
                # clean input
                if self.set_encoder.dim_reducer is not None:
                    reduced = self.set_encoder.dim_reducer(s["var"])  # [1,N,R]
                else:
                    reduced = s["var"]
                norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
                v_norm = reduced / (norms + 1e-8)
                x_clean = v_norm * s["val"]
                z_list, _ = self.set_encoder.encode(x_clean)
                _z, mu, logvar = z_list[-1]
                # [1,1,D] -> [D]
                kl_dim = 0.5 * (logvar.exp().squeeze(0).squeeze(0) + mu.squeeze(0).squeeze(0).pow(2) - 1.0 - logvar.squeeze(0).squeeze(0))
                if kl_dim_sum is None:
                    kl_dim_sum = kl_dim
                else:
                    kl_dim_sum = kl_dim_sum + kl_dim
                count += 1
        if kl_dim_sum is None or count == 0:
            return torch.zeros(self.latent_dim, device=var.device)
        return kl_dim_sum / float(count)
