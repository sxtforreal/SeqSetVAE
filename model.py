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
        freeze_ratio: float = 0.0,  # Default: no freezing
        pretrained_ckpt: str = None,
        w: float = 1.0,
        free_bits: float = 0.1,
        warmup_beta: bool = True,
        max_beta: float = 0.1,
        beta_warmup_steps: int = 5000,
        kl_annealing: bool = True,
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
            ckpt = torch.load(pretrained_ckpt, map_location='cpu')
            state_dict = ckpt.get("state_dict", ckpt)
            setvae_state = {
                k.replace("setvae.", ""): v
                for k, v in state_dict.items()
                if k.startswith("setvae.")
            }
            self.setvae.load_state_dict(setvae_state, strict=False)
            del ckpt, state_dict, setvae_state

        # Freeze X% pretrained parameters
        set_params = list(self.setvae.parameters())
        freeze_cnt = int(len(set_params) * freeze_ratio)
        for p in set_params[:freeze_cnt]:
            p.requires_grad = False

        # Transformer encoder with improved configuration
        enc_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',  # Use GELU activation function
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # Dynamic positional encoding - no fixed max sequence length
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Time encoder - convert continuous time to embeddings
        self.time_encoder = nn.Sequential(
            nn.Linear(1, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim),
            nn.Tanh()
        )

        # Decoder & Classifier with improved architecture
        self.decoder = _SetDecoder(
            latent_dim,
            reduced_dim,
            levels,
            heads,
            m,
        )
        
        # Improved classification head
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, num_classes)
        )

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

    def _apply_positional_encoding(self, x: torch.Tensor, pos: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Apply learned positional encoding and time encoding for variable length sequences.
        
        Args:
            x: Tensor of shape [B, S, D]
            pos: Tensor of shape [B, S] representing each set's minute value
            padding_mask: Boolean mask of shape [B, S] where True indicates padding
        Returns:
            Tensor of same shape as x with positional encoding applied.
        """
        B, S, D = x.shape
        
        # Positional encoding (based on sequence position)
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding.expand(B, S, -1) * torch.sin(positions.unsqueeze(-1).float() * 0.01)
        
        # Time encoding (based on actual time)
        time_emb = self.time_encoder(pos.unsqueeze(-1))
        
        # Combine positional and time encoding
        encoded = x + pos_emb + time_emb
        
        # Zero out padding positions
        if padding_mask is not None:
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        return encoded

    def _create_causal_mask(self, seq_len, device, padding_mask=None):
        """Create causal mask for variable length sequences with padding support"""
        # Create basic causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        
        # Add some randomness during training (simulate real clinical scenarios)
        if self.training:
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

    def forward(self, sets, padding_mask=None):
        """
        Forward pass with support for variable length sequences and batch processing.
        
        Args:
            sets: List of patient sets, where each patient has a list of set dictionaries
                  Each set dict contains: {"var": [1, N, D], "val": [1, N, 1], "minute": [1, N, 1]}
            padding_mask: [B, S] boolean mask where True indicates padding (optional)
        
        Returns:
            logits: [B, num_classes] Classification logits
            recon_loss: Scalar reconstruction loss
            kl_loss: Scalar KL divergence loss
        """
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
        S = len(sets)
        z_prims, kl_total = [], 0.0
        pos_list = []
        
        # Get current beta value
        current_beta = self.get_current_beta()
        
        # Store latent variable information (for collapse detector)
        all_z_lists = []
        
        for s_dict in sets:
            var, val, time = (
                s_dict["var"],
                s_dict["val"],
                s_dict["minute"],
            )
            assert time.unique().numel() == 1, "Time is not constant in this set"
            minute_val = time.unique().float()  # Ensure float type
            pos_list.append(minute_val)
            
            recon, z_list, _ = self.setvae(var, val)
            z_sample, mu, logvar = z_list[-1]  # Choose the deepest layer
            z_prims.append(z_sample.squeeze(1))  # -> [B, latent]
            
            # Collect latent variable information (for collapse detector)
            all_z_lists.append(z_list)
            
            # Improved KL loss calculation
            # Use more stable KL divergence calculation
            kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
            
            # Apply free bits
            min_kl = self.free_bits * self.latent_dim
            kl_div = torch.clamp(kl_div, min=min_kl)
            
            # Add KL regularization term to prevent variance from being too small
            var_reg = -0.1 * torch.mean(logvar)
            kl_total += kl_div.mean() + var_reg
            
        kl_total = kl_total / S
        z_seq = torch.stack(z_prims, dim=1)  # [B, S, latent]
        pos_tensor = torch.stack(pos_list, dim=1)  # [B, S]
        
        # Apply positional and time encoding
        z_seq = self._apply_positional_encoding(z_seq, pos_tensor, None)
        
        # Add layer normalization
        z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])
        
        # Create attention mask for transformer
        attn_mask = self._create_causal_mask(S, z_seq.device, None)
        
        # Apply transformer
        h_seq = self.transformer(z_seq, mask=attn_mask[0] if attn_mask.dim() == 3 else attn_mask)

        # Reconstruction with improved loss
        recon_loss_total = 0.0
        valid_sets = 0
        for idx, s_dict in enumerate(sets):
            N_t = s_dict["var"].size(1)
            recon = self.decoder(h_seq[:, idx], N_t, noise_std=0.3)  # Reduce noise
            if self.setvae.setvae.dim_reducer is not None:
                reduced = self.setvae.setvae.dim_reducer(s_dict["var"])
            else:
                reduced = s_dict["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * s_dict["val"]
            recon_loss_total += chamfer_recon_loss(recon, target_x)
            valid_sets += 1
            
        if valid_sets > 0:
            recon_loss_total /= valid_sets

        # Classification with attention pooling
        attn_weights = F.softmax(torch.sum(h_seq * z_seq, dim=-1), dim=1)  # [B, S]
        final_rep = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        logits = self.cls_head(final_rep)
        
        # Save latent variable information for collapse detector
        if all_z_lists:
            # Merge latent variable information from all sets (use first set as representative)
            self._last_z_list = all_z_lists[0] if all_z_lists else None
        
        return logits, recon_loss_total, kl_total * current_beta

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
                    "var": torch.empty(0, patient_var.size(-1), device=patient_var.device).unsqueeze(0),
                    "val": torch.empty(0, 1, device=patient_val.device).unsqueeze(0),
                    "minute": torch.empty(0, 1, device=patient_time.device).unsqueeze(0),
                })
            else:
                # Find unique consecutive set IDs
                unique_set_ids, counts = torch.unique_consecutive(patient_set_ids.squeeze(-1), return_counts=True)
                indices = torch.split(torch.arange(len(patient_set_ids), device=patient_set_ids.device), counts.tolist())
                
                for idx in indices:
                    if len(idx) > 0:  # Only add non-empty sets
                        patient_sets.append({
                            "var": patient_var[idx].unsqueeze(0),  # [1, set_size, D]
                            "val": patient_val[idx].unsqueeze(0),  # [1, set_size, 1]
                            "minute": patient_time[idx].unsqueeze(0),  # [1, set_size, 1]
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
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        valid_patients = 0
        
        for patient_sets in all_patient_sets:
            if len(patient_sets) == 0:
                # Skip empty patients
                continue
                
            # Process this patient's sets
            logits, recon_loss, kl_loss = self(patient_sets)
            all_logits.append(logits)
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            valid_patients += 1
        
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
        
        # Improved loss calculation
        pred_loss = F.cross_entropy(logits, label, label_smoothing=0.1)  # Add label smoothing
        
        # Dynamic weight adjustment
        if stage == "train":
            # Focus more on reconstruction early in training, more on classification later
            recon_weight = max(0.5, 1.0 - self.current_step / 10000)
            pred_weight = min(self.w, self.current_step / 5000)
        else:
            recon_weight = 0.5
            pred_weight = self.w
            
        total_loss = pred_weight * pred_loss + recon_weight * recon_loss + kl_loss

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
        self.log_dict(
            {
                f"{stage}_loss": total_loss,
                f"{stage}_recon": recon_loss,
                f"{stage}_kl": kl_loss,
                f"{stage}_pred": pred_loss,
                f"{stage}_beta": current_beta,
                f"{stage}_recon_weight": recon_weight,
                f"{stage}_pred_weight": pred_weight,
            },
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        
        # Store logged metrics for collapse detector
        if stage == "train":
            self.logged_metrics = {
                'train_kl': kl_loss,
                'train_recon': recon_loss,
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

    def configure_optimizers(self):
        # Set different learning rates for different parts
        setvae_params = list(self.setvae.parameters())
        transformer_params = list(self.transformer.parameters())
        other_params = [
            p for n, p in self.named_parameters() 
            if not any(n.startswith(prefix) for prefix in ['setvae', 'transformer'])
        ]
        
        # Use smaller learning rate for pretrained SetVAE
        param_groups = [
            {'params': setvae_params, 'lr': self.lr * 0.1, 'name': 'setvae'},
            {'params': transformer_params, 'lr': self.lr, 'name': 'transformer'},
            {'params': other_params, 'lr': self.lr, 'name': 'others'}
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,  # Add weight decay
        )
        
        # Improved learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=1000,  # Steps for first restart
            T_mult=2,  # Multiplication factor for restart intervals
            eta_min=self.lr * 0.01  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_auc",
            },
        }
