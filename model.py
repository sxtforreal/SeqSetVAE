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
        skip_pretrained_on_resume: bool = False,  # New parameter: whether to skip pretrained loading when resuming
        use_focal_loss: bool = False,
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
        
        # Pretrained loading disabled: always start from random initialization

        # Freeze X% pretrained parameters
        set_params = list(self.setvae.parameters())
        freeze_cnt = int(len(set_params) * freeze_ratio)
        for p in set_params[:freeze_cnt]:
            p.requires_grad = False

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
        
        # Enhanced classification head for better AUC/AUPRC performance
        self.cls_head = nn.Sequential(
            # First layer with normalization and residual connection
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second layer with normalization
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer with normalization
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LayerNorm(latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Final classification layer
            nn.Linear(latent_dim // 4, num_classes)
        )
        
        # Alternative: Multi-scale feature fusion for better representation
        self.feature_fusion = nn.ModuleDict({
            'global_pool': nn.AdaptiveAvgPool1d(1),
            'max_pool': nn.AdaptiveMaxPool1d(1),
            'attention_pool': nn.MultiheadAttention(
                embed_dim=latent_dim, 
                num_heads=4, 
                batch_first=True,
                dropout=0.1
            )
        })
        
        # Feature projection for fusion
        self.feature_projection = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),  # 3 pooling methods
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        self.use_focal_loss = use_focal_loss
        self.focal_loss_fn = (
            FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
            if use_focal_loss
            else None
        )
        
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
        
        # Apply transformer with enhanced processing
        h_seq = self.transformer(z_seq, mask=attn_mask[0] if attn_mask.dim() == 3 else attn_mask)
        
        # Apply post-transformer normalization for better feature quality
        h_seq = self.post_transformer_norm(h_seq)
        
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
        
        # Enhanced feature extraction using multi-scale pooling
        enhanced_features = self._extract_enhanced_features(h_seq)
        
        # Alternative: Use attention-based pooling as fallback
        if enhanced_features is None:
            attn_weights = F.softmax(torch.sum(h_seq * z_seq, dim=-1), dim=1)  # [B, S]
            enhanced_features = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        # Classification with enhanced features
        logits = self.cls_head(enhanced_features)
        
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

    def _extract_enhanced_features(self, h_t):
        """
        Extract enhanced features using multi-scale pooling and fusion
        for better AUC/AUPRC performance
        """
        B, S, D = h_t.shape
        
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
        if self.use_focal_loss and self.focal_loss_fn is not None:
            pred_loss = self.focal_loss_fn(logits, label)
        else:
            pred_loss = F.cross_entropy(logits, label, label_smoothing=0.1)
        
        # Optimized weight strategy for better AUC/AUPRC
        if stage == "train":
            # More stable weight strategy: focus on classification throughout training
            # Start with balanced weights, gradually increase classification focus
            progress = min(1.0, self.current_step / 5000)
            recon_weight = 0.8 * (1.0 - progress) + 0.3 * progress  # Decrease from 0.8 to 0.3
            pred_weight = self.w * (0.5 + 0.5 * progress)  # Increase from 0.5*w to w
        else:
            # Validation: use balanced weights for evaluation
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
        # Set different learning rates for different parts
        setvae_params = list(self.setvae.parameters())
        transformer_params = list(self.transformer.parameters())
        other_params = [
            p for n, p in self.named_parameters() 
            if not any(n.startswith(prefix) for prefix in ['setvae', 'transformer'])
        ]
        
        # Use smaller learning rate for pretrained SetVAE
        param_groups = [
            {'params': setvae_params, 'lr': self.lr * 0.05, 'name': 'setvae'},  # Reduced from 0.1
            {'params': transformer_params, 'lr': self.lr, 'name': 'transformer'},
            {'params': other_params, 'lr': self.lr, 'name': 'others'}
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.02,  # Increased weight decay for better regularization
        )
        
        # Improved learning rate scheduler for better AUC/AUPRC
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',  # Monitor training loss (lower is better)
            factor=0.7,  # Reduce LR by 30% when plateau
            patience=200,  # Wait 200 steps before reducing LR
            verbose=True,
            min_lr=self.lr * 0.001  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Check every epoch instead of every step
                "frequency": 1,
                "monitor": "val_loss",  # Monitor validation loss for scheduling
            },
        }
