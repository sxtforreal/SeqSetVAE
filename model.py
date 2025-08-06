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
        freeze_ratio: float = 0.0,  # 默认不冻结
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
            activation='gelu',  # 使用GELU激活函数
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # 学习的位置编码替代RoPE
        self.max_seq_len = 512  # 假设最大序列长度
        self.pos_embedding = nn.Embedding(self.max_seq_len, latent_dim)
        
        # 时间编码器 - 将连续时间转换为嵌入
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
        
        # 改进的分类头
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
        """计算当前的beta值，支持warmup和annealing"""
        if not self.warmup_beta:
            return self.max_beta
            
        if self.current_step < self.beta_warmup_steps:
            # Linear warmup
            return self.max_beta * (self.current_step / self.beta_warmup_steps)
        else:
            return self.max_beta

    def _apply_positional_encoding(self, x: torch.Tensor, pos: torch.Tensor):
        """
        Apply learned positional encoding and time encoding.
        
        Args:
            x: Tensor of shape [B, S, D]
            pos: Tensor of shape [B, S] representing each set's minute value.
        Returns:
            Tensor of same shape as x with positional encoding applied.
        """
        B, S, D = x.shape
        
        # 位置编码（基于序列位置）
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        
        # 时间编码（基于实际时间）
        time_emb = self.time_encoder(pos.unsqueeze(-1))
        
        # 组合位置编码和时间编码
        return x + pos_emb + time_emb

    def _causal_mask(self, unique_sets, device):
        # 使用更宽松的causal mask，允许一定程度的双向信息流
        mask = torch.triu(
            torch.ones(unique_sets, unique_sets, device=device), diagonal=1
        )
        # 添加一些随机性，允许部分未来信息泄露（模拟真实临床场景）
        if self.training:
            random_mask = torch.rand(unique_sets, unique_sets, device=device) < 0.1
            mask = mask & ~random_mask
        return mask.bool()

    def forward(self, sets):
        """sets : list of dict{[B, N, D],[B, N, 1],[B, N, 1]}"""

        S = len(sets)
        z_prims, kl_total = [], 0.0
        pos_list = []
        
        # 获取当前beta值
        current_beta = self.get_current_beta()
        
        for s_dict in sets:
            var, val, time = (
                s_dict["var"],
                s_dict["val"],
                s_dict["minute"],
            )
            assert time.unique().numel() == 1, "Time is not constant in this set"
            minute_val = time.unique().float()  # 确保是float类型
            pos_list.append(minute_val)
            
            recon, z_list, _ = self.setvae(var, val)
            z_sample, mu, logvar = z_list[-1]  # Choose the deepest layer
            z_prims.append(z_sample.squeeze(1))  # -> [B, latent]
            
            # 改进的KL损失计算
            # 使用更稳定的KL散度计算
            kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
            
            # 应用free bits
            min_kl = self.free_bits * self.latent_dim
            kl_div = torch.clamp(kl_div, min=min_kl)
            
            # 添加KL正则化项，防止方差过小
            var_reg = -0.1 * torch.mean(logvar)
            kl_total += kl_div.mean() + var_reg
            
        kl_total = kl_total / S
        z_seq = torch.stack(z_prims, dim=1)  # [B, S, latent]
        pos_tensor = torch.stack(pos_list, dim=1)  # [B, S]
        
        # Apply positional and time encoding
        z_seq = self._apply_positional_encoding(z_seq, pos_tensor)
        
        # 添加层归一化
        z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])
        
        h_seq = self.transformer(z_seq, mask=self._causal_mask(S, z_seq.device))

        # Reconstruction with improved loss
        recon_loss_total = 0.0
        for idx, s_dict in enumerate(sets):
            N_t = s_dict["var"].size(1)
            recon = self.decoder(h_seq[:, idx], N_t, noise_std=0.3)  # 降低噪声
            if self.setvae.setvae.dim_reducer is not None:
                reduced = self.setvae.setvae.dim_reducer(s_dict["var"])
            else:
                reduced = s_dict["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * s_dict["val"]
            recon_loss_total += chamfer_recon_loss(recon, target_x)
        recon_loss_total /= S

        # Classification with attention pooling
        # 使用注意力机制进行序列级聚合
        attn_weights = F.softmax(torch.sum(h_seq * z_seq, dim=-1), dim=1)  # [B, S]
        final_rep = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        logits = self.cls_head(final_rep)
        
        return logits, recon_loss_total, kl_total * current_beta

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
        logits, recon_loss, kl_loss = self(sets)
        
        # 改进的损失计算
        pred_loss = F.cross_entropy(logits, label, label_smoothing=0.1)  # 添加标签平滑
        
        # 动态权重调整
        if stage == "train":
            # 训练早期更关注重构，后期更关注分类
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

        # 记录更多有用的指标
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
        
        if stage == "train":
            self.current_step += 1
            
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
        # 分别为不同部分设置不同的学习率
        setvae_params = list(self.setvae.parameters())
        transformer_params = list(self.transformer.parameters())
        other_params = [
            p for n, p in self.named_parameters() 
            if not any(n.startswith(prefix) for prefix in ['setvae', 'transformer'])
        ]
        
        # 为预训练的SetVAE使用较小的学习率
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
            weight_decay=0.01,  # 添加权重衰减
        )
        
        # 改进的学习率调度器
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=1000,  # 第一次重启的步数
            T_mult=2,  # 重启间隔的倍增因子
            eta_min=self.lr * 0.01  # 最小学习率
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
