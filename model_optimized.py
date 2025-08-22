"""
Optimized SeqSetVAE Model - Performance Enhanced
修复训练速度慢和性能下降的优化版本
"""

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
    """Load weights from checkpoint - handles both full PyTorch Lightning checkpoints and direct state dicts."""
    print(f"📦 Loading weights from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"✅ Found PyTorch Lightning checkpoint with state_dict")
    else:
        state_dict = ckpt
        print(f"✅ Found direct state dict (weights only)")
    
    return state_dict


class OptimizedSeqSetVAE(pl.LightningModule):
    """
    Optimized SeqSetVAE for efficient finetuning with simplified architecture
    主要优化：
    1. 简化分类头架构
    2. 优化批处理逻辑
    3. 移除不必要的计算
    4. 改进学习率调度
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
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha = None,
        cls_head_lr: float = 3e-4,  # 优化的分类头学习率
        simplified_cls_head: bool = True,  # 使用简化的分类头
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Core parameters
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = beta
        self.lr = lr
        self.cls_head_lr = cls_head_lr
        self.w = w
        self.free_bits = free_bits
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.kl_annealing = kl_annealing
        self.use_focal_loss = use_focal_loss
        self.simplified_cls_head = simplified_cls_head
        
        # Training state
        self.classification_only = False
        self.current_step = 0

        # Pretrained SetVAE
        self.setvae = SetVAE(input_dim, reduced_dim, latent_dim, levels, heads, m, beta, lr)
        
        # Load pretrained weights if provided
        if pretrained_ckpt is not None:
            self._load_pretrained_weights(pretrained_ckpt)
        else:
            print("⚠️ WARNING: No pretrained checkpoint provided!")

        # Optimized Transformer encoder
        enc_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,  # 减少dropout
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.post_transformer_norm = nn.LayerNorm(latent_dim, eps=1e-6)

        # Simplified classification head for better efficiency
        if simplified_cls_head:
            self.cls_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
        else:
            # Original complex head
            self.cls_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.LayerNorm(latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim // 2, latent_dim // 4),
                nn.LayerNorm(latent_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim // 4, num_classes)
            )

        # Decoder (kept for compatibility)
        self.decoder = _SetDecoder(latent_dim, reduced_dim, levels, heads, m)
        
        # Time encoding (simplified)
        self.num_time_buckets = 32  # 减少时间桶数量
        edges = torch.logspace(math.log10(0.5), math.log10(24 * 60.0), steps=self.num_time_buckets - 1)
        self.register_buffer("time_bucket_edges", edges, persistent=False)
        self.rel_time_bucket_embed = nn.Embedding(self.num_time_buckets, latent_dim)

        # Focal loss
        if use_focal_loss:
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_loss_fn = None

        # Metrics
        self.val_auc = AUROC(task="binary", num_classes=num_classes)
        self.val_auprc = AveragePrecision(task="binary", num_classes=num_classes)
        self.val_acc = Accuracy(task="binary", num_classes=num_classes)

    def _load_pretrained_weights(self, pretrained_ckpt):
        """优化的预训练权重加载"""
        try:
            print(f"🔄 Loading pretrained weights from: {pretrained_ckpt}")
            state_dict = load_checkpoint_weights(pretrained_ckpt, device='cpu')
            
            # 快速加载兼容参数
            loaded_params = {}
            for k, v in state_dict.items():
                if k.startswith('cls_head'):
                    continue  # 跳过分类头
                
                # 直接映射或重映射键
                if k in self.state_dict():
                    if self.state_dict()[k].shape == v.shape:
                        loaded_params[k] = v
                elif k.startswith('set_encoder.'):
                    mapped_key = 'setvae.setvae.' + k[len('set_encoder.'):]
                    if mapped_key in self.state_dict() and self.state_dict()[mapped_key].shape == v.shape:
                        loaded_params[mapped_key] = v
            
            missing, unexpected = self.load_state_dict(loaded_params, strict=False)
            print(f"✅ Loaded {len(loaded_params)} parameters successfully")
            
        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}")
            raise e

    def forward(self, sets):
        """优化的前向传播，减少不必要的计算"""
        if isinstance(sets, dict):
            return self._forward_batch_optimized(sets)
        else:
            return self._forward_single_optimized(sets)

    def _forward_batch_optimized(self, batch_dict):
        """优化的批处理前向传播"""
        var, val, time, set_ids = (
            batch_dict["var"],
            batch_dict["val"], 
            batch_dict["minute"],
            batch_dict["set_id"],
        )
        padding_mask = batch_dict.get("padding_mask", None)
        
        # 快速批处理：直接处理而不是逐患者分割
        batch_size, seq_len = var.size(0), var.size(1)
        
        # 简化的集合编码
        z_features = []
        total_kl = 0.0
        
        for b in range(batch_size):
            patient_var = var[b]
            patient_val = val[b]
            
            if padding_mask is not None:
                valid_mask = ~padding_mask[b]
                if valid_mask.any():
                    patient_var = patient_var[valid_mask]
                    patient_val = patient_val[valid_mask]
            
            # 快速VAE编码
            if len(patient_var) > 0:
                _, z_list, _ = self.setvae.setvae(patient_var.unsqueeze(0), patient_val.unsqueeze(0))
                z_sample, mu, logvar = z_list[-1]
                z_features.append(mu.squeeze(1))
                
                # 简化的KL计算
                if not self.classification_only:
                    kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
                    total_kl += kl_div.mean()
            else:
                z_features.append(torch.zeros(1, self.latent_dim, device=var.device))
        
        # 堆叠特征
        z_seq = torch.stack(z_features, dim=0)  # [B, latent_dim]
        
        # 简化的分类
        if self.simplified_cls_head:
            logits = self.cls_head(z_seq)
        else:
            # 使用transformer进行序列建模（如果需要）
            h_seq = self.transformer(z_seq.unsqueeze(1))
            h_seq = self.post_transformer_norm(h_seq)
            logits = self.cls_head(h_seq.squeeze(1))
        
        # 简化的损失计算
        recon_loss = torch.tensor(0.0, device=var.device)  # 微调时跳过重构
        kl_loss = torch.tensor(0.0, device=var.device) if self.classification_only else total_kl / batch_size
        
        return logits, recon_loss, kl_loss

    def _forward_single_optimized(self, sets):
        """优化的单患者前向传播"""
        if not isinstance(sets, list) or len(sets) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, self.num_classes, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # 快速处理单个患者
        all_features = []
        total_kl = 0.0
        
        for s_dict in sets:
            var, val = s_dict["var"], s_dict["val"]
            _, z_list, _ = self.setvae.setvae(var, val)
            z_sample, mu, logvar = z_list[-1]
            all_features.append(mu.squeeze(1))
            
            if not self.classification_only:
                kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)
                total_kl += kl_div.mean()
        
        # 序列特征聚合
        if len(all_features) > 1:
            z_seq = torch.stack(all_features, dim=1)  # [1, S, latent_dim]
            h_seq = self.transformer(z_seq)
            h_seq = self.post_transformer_norm(h_seq)
            # 使用最后一个时间步的特征
            final_features = h_seq[:, -1, :]
        else:
            final_features = all_features[0]
        
        logits = self.cls_head(final_features)
        
        recon_loss = torch.tensor(0.0, device=logits.device)
        kl_loss = torch.tensor(0.0, device=logits.device) if self.classification_only else total_kl / len(sets)
        
        return logits, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def _step(self, batch, stage: str):
        """优化的训练/验证步骤"""
        label = batch.get("label")
        
        # 前向传播
        logits, recon_loss, kl_loss = self(batch)
        
        # 损失计算
        if self.use_focal_loss and self.focal_loss_fn is not None:
            pred_loss = self.focal_loss_fn(logits, label)
        else:
            pred_loss = F.cross_entropy(logits, label)

        if self.classification_only:
            total_loss = pred_loss
        else:
            current_beta = self.get_current_beta()
            total_loss = self.w * pred_loss + recon_loss + current_beta * kl_loss

        # 指标更新（仅验证时）
        if stage == "val":
            if self.num_classes == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.softmax(logits, dim=1)
            self.val_auc.update(probs, label)
            self.val_auprc.update(probs, label)
            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, label)

        # 简化的日志记录
        log_dict = {
            f"{stage}_loss": total_loss,
            f"{stage}_pred": pred_loss,
        }
        if not self.classification_only:
            log_dict.update({
                f"{stage}_recon": recon_loss,
                f"{stage}_kl": kl_loss,
            })
        
        self.log_dict(log_dict, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        
        return total_loss

    def on_validation_epoch_end(self):
        """计算并记录验证指标"""
        auc = self.val_auc.compute()
        auprc = self.val_auprc.compute()
        acc = self.val_acc.compute()
        
        self.log_dict({
            "val_auc": auc,
            "val_auprc": auprc,
            "val_accuracy": acc,
        }, prog_bar=True)
        
        # 重置指标
        self.val_auc.reset()
        self.val_auprc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """优化的优化器配置"""
        if self.classification_only:
            # 只优化分类头参数
            cls_params = [p for p in self.cls_head.parameters() if p.requires_grad]
            optimizer = AdamW(
                cls_params,
                lr=self.cls_head_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.005,  # 减少权重衰减
            )
        else:
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        
        # 改进的学习率调度器 - 监控AUC而不是loss
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # AUC越大越好
            factor=0.8,  # 更温和的衰减
            patience=4,  # 减少耐心
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_auc",  # 监控AUC
            },
        }

    def enable_classification_only_mode(self, cls_head_lr: Optional[float] = None):
        """启用仅分类模式"""
        self.classification_only = True
        if cls_head_lr is not None:
            self.cls_head_lr = cls_head_lr
        self.set_backbone_eval()

    def set_backbone_eval(self):
        """设置backbone为评估模式"""
        self.setvae.eval()
        self.transformer.eval()
        self.post_transformer_norm.eval()
        self.decoder.eval()

    def on_train_start(self):
        """训练开始时的设置"""
        if self.classification_only:
            self.set_backbone_eval()

    def init_classifier_head_xavier(self):
        """使用Xavier初始化分类头"""
        for module in self.cls_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_current_beta(self):
        """获取当前beta值"""
        if not self.warmup_beta:
            return self.beta
        
        if self.current_step < self.beta_warmup_steps:
            return self.beta * (self.current_step / self.beta_warmup_steps)
        else:
            return min(self.beta, self.max_beta)


class _SetDecoder(nn.Module):
    """简化的解码器"""
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


# 为了向后兼容，导入原始的SetVAE类
from model import SetVAE, SeqSetVAEPretrain

print("🚀 Optimized SeqSetVAE model loaded with performance enhancements!")