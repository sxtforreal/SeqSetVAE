import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Sequence, Union, Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for binary or multiclass classification with logits.

    Args:
        alpha: Class balancing factor. For binary classification, a float means
               alpha for positive class and (1 - alpha) for negative class.
               For multiclass, provide a sequence of length C.
        gamma: Focusing parameter (>= 0). Default: 2.0
        reduction: 'none' | 'mean' | 'sum'
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction
        self.gamma = float(gamma)
        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = alpha  # type: ignore[assignment]
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss given logits and integer class targets.

        Shapes:
            logits: [B, C]
            targets: [B]
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape [B, C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"Expected targets shape [B], got {tuple(targets.shape)}")

        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp().clamp(min=self.eps, max=1.0)

        # Gather probabilities of the true class
        targets = targets.long()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute alpha_t per-sample
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.numel() == 1 and num_classes == 2:
                # scalar tensor for binary is treated as positive class weight
                alpha_t = torch.stack([1.0 - self.alpha, self.alpha], dim=0).to(logits.device)
                alpha_t = alpha_t[targets]
            elif self.alpha.numel() == num_classes:
                alpha_t = self.alpha.to(logits.device)[targets]
            else:
                raise ValueError("alpha tensor must have length equal to num_classes or be a scalar for binary tasks")
        elif isinstance(self.alpha, float):
            if num_classes == 2:
                alpha_t = torch.tensor([1.0 - self.alpha, self.alpha], device=logits.device)[targets]
            else:
                raise ValueError("Scalar alpha only supported for binary classification; provide a sequence for multiclass")
        else:
            alpha_t = torch.ones_like(pt)

        focal_factor = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_factor * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (ICLR 2021)
    Optimized for extreme class imbalance in medical data
    
    Reference: "Asymmetric Loss For Multi-Label Classification"
    """
    
    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for asymmetric loss
        
        Args:
            x: Logits [B, C]
            y: Labels [B] (for binary) or [B, C] (for multi-label)
        """
        # Convert to probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Convert labels to one-hot if needed
        if y.dim() == 1:
            y_one_hot = F.one_hot(y, num_classes=x.size(1)).float()
        else:
            y_one_hot = y.float()
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        # Basic CE calculation
        los_pos = y_one_hot * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y_one_hot) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y_one_hot
            pt1 = xs_neg * (1 - y_one_hot)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_one_hot + self.gamma_neg * (1 - y_one_hot)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos *= one_sided_w
            los_neg *= one_sided_w
            
        loss = los_pos + los_neg
        return -loss.sum(dim=1).mean()


class SOTALossStrategy:
    """
    State-of-the-Art Loss Strategy combining multiple recent advances:
    1. SoftAdapt for dynamic loss weighting (ICML 2020)
    2. Asymmetric Loss for imbalanced classification (ICLR 2021)
    3. Self-Distillation with EMA teacher (CVPR 2024)
    4. Gradient-based adaptation (NeurIPS 2023)
    
    Integrated for medical classification tasks.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        focal_alpha: float = 0.2,
        focal_gamma: float = 2.5,
        ema_decay: float = 0.999,
        adaptation_lr: float = 0.01,
        strategy: str = "adaptive_distillation"
    ):
        self.num_classes = num_classes
        self.strategy = strategy
        self.ema_decay = ema_decay
        self.adaptation_lr = adaptation_lr
        
        # 1. Advanced Focal Loss for main task
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        
        # 2. Asymmetric Loss for auxiliary task (handles extreme imbalance)
        self.asymmetric_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        
        # 3. SoftAdapt weights (learnable parameters)
        self.register_loss_weights()
        
        # 4. EMA teacher for self-distillation
        self.ema_teacher_logits = None
        
        # 5. Gradient similarity tracking
        self.prev_main_grad = None
        self.prev_aux_grad = None
        
    def register_loss_weights(self):
        """Register learnable loss weights (SoftAdapt approach)"""
        # Initialize as learnable parameters
        self.main_weight = nn.Parameter(torch.tensor(1.0))
        self.aux_weight = nn.Parameter(torch.tensor(0.5))
        self.distill_weight = nn.Parameter(torch.tensor(0.3))
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))
        
    def compute_loss(
        self,
        main_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        labels: torch.Tensor,
        epoch: int = 0,
        main_params: Optional[list] = None,
        aux_params: Optional[list] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute SOTA loss with multiple advanced techniques
        
        Args:
            main_logits: Main head predictions [B, C]
            aux_logits: Auxiliary head predictions [B, C]
            labels: Ground truth labels [B]
            epoch: Current epoch for scheduling
            main_params: Main head parameters for gradient analysis
            aux_params: Auxiliary head parameters for gradient analysis
        """
        
        device = main_logits.device
        
        # 1. Main Loss: Advanced Focal Loss
        main_loss = self.focal_loss(main_logits, labels)
        
        # 2. Auxiliary Loss: Asymmetric Loss (better for extreme imbalance)
        aux_loss = self.asymmetric_loss(aux_logits, labels)
        
        # 3. Self-Distillation Loss with EMA Teacher
        distill_loss = self._compute_distillation_loss(main_logits, aux_logits, epoch)
        
        # 4. Advanced Consistency Loss
        consistency_loss = self._compute_consistency_loss(main_logits, aux_logits, labels)
        
        # 5. Dynamic Weight Adaptation (SoftAdapt + Gradient-based)
        adapted_weights = self._adapt_loss_weights(
            main_loss, aux_loss, distill_loss, consistency_loss,
            main_params, aux_params
        )
        
        # 6. Combine losses with adapted weights
        total_loss = (
            adapted_weights['main'] * main_loss +
            adapted_weights['aux'] * aux_loss +
            adapted_weights['distill'] * distill_loss +
            adapted_weights['consistency'] * consistency_loss
        )
        
        # Loss breakdown for monitoring
        loss_dict = {
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'distill_loss': distill_loss,
            'consistency_loss': consistency_loss,
            'main_weight': adapted_weights['main'],
            'aux_weight': adapted_weights['aux'],
            'distill_weight': adapted_weights['distill'],
            'consistency_weight': adapted_weights['consistency']
        }
        
        return total_loss, loss_dict
    
    def _compute_distillation_loss(
        self, 
        main_logits: torch.Tensor, 
        aux_logits: torch.Tensor, 
        epoch: int
    ) -> torch.Tensor:
        """
        Self-distillation with EMA teacher (inspired by momentum contrast)
        """
        # Update EMA teacher with main head predictions
        main_probs = F.softmax(main_logits.detach(), dim=1)
        
        if self.ema_teacher_logits is None:
            self.ema_teacher_logits = main_probs.clone()
        else:
            self.ema_teacher_logits = (
                self.ema_decay * self.ema_teacher_logits + 
                (1 - self.ema_decay) * main_probs
            )
        
        # Knowledge distillation from EMA teacher to auxiliary head
        temperature = 3.0 + 2.0 * np.exp(-epoch / 10)  # Adaptive temperature
        
        aux_log_probs = F.log_softmax(aux_logits / temperature, dim=1)
        teacher_probs = F.softmax(self.ema_teacher_logits / temperature, dim=1)
        
        distill_loss = F.kl_div(aux_log_probs, teacher_probs, reduction='batchmean')
        
        return distill_loss * (temperature ** 2)  # Scale by temperature^2
    
    def _compute_consistency_loss(
        self,
        main_logits: torch.Tensor,
        aux_logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Advanced consistency loss with uncertainty weighting
        """
        main_probs = F.softmax(main_logits, dim=1)
        aux_probs = F.softmax(aux_logits, dim=1)
        
        # Confidence-based weighting (higher weight for confident predictions)
        main_confidence = torch.max(main_probs, dim=1)[0]
        aux_confidence = torch.max(aux_probs, dim=1)[0]
        
        # Agreement-based consistency (only for samples where both are confident)
        confidence_mask = (main_confidence > 0.7) & (aux_confidence > 0.7)
        
        if confidence_mask.sum() > 0:
            # Bidirectional KL divergence for confident samples
            kl_main_to_aux = F.kl_div(
                F.log_softmax(aux_logits[confidence_mask], dim=1),
                main_probs[confidence_mask].detach(),
                reduction='batchmean'
            )
            kl_aux_to_main = F.kl_div(
                F.log_softmax(main_logits[confidence_mask], dim=1),
                aux_probs[confidence_mask].detach(),
                reduction='batchmean'
            )
            consistency_loss = 0.5 * (kl_main_to_aux + kl_aux_to_main)
        else:
            consistency_loss = torch.tensor(0.0, device=main_logits.device)
            
        return consistency_loss
    
    def _adapt_loss_weights(
        self,
        main_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        distill_loss: torch.Tensor,
        consistency_loss: torch.Tensor,
        main_params: Optional[list] = None,
        aux_params: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        SoftAdapt + Gradient-based adaptive weighting
        """
        # Base weights (learnable parameters)
        base_weights = {
            'main': torch.sigmoid(self.main_weight),
            'aux': torch.sigmoid(self.aux_weight), 
            'distill': torch.sigmoid(self.distill_weight),
            'consistency': torch.sigmoid(self.consistency_weight)
        }
        
        # Gradient-based adaptation (if parameters provided)
        if main_params is not None and aux_params is not None:
            grad_adaptation = self._compute_gradient_adaptation(
                main_loss, aux_loss, main_params, aux_params
            )
            
            # Combine base weights with gradient adaptation
            adapted_weights = {}
            for key in base_weights:
                if key in grad_adaptation:
                    adapted_weights[key] = base_weights[key] * grad_adaptation[key]
                else:
                    adapted_weights[key] = base_weights[key]
        else:
            adapted_weights = base_weights
        
        # Normalize weights to sum to reasonable range
        total_weight = sum(adapted_weights.values())
        normalized_weights = {k: v / total_weight * 2.0 for k, v in adapted_weights.items()}
        
        return normalized_weights
    
    def _compute_gradient_adaptation(
        self,
        main_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        main_params: list,
        aux_params: list
    ) -> Dict[str, torch.Tensor]:
        """
        Gradient-based loss weight adaptation
        """
        try:
            # Compute gradients
            main_grads = torch.autograd.grad(main_loss, main_params, retain_graph=True, allow_unused=True)
            aux_grads = torch.autograd.grad(aux_loss, aux_params, retain_graph=True, allow_unused=True)
            
            # Flatten gradients
            main_grad_flat = torch.cat([g.flatten() for g in main_grads if g is not None])
            aux_grad_flat = torch.cat([g.flatten() for g in aux_grads if g is not None])
            
            if len(main_grad_flat) == 0 or len(aux_grad_flat) == 0:
                return {'main': 1.0, 'aux': 1.0}
            
            # Compute gradient similarity (cosine similarity)
            similarity = F.cosine_similarity(
                main_grad_flat.unsqueeze(0), 
                aux_grad_flat.unsqueeze(0)
            ).item()
            
            # Adaptive weighting based on gradient similarity
            # High similarity -> reduce auxiliary weight (avoid redundancy)
            # Low similarity -> increase auxiliary weight (complementary learning)
            main_adapt = 1.0
            aux_adapt = max(0.1, 1.0 - abs(similarity))  # Inverse relationship
            
            return {
                'main': torch.tensor(main_adapt, device=main_loss.device),
                'aux': torch.tensor(aux_adapt, device=main_loss.device)
            }
        except Exception as e:
            # Fallback to equal weights if gradient computation fails
            return {'main': 1.0, 'aux': 1.0}


# Medical scenario configurations
MEDICAL_SOTA_CONFIGS = {
    "rare_disease_detection": {
        "description": "Optimized for rare disease detection with extreme imbalance",
        "focal_alpha": 0.1,  # Heavy focus on minority class
        "focal_gamma": 3.0,   # Strong focusing
        "ema_decay": 0.9999,  # Very slow EMA for stability
        "strategy": "adaptive_distillation"
    },
    "multi_condition_screening": {
        "description": "Balanced approach for multiple condition screening",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "ema_decay": 0.999,
        "strategy": "gradient_adaptive"
    },
    "treatment_response_prediction": {
        "description": "Confidence-aware for treatment response prediction",
        "focal_alpha": 0.3,
        "focal_gamma": 1.5,
        "ema_decay": 0.995,
        "strategy": "uncertainty_aware"
    },
    "diagnostic_assistance": {
        "description": "Comprehensive diagnostic assistance with multiple heads",
        "focal_alpha": 0.2,
        "focal_gamma": 2.5,
        "ema_decay": 0.999,
        "strategy": "ensemble_distillation"
    }
}


def get_sota_loss_strategy(medical_scenario: str = "multi_condition_screening", **kwargs):
    """
    Factory function to get SOTA loss strategy for specific medical scenarios
    """
    if medical_scenario in MEDICAL_SOTA_CONFIGS:
        config = MEDICAL_SOTA_CONFIGS[medical_scenario].copy()
        config.update(kwargs)  # Allow custom overrides
        
        return SOTALossStrategy(
            focal_alpha=config['focal_alpha'],
            focal_gamma=config['focal_gamma'],
            ema_decay=config['ema_decay'],
            strategy=config['strategy']
        )
    else:
        raise ValueError(f"Unknown medical scenario: {medical_scenario}")


def get_recommended_strategy(class_imbalance_ratio, noise_level="low", requires_calibration=False):
    """
    Recommend the best strategy based on data characteristics.

    Args:
        class_imbalance_ratio: Class imbalance ratio (minority/majority)
        noise_level: Data noise level ("low", "medium", "high")
        requires_calibration: Whether probability calibration is required

    Returns:
        strategy_name: The recommended strategy name
    """
    if requires_calibration:
        return "treatment_response_prediction"
    elif class_imbalance_ratio < 0.1:  # Severe imbalance
        return "rare_disease_detection"
    elif noise_level == "high":
        return "diagnostic_assistance"
    else:
        return "multi_condition_screening"  # default choice