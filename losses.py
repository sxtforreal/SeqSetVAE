import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union


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