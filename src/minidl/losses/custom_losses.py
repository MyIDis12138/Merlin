import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_registry import LossRegistry


@LossRegistry.register("focal_loss")
class FocalLoss(nn.Module):
    """Focal Loss.

    Implementation of Focal Loss from the paper
    "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).

    Args:
        alpha: Weighting factor for the rare class
        gamma: Focusing parameter
        reduction: Reduction method
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Predicted logits
            targets: Ground truth labels

        Returns:
            Computed loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


@LossRegistry.register("dice_loss")
class DiceLoss(nn.Module):
    """Dice Loss.

    Implementation of Dice Loss commonly used in medical image segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Predicted probabilities (after sigmoid/softmax)
            targets: Ground truth labels

        Returns:
            Computed loss
        """
        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss
        return 1.0 - dice
