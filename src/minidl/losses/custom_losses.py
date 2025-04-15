import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_registry import LossRegistry


@LossRegistry.register("balanced_ce")
class Balanced_CrossEntropy(nn.Module):
    """Balanced Cross Entropy Loss.

    Implementation of Cross Entropy Loss with automatic class weighting
    to handle imbalanced datasets.

    Args:
        weight_type: Strategy for computing class weights ('inverse', 'effective', or None)
        beta: Parameter for effective number weighting when weight_type='effective'
        reduction: Reduction method ('mean', 'sum', or 'none')
        epsilon: Small value to prevent division by zero
    """

    def __init__(self, weight_type: str = "inverse", beta: float = 0.9, reduction: str = "mean", epsilon: float = 1e-6):
        super().__init__()
        self.weight_type = weight_type
        self.beta = beta
        self.reduction = reduction
        self.epsilon = epsilon
        self.register_buffer("class_weights", None)

    def _compute_class_weights(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Compute class weights based on class distribution in the current batch.

        Args:
            targets: Ground truth labels
            num_classes: Number of classes

        Returns:
            Tensor of class weights
        """
        class_counts = torch.zeros(num_classes, device=targets.device)
        for c in range(num_classes):
            class_counts[c] = (targets == c).sum().float()

        class_counts = class_counts + self.epsilon

        if self.weight_type == "inverse":
            class_weights = 1.0 / class_counts
            class_weights = class_weights * (num_classes / class_weights.sum())
        elif self.weight_type == "effective":
            # Effective number of samples (from "Class-Balanced Loss Based on Effective Number of Samples")
            # https://arxiv.org/abs/1901.05555
            effective_num = 1.0 - torch.pow(self.beta, class_counts)
            class_weights = (1.0 - self.beta) / (effective_num + self.epsilon)
            class_weights = class_weights / class_weights.sum() * num_classes
        else:
            class_weights = torch.ones_like(class_counts)

        return class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Computed loss
        """
        num_classes = inputs.size(1)

        # Compute class weights based on the batch
        class_weights = self._compute_class_weights(targets, num_classes)
        self.class_weights = class_weights  # Store for possible inspection

        # Apply weighted cross entropy loss
        loss = F.cross_entropy(inputs, targets, weight=class_weights, reduction=self.reduction)

        return loss


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

    def __init__(self, alpha: float | list = 0.25, gamma: float = 2.0, reduction: str = "mean", epsilon=1e-6):
        super().__init__()

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, (float, int)):
            if not 0 <= alpha <= 1:
                print(f"Warning: Scalar alpha={alpha} is outside the typical [0, 1] range.")
            self.alpha = float(alpha)
        else:
            raise TypeError("Argument 'alpha' must be float, list, tuple.")

        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Predicted logits
            targets: Ground truth labels

        Returns:
            Computed loss
        """
        num_classes = inputs.shape[1]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, float):
            alpha_factor = self.alpha
        elif isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            if len(self.alpha) != num_classes:
                raise ValueError(f"Length of alpha tensor ({len(self.alpha)}) " f"must match number of classes ({num_classes}).")

            alpha_factor = self.alpha[targets]
        else:
            raise TypeError("Internal error: self.alpha is not float or tensor.")

        focal_loss = alpha_factor * (1 - pt + self.epsilon) ** self.gamma * ce_loss

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
