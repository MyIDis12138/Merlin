import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .metrics_registry import MetricRegistry

logger = logging.getLogger(__name__)


@MetricRegistry.register("accuracy")
def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy score.

    Args:
        outputs: Model outputs (logits or probabilities)
        targets: Ground truth labels

    Returns:
        Accuracy score
    """
    targets_np = targets.cpu().numpy()
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        preds = outputs.argmax(dim=1).cpu().numpy()
    elif outputs.ndim == 1 or outputs.shape[1] == 1:
        probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    else:
        logger.error(f"Unsupported output shape for accuracy: {outputs.shape}")
        return 0.0

    return float(accuracy_score(targets_np, preds))


@MetricRegistry.register("f1_score")
def compute_f1_score(outputs: torch.Tensor, targets: torch.Tensor, average: str = "weighted") -> float:
    """Compute F1 score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        average: Averaging strategy ('micro', 'macro', 'weighted', etc.)

    Returns:
        F1 score, or accuracy if F1 calculation fails.
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    unique_targets = np.unique(targets_np)
    unique_preds = np.unique(preds)

    if len(unique_targets) < 2 and len(unique_preds) < 2:
        # logger.warning("F1 score is not well-defined for single-class data/predictions. Falling back to accuracy.")
        return float(accuracy_score(targets_np, preds))

    try:
        return float(f1_score(targets_np, preds, average=average, zero_division=0))
    except ValueError as e:
        logger.warning(f"Could not compute F1 score (average='{average}'): {e}. Falling back to accuracy.")
        return float(accuracy_score(targets_np, preds))


@MetricRegistry.register("precision")
def compute_precision(outputs: torch.Tensor, targets: torch.Tensor, average: str = "weighted") -> float:
    """Compute precision score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        average: Averaging strategy ('micro', 'macro', 'weighted', etc.)

    Returns:
        Precision score, or accuracy if precision calculation fails.
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    unique_targets = np.unique(targets_np)
    unique_preds = np.unique(preds)

    if len(unique_targets) < 2 and len(unique_preds) < 2:
        logger.warning("Precision score is not well-defined for single-class data/predictions. Falling back to accuracy.")
        return float(accuracy_score(targets_np, preds))

    try:
        return float(precision_score(targets_np, preds, average=average, zero_division=0))
    except ValueError as e:
        logger.warning(f"Could not compute Precision score (average='{average}'): {e}. Falling back to accuracy.")
        # Fallback to accuracy if precision_score fails
        return float(accuracy_score(targets_np, preds))


@MetricRegistry.register("recall")
def compute_recall(outputs: torch.Tensor, targets: torch.Tensor, average: str = "weighted") -> float:
    """Compute recall score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        average: Averaging strategy ('micro', 'macro', 'weighted', etc.)

    Returns:
        Recall score, or accuracy if recall calculation fails.
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    unique_targets = np.unique(targets_np)
    unique_preds = np.unique(preds)

    if len(unique_targets) < 2 and len(unique_preds) < 2:
        logger.warning("Recall score is not well-defined for single-class data/predictions. Falling back to accuracy.")
        return float(accuracy_score(targets_np, preds))

    try:
        return float(recall_score(targets_np, preds, average=average, zero_division=0))
    except ValueError as e:
        logger.warning(f"Could not compute Recall score (average='{average}'): {e}. Falling back to accuracy.")
        # Fallback to accuracy if recall_score fails
        return float(accuracy_score(targets_np, preds))


@MetricRegistry.register("auc")
def compute_auc(outputs: torch.Tensor, targets: torch.Tensor, multi_class: str = "ovr") -> float:
    """Compute area under ROC curve. Handles binary and multi-class cases.

    Args:
        outputs: Model outputs (logits). Assumes shape (batch_size, num_classes).
                 For binary, num_classes can be 1 or 2.
        targets: Ground truth labels. Shape (batch_size,).
        multi_class: Strategy for multi-class classification ('ovr' or 'ovo').

    Returns:
        AUC score, 0.0 if only one class present in targets, 0.5 if calculation fails otherwise.
    """
    targets_np = targets.cpu().numpy()

    unique_classes = np.unique(targets_np)
    if len(unique_classes) <= 1:
        logger.warning("AUC is not defined for samples from only one class. Returning 0.0.")
        return 0.0

    # Convert logits to probabilities
    if outputs.ndim == 1 or outputs.shape[1] == 1:
        probs = torch.sigmoid(outputs).cpu().numpy()
        n_classes = 2
    else:
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        n_classes = probs.shape[1]

    if n_classes <= 1:
        # This case should ideally be caught by unique_classes check, but added for safety
        logger.warning("AUC calculation requires probabilities for at least 2 classes. Returning 0.0.")
        return 0.0

    try:
        if probs.ndim == 2 and probs.shape[1] == 2:
            return float(roc_auc_score(targets_np, probs[:, 1]))
        elif probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
            return float(roc_auc_score(targets_np, probs.squeeze()))

        elif n_classes > 2:
            return float(roc_auc_score(targets_np, probs, multi_class=multi_class, average="weighted"))
        else:
            logger.error(f"Unexpected probability shape for AUC calculation: {probs.shape}")
            return 0.5  # Fallback for unexpected state

    except ValueError as e:
        logger.warning(f"Could not compute AUC score: {e}. Returning 0.5 (random guess baseline).")
        return 0.5


class MetricsCalculator:
    """Utility class for computing multiple metrics at once."""

    def __init__(self, metrics_dict: dict[str, callable]):
        """Initialize with dict of metric functions.

        Args:
            metrics_dict: dictionary mapping metric names to metric functions
        """
        self.metrics_dict = metrics_dict

    def compute_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor, metric_kwargs: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, float]:
        """Compute all registered metrics.

        Args:
            outputs: Model outputs (logits or probabilities depending on metric)
            targets: Ground truth labels
            metric_kwargs: Optional dict of kwargs for specific metrics,
                           e.g., {'f1_score': {'average': 'macro'}}

        Returns:
            dictionary of metric name to score
        """
        results = {}
        metric_kwargs = metric_kwargs or {}

        for name, metric_fn in self.metrics_dict.items():
            kwargs = metric_kwargs.get(name, {})
            try:
                results[name] = metric_fn(outputs, targets, **kwargs)
            except Exception as e:
                logger.error(f"Failed to compute metric '{name}': {e}", exc_info=True)
                results[name] = 0.0

        return results
