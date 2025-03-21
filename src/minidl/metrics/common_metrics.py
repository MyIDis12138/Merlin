import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Any

from .metrics_registry import MetricRegistry


@MetricRegistry.register("accuracy")
def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels

    Returns:
        Accuracy score
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    return float(accuracy_score(targets, preds))


@MetricRegistry.register("f1_score")
def compute_f1_score(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    average: str = "macro"
) -> float:
    """Compute F1 score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        average: Averaging strategy ('micro', 'macro', 'weighted', etc.)

    Returns:
        F1 score
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Handle case where only one class is present in targets
    try:
        return float(f1_score(targets, preds, average=average, zero_division=0))
    except:
        # Fallback to accuracy if f1_score fails
        return float(accuracy_score(targets, preds))


@MetricRegistry.register("precision")
def compute_precision(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    average: str = "macro"
) -> float:
    """Compute precision score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        average: Averaging strategy ('micro', 'macro', 'weighted', etc.)

    Returns:
        Precision score
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    try:
        return float(precision_score(targets, preds, average=average, zero_division=0))
    except:
        # Fallback to accuracy if precision_score fails
        return float(accuracy_score(targets, preds))


@MetricRegistry.register("recall")
def compute_recall(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    average: str = "macro"
) -> float:
    """Compute recall score.

    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        average: Averaging strategy ('micro', 'macro', 'weighted', etc.)

    Returns:
        Recall score
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    try:
        return float(recall_score(targets, preds, average=average, zero_division=0))
    except:
        # Fallback to accuracy if recall_score fails
        return float(accuracy_score(targets, preds))


@MetricRegistry.register("auc")
def compute_auc(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    multi_class: str = "ovr"
) -> float:
    """Compute area under ROC curve.

    Args:
        outputs: Model outputs (probabilities)
        targets: Ground truth labels
        multi_class: Strategy for multi-class classification ('ovr' or 'ovo')

    Returns:
        AUC score
    """
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Check if we have enough classes and samples
    n_classes = probs.shape[1]
    
    if n_classes <= 1:
        return 0.0
        
    # For binary classification
    if n_classes == 2:
        try:
            return float(roc_auc_score(targets, probs[:, 1]))
        except:
            return 0.0
    
    # For multi-class classification
    try:
        return float(roc_auc_score(targets, probs, multi_class=multi_class))
    except:
        # Fallback to 0.5 (random classifier) if AUC computation fails
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
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        metric_kwargs: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, float]:
        """Compute all registered metrics.
        
        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
            metric_kwargs: Optional dict of kwargs for specific metrics
            
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
                results[name] = 0.0
                
        return results