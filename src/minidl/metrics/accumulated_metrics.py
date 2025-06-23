import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class AccumulatedMetricsCalculator(ABC):
    """
    Base class for computing accumulated metrics across batches.
    
    This class accumulates predictions and targets across all batches,
    then computes metrics on the complete dataset. This is especially
    important for metrics like precision, recall, and AUC that can be
    misleading when averaged across batches.
    """

    def __init__(self, average: str = "weighted"):
        """Initialize the accumulated metrics calculator.

        Args:
            average: Averaging strategy for multi-class metrics ('micro', 'macro', 'weighted', etc.)
        """
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.all_outputs = []
        self.all_targets = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Accumulate batch outputs and targets.
        
        Args:
            outputs: Model outputs (logits or probabilities)
            targets: Ground truth labels
        """
        self.all_outputs.append(outputs.detach().cpu())
        self.all_targets.append(targets.detach().cpu())
    
    def get_current_size(self) -> int:
        """Get the current number of accumulated samples."""
        return sum(targets.shape[0] for targets in self.all_targets)
    
    def get_predictions_and_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the accumulated predictions and targets.
        
        Returns:
            Tuple of (all_outputs, all_targets)
        """
        if not self.all_outputs or not self.all_targets:
            return torch.tensor([]), torch.tensor([])
        
        all_outputs = torch.cat(self.all_outputs, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        
        return all_outputs, all_targets
    
    def _prepare_predictions_and_probabilities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare predictions and probabilities from accumulated outputs.
        
        Returns:
            Tuple of (targets_np, preds, probs)
        """
        if not self.all_outputs or not self.all_targets:
            raise ValueError("No data accumulated for metric computation")
        
        all_outputs = torch.cat(self.all_outputs, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        
        targets_np = all_targets.cpu().numpy()
        
        # Handle different output shapes
        if all_outputs.ndim > 1 and all_outputs.shape[1] > 1:
            # Multi-class case
            preds = all_outputs.argmax(dim=1).cpu().numpy()
            probs = torch.nn.functional.softmax(all_outputs, dim=1).cpu().numpy()
        elif all_outputs.ndim == 1 or all_outputs.shape[1] == 1:
            # Binary case
            probs = torch.sigmoid(all_outputs.squeeze()).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            # For binary case, create 2D probability array for consistency
            probs = np.column_stack([1 - probs, probs])
        else:
            raise ValueError(f"Unsupported output shape for accumulated metrics: {all_outputs.shape}")
        
        return targets_np, preds, probs
    
    @abstractmethod
    def compute_metric(self) -> float:
        """Compute the specific metric on accumulated data.
        
        Returns:
            The computed metric value
        """
        pass
    
    def compute(self) -> float:
        """Compute the metric with error handling.
        
        Returns:
            The computed metric value with fallback on errors
        """
        try:
            return self.compute_metric()
        except Exception as e:
            logger.error(f"Error computing accumulated metric {self.__class__.__name__}: {e}")
            return self._get_fallback_value()
    
    def _get_fallback_value(self) -> float:
        """Get fallback value for failed metric computation."""
        return 0.0


class AccumulatedPrecision(AccumulatedMetricsCalculator):
    """Accumulated precision metric calculator."""
    
    def compute_metric(self) -> float:
        """Compute precision on accumulated data."""
        targets_np, preds, _ = self._prepare_predictions_and_probabilities()
        
        unique_targets = np.unique(targets_np)
        unique_preds = np.unique(preds)
        
        if len(unique_targets) < 2 and len(unique_preds) < 2:
            return float(accuracy_score(targets_np, preds))
        
        return float(precision_score(targets_np, preds, average=self.average, zero_division=0))


class AccumulatedRecall(AccumulatedMetricsCalculator):
    """Accumulated recall metric calculator."""
    
    def compute_metric(self) -> float:
        """Compute recall on accumulated data."""
        targets_np, preds, _ = self._prepare_predictions_and_probabilities()
        
        unique_targets = np.unique(targets_np)
        unique_preds = np.unique(preds)
        
        if len(unique_targets) < 2 and len(unique_preds) < 2:
            return float(accuracy_score(targets_np, preds))
        
        return float(recall_score(targets_np, preds, average=self.average, zero_division=0))


class AccumulatedAUC(AccumulatedMetricsCalculator):
    """Accumulated AUC metric calculator."""
    
    def compute_metric(self) -> float:
        """Compute AUC on accumulated data."""
        targets_np, _, probs = self._prepare_predictions_and_probabilities()
        
        unique_classes = np.unique(targets_np)
        if len(unique_classes) <= 1:
            logger.warning("AUC is not defined for samples from only one class. Returning 0.0.")
            return 0.0
        
        if probs.shape[1] == 2:
            # Binary case
            return float(roc_auc_score(targets_np, probs[:, 1]))
        else:
            # Multi-class case
            return float(roc_auc_score(targets_np, probs, multi_class='ovr', average='weighted'))
    
    def _get_fallback_value(self) -> float:
        """Get fallback value for failed AUC computation."""
        return 0.5  # Random guess baseline


class AccumulatedAccuracy(AccumulatedMetricsCalculator):
    """Accumulated accuracy metric calculator."""
    
    def compute_metric(self) -> float:
        """Compute accuracy on accumulated data."""
        targets_np, preds, _ = self._prepare_predictions_and_probabilities()
        return float(accuracy_score(targets_np, preds))


class AccumulatedF1Score(AccumulatedMetricsCalculator):
    """Accumulated F1 score metric calculator."""
    
    def compute_metric(self) -> float:
        """Compute F1 score on accumulated data."""
        targets_np, preds, _ = self._prepare_predictions_and_probabilities()
        
        unique_targets = np.unique(targets_np)
        unique_preds = np.unique(preds)
        
        if len(unique_targets) < 2 and len(unique_preds) < 2:
            return float(accuracy_score(targets_np, preds))
        
        return float(f1_score(targets_np, preds, average=self.average, zero_division=0))


class MultiAccumulatedMetricsCalculator:
    """
    Utility class for computing multiple accumulated metrics at once.
    """
    
    def __init__(self, metrics_config: dict[str, dict[str, Any]] | None = None):
        """Initialize with multiple metric calculators.
        
        Args:
            metrics_config: Dict mapping metric names to their config.
                           Example: {'precision': {'average': 'weighted'}, 'recall': {'average': 'macro'}}
        """
        if metrics_config is None:
            metrics_config = {
                'precision': {'average': 'weighted'},
                'recall': {'average': 'weighted'},
                'auc': {'average': 'weighted'}
            }
        
        self.metrics = {}
        for metric_name, config in metrics_config.items():
            self.metrics[metric_name] = self._create_metric_calculator(metric_name, config)
    
    def _create_metric_calculator(self, metric_name: str, config: dict[str, Any]) -> AccumulatedMetricsCalculator:
        """Create a specific metric calculator."""
        metric_classes = {
            'precision': AccumulatedPrecision,
            'recall': AccumulatedRecall,
            'auc': AccumulatedAUC,
            'accuracy': AccumulatedAccuracy,
            'f1_score': AccumulatedF1Score,
        }
        
        if metric_name not in metric_classes:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        return metric_classes[metric_name](**config)
    
    def reset(self):
        """Reset all metric calculators."""
        for metric_calc in self.metrics.values():
            metric_calc.reset()
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Update all metric calculators with batch data."""
        for metric_calc in self.metrics.values():
            metric_calc.update(outputs, targets)
    
    def compute_metrics(self) -> dict[str, float]:
        """Compute all metrics."""
        results = {}
        for metric_name, metric_calc in self.metrics.items():
            results[metric_name] = metric_calc.compute()
        return results
    
    def get_current_size(self) -> int:
        """Get the current number of accumulated samples."""
        if self.metrics:
            # All metrics should have the same size, so we can use any of them
            return next(iter(self.metrics.values())).get_current_size()
        return 0 