import functools
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, cast

import torch
from tqdm import tqdm

from .base_runner import BaseRunner
from .runner_registry import RunnerRegistry

T = TypeVar("T")


class PrioritizedHook(Protocol):
    priority: int


def ensure_model_initialized(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure model is initialized before calling the function."""

    @functools.wraps(func)
    def wrapper(self: "EpochBasedRunner", *args: Any, **kwargs: Any) -> T:
        if self.model is None:
            self.logger.error(f"Cannot execute {func.__name__}: model is None")
            if func.__name__.endswith("_step"):
                step_metrics = {"loss": torch.tensor(float("inf"), device=self.device)}
                if func.__name__ == "val_step" or func.__name__ == "test_step":
                    step_metrics["accuracy"] = torch.tensor(0.0, device=self.device)
                return step_metrics  # type: ignore
            elif func.__name__.endswith("_epoch"):
                epoch_metrics: Dict[str, float] = {"loss": float("inf")}
                if func.__name__ == "val_epoch" or func.__name__ == "test_epoch":
                    epoch_metrics["accuracy"] = 0.0
                return epoch_metrics  # type: ignore
        return func(self, *args, **kwargs)

    return wrapper


@RunnerRegistry.register("epoch_based_runner")
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner trains the model for a fixed number of epochs.

    Attributes:
        max_epochs (int): Maximum number of epochs to train for
        current_epoch (int): Current epoch
        hooks (List[PrioritizedHook]): List of hooks to call at different stages
        iter (int): Current iteration
        train_metrics (Dict[str, float]): Training metrics
        val_metrics (Dict[str, float]): Validation metrics
        test_metrics (Dict[str, float]): Test metrics
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        """Initialize the runner.

        Args:
            config: Configuration dictionary for the experiment
            device: Device to run the experiment on, defaults to cuda if available
        """
        super().__init__(config, device)

        runner_config = config.get("runner", {})
        self.max_epochs = runner_config.get("max_epochs", 100)
        self.current_epoch = 0
        self.iter = 0

        self.hooks: List[PrioritizedHook] = []
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        self.best_val_metric: float = float("inf")

    def register_hook(self, hook: PrioritizedHook) -> None:
        """Register a hook function.

        Args:
            hook: Hook function to register
        """
        self.hooks.append(hook)
        self.hooks.sort(key=lambda x: x.priority)

    def call_hooks(self, stage: str) -> None:
        """Call all registered hooks for a specific stage.

        Args:
            stage: Stage to call hooks for (e.g., 'before_train_epoch', 'after_train_epoch')
        """
        for hook in self.hooks:
            if hasattr(hook, stage):
                getattr(hook, stage)(self)

    @ensure_model_initialized
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of loss values
        """
        # TODO:temporary indexing, more general indexing should be used
        inputs = batch["images"].to(self.device)
        targets = batch["molecular_subtype"].to(self.device)

        model = cast(torch.nn.Module, self.model)
        loss_fn = cast(torch.nn.Module, self.loss_fn)
        optimizer = cast(torch.optim.Optimizer, self.optimizer)

        if loss_fn is None or optimizer is None:
            self.logger.error("Cannot perform train step: loss_fn or optimizer is None")
            return {"loss": torch.tensor(float("inf"), device=self.device)}

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss}

    @ensure_model_initialized
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        # TODO:temporary indexing, more general indexing should be used
        inputs = batch["images"].to(self.device)
        targets = batch["molecular_subtype"].to(self.device)

        model = cast(torch.nn.Module, self.model)
        loss_fn = cast(torch.nn.Module, self.loss_fn)

        if loss_fn is None:
            self.logger.error("Cannot perform validation step: loss_fn is None")
            return {"loss": torch.tensor(float("inf"), device=self.device), "accuracy": torch.tensor(0.0, device=self.device)}

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == targets).sum().item()
            total = targets.size(0)
            accuracy = correct / total

        return {"loss": loss, "accuracy": torch.tensor(accuracy, device=self.device)}

    def test_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform a single test step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        return self.val_step(batch)

    @ensure_model_initialized
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        model = cast(torch.nn.Module, self.model)
        model.train()

        epoch_metrics: Dict[str, float] = {}
        self.call_hooks("before_train_epoch")

        if self.train_dataloader is None:
            self.logger.error("Cannot train epoch: train_dataloader is None")
            return {"loss": float("inf")}

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}/{self.max_epochs}")
        for batch_idx, batch in enumerate(pbar):
            self.call_hooks("before_train_step")
            step_metrics = self.train_step(batch)

            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v.item()

            pbar.set_postfix({k: f"{v / (batch_idx + 1):.4f}" for k, v in epoch_metrics.items()})
            self.iter += 1
            self.call_hooks("after_train_step")

        dataloader_len = len(self.train_dataloader)
        for k in epoch_metrics:
            epoch_metrics[k] /= float(dataloader_len)

        self.train_metrics = epoch_metrics
        self.call_hooks("after_train_epoch")

        return epoch_metrics

    @ensure_model_initialized
    def val_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        model = cast(torch.nn.Module, self.model)
        model.eval()

        epoch_metrics: Dict[str, float] = {}
        self.call_hooks("before_val_epoch")

        if self.val_dataloader is None:
            self.logger.error("Cannot validate epoch: val_dataloader is None")
            return {"loss": float("inf"), "accuracy": 0.0}

        pbar = tqdm(self.val_dataloader, desc="Validation")
        for batch_idx, batch in enumerate(pbar):
            self.call_hooks("before_val_step")
            step_metrics = self.val_step(batch)

            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v.item()

            pbar.set_postfix({k: f"{v / (batch_idx + 1):.4f}" for k, v in epoch_metrics.items()})
            self.call_hooks("after_val_step")

        dataloader_len = len(self.val_dataloader)
        for k in epoch_metrics:
            epoch_metrics[k] /= float(dataloader_len)

        self.val_metrics = epoch_metrics
        self.call_hooks("after_val_epoch")

        return epoch_metrics

    @ensure_model_initialized
    def test_epoch(self) -> Dict[str, float]:
        """Test for one epoch.

        Returns:
            Dictionary of test metrics
        """
        model = cast(torch.nn.Module, self.model)
        model.eval()

        epoch_metrics: Dict[str, float] = {}
        self.call_hooks("before_test_epoch")

        if self.test_dataloader is None:
            self.logger.error("Cannot test epoch: test_dataloader is None")
            return {"loss": float("inf"), "accuracy": 0.0}

        pbar = tqdm(self.test_dataloader, desc="Testing")
        for batch_idx, batch in enumerate(pbar):
            self.call_hooks("before_test_step")
            step_metrics = self.test_step(batch)

            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v.item()

            pbar.set_postfix({k: f"{v / (batch_idx + 1):.4f}" for k, v in epoch_metrics.items()})
            self.call_hooks("after_test_step")

        dataloader_len = len(self.test_dataloader)
        for k in epoch_metrics:
            epoch_metrics[k] /= float(dataloader_len)

        self.test_metrics = epoch_metrics
        self.call_hooks("after_test_epoch")

        return epoch_metrics

    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        start_time = time.time()

        self.call_hooks("before_train")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch}/{self.max_epochs}, Train metrics: {train_metrics}")

            if self.val_dataloader is not None:
                val_metrics = self.val_epoch()
                self.logger.info(f"Epoch {epoch}/{self.max_epochs}, Val metrics: {val_metrics}")

                if self.config.get("save_best", True):
                    if val_metrics["loss"] < self.best_val_metric:
                        self.best_val_metric = val_metrics["loss"]
                        self.save_checkpoint("best_model", {"epoch": epoch, "metrics": val_metrics})

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.config.get("checkpoint_interval", 10) == 0:
                self.save_checkpoint(f"epoch_{epoch}", {"epoch": epoch, "metrics": train_metrics})

        self.call_hooks("after_train")

        end_time = time.time()
        self.logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            self.logger.warning("No validation dataloader available")
            return {}

        self.logger.info("Starting validation")
        val_metrics = self.val_epoch()

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test the model.

        Returns:
            Dictionary of test metrics
        """
        if self.test_dataloader is None:
            self.logger.warning("No test dataloader available")
            return {}

        self.logger.info("Starting testing")
        test_metrics = self.test_epoch()

        return test_metrics
