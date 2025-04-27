import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch.amp import GradScaler
from tqdm import tqdm

from minidl.transforms import build_transform_pipeline

from .base_runner import BaseRunner
from .runner_registry import RunnerRegistry

T = TypeVar("T")


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
                epoch_metrics: dict[str, float] = {"loss": float("inf")}
                if func.__name__ == "val_epoch" or func.__name__ == "test_epoch":
                    epoch_metrics["accuracy"] = 0.0
                return epoch_metrics  # type: ignore
        return func(self, *args, **kwargs)

    return wrapper


def compute_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """Compute the norm of gradients for all parameters.

    Args:
        model: The model containing parameters to compute gradient norms for
        norm_type: Type of the norm (e.g., 2 for L2 norm)

    Returns:
        Norm of gradients
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)

    # Calculate norm based on norm_type
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

    return total_norm


@RunnerRegistry.register("epoch_based_runner")
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner trains the model for a fixed number of epochs.

    Attributes:
        max_epochs (int): Maximum number of epochs to train for
        current_epoch (int): Current epoch
        hooks (list[PrioritizedHook]): list of hooks to call at different stages
        iter (int): Current iteration
        train_metrics (dict[str, float]): Training metrics
        val_metrics (dict[str, float]): Validation metrics
        test_metrics (dict[str, float]): Test metrics
    """

    def __init__(self, config: dict[str, Any], device: torch.device | None = None):
        """Initialize the runner.

        Args:
            config: Configuration dictionary for the experiment
            device: Device to run the experiment on, defaults to cuda if available
        """
        super().__init__(config, device)

        runner_config = config.get("runner", {})
        self.max_epochs = runner_config.get("max_epochs", 100)

        training_config = config.get("training", {})

        grad_clip_config = training_config.get("grad_clip", {})
        self.grad_clip_enabled = grad_clip_config.get("enabled", False)
        self.grad_clip_type = grad_clip_config.get("type", "norm")
        self.grad_clip_value = grad_clip_config.get("value", 0.0)

        self.current_epoch = 0
        self.iter = 0

        self.train_metrics: dict[str, float] = {}
        self.val_metrics: dict[str, float] = {}
        self.test_metrics: dict[str, float] = {}
        self.best_val_metric: float = float("inf")

        # Initialize gradient norm tracking
        self.grad_norm = torch.tensor(0.0)
        self.grad_norm_history: list[float] = []
        self.train_step_metrics: dict[str, torch.Tensor] = {}

    @ensure_model_initialized
    def train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of loss values and metrics
        """
        # TODO:temporary indexing, more general indexing should be used
        inputs = batch["images"].to(self.device)
        targets = batch["clinical_label"].to(self.device)

        loss_fn = self.loss_fn
        optimizer = self.optimizer

        if loss_fn is None or optimizer is None:
            self.logger.error("Cannot perform train step: loss_fn or optimizer is None")
            return {"loss": torch.tensor(float("inf"), device=self.device)}

        scaler = GradScaler(self.device)
        optimizer.zero_grad()
        with torch.amp.autocast(self.device.type):
            outputs = self.model(inputs)  # type: ignore
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()

        self.grad_norm = compute_gradient_norm(self.model, norm_type=2.0)  # type: ignore
        self.grad_norm_history.append(float(self.grad_norm))

        if self.grad_clip_enabled and self.grad_clip_value > 0:
            scaler.unscale_(optimizer)

            if self.grad_clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            elif self.grad_clip_type == "value":
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)

        scaler.step(optimizer)
        scaler.update()

        # Store step metrics for hooks
        self.train_step_metrics = {"loss": loss, "grad_norm": self.grad_norm}

        return {"loss": loss, "grad_norm": self.grad_norm}

    @ensure_model_initialized
    def val_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a single validation step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        # TODO:temporary indexing, more general indexing should be used
        inputs = batch["images"].to(self.device)
        targets = batch["clinical_label"].to(self.device)

        model = self.model
        loss_fn = self.loss_fn

        if loss_fn is None:
            self.logger.error("Cannot perform validation step: loss_fn is None")
            return {"loss": torch.tensor(float("inf"), device=self.device), "accuracy": torch.tensor(0.0, device=self.device)}

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            metrics = {}
            if hasattr(self, "metrics_calculator"):
                metrics = self.metrics_calculator.compute_metrics(outputs, targets)
                metrics = {k: torch.tensor(v, device=self.device) for k, v in metrics.items()}

        return {"loss": loss, **metrics}

    def test_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a single test step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        return self.val_step(batch)

    @ensure_model_initialized
    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        model = self.model
        model.train()

        epoch_metrics: dict[str, float] = {}
        self.call_hooks("before_train_epoch")

        if self.train_dataloader is None:
            self.logger.error("Cannot train epoch: train_dataloader is None")
            return {"loss": float("inf")}

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")
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
    def val_epoch(self) -> dict[str, float]:
        """Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        model = self.model
        model.eval()

        epoch_metrics: dict[str, float] = {}
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
    def test_epoch(self) -> dict[str, float]:
        """Test for one epoch.

        Returns:
            Dictionary of test metrics
        """
        model = self.model
        model.eval()

        epoch_metrics: dict[str, float] = {}
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

        if self.grad_clip_enabled and self.grad_clip_value > 0:
            self.logger.info(f"Gradient clipping enabled: type={self.grad_clip_type}, value={self.grad_clip_value}")
        else:
            self.logger.info("Gradient clipping disabled")

        start_time = time.time()

        self.call_hooks("before_train")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch}/{self.max_epochs}, Train metrics: {train_metrics}")

            if self.val_dataloader is not None:
                val_metrics = self.val_epoch()
                self.logger.info(f"Epoch {epoch}/{self.max_epochs}, Val metrics: {val_metrics}")

            if self.scheduler is not None:
                self.scheduler.step()

        self.call_hooks("after_train")

        end_time = time.time()
        self.logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    def validate(self) -> dict[str, float]:
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

    def test(self) -> dict[str, float]:
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


@RunnerRegistry.register("epoch_based_multimodal_runner")
class MultimodalRunner(EpochBasedRunner):
    """Runner designed to handle multimodal data while following the structure of EpochBasedTrCudaRunner."""

    def __init__(self, config: dict[str, Any], device: torch.device | None = None):
        """Initialize the runner with multimodal support.

        Args:
            config: Configuration dictionary for the experiment
            device: Device to run the experiment on
        """
        super().__init__(config, device)

        mm_config = config.get("dataset", {})

        self.required_phases = mm_config["params"].get("required_phases", [])
        self.required_phases_len = len(self.required_phases) if self.required_phases else 3
        self.target_key = mm_config.get("target_key", "clinical_label")
        self.logger.info(f"Training with phases: {self.required_phases}")

        runner_config = config.get("runner", {})
        transforms_config = runner_config.get("transforms", {})
        for split in ["train", "val", "test"]:
            transform_configs = transforms_config.get(split, {})
            if transform_configs:
                transform = build_transform_pipeline(transform_configs)
                self.__setattr__(f"{split}_transform", transform)

    def apply_batch_transform(self, batch_data, transform_pipeline):
        """
        Apply transforms to each item in a batch while keeping data on the device
        and preserving all other keys in the batch.
        """
        if "images" in batch_data:
            batch_size = batch_data["images"].shape[0]

            transformed_images = []
            for i in range(batch_size):
                sample = {"images": [batch_data["images"][i][j] for j in range(self.required_phases_len)]}

                transformed = transform_pipeline(sample)
                transformed_images.append(transformed["images"].squeeze(0))

            batch_data["images"] = torch.stack(transformed_images, dim=0)

        return batch_data

    @ensure_model_initialized
    def train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a single training step with multimodal inputs."""
        # Apply transforms to the batch
        if hasattr(self, "train_transform"):
            batch = self.apply_batch_transform(batch, self.train_transform)

        # Move all tensors to the device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        targets = batch.get(self.target_key)
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        if loss_fn is None or optimizer is None:
            self.logger.error("Cannot perform train step: loss_fn or optimizer is None")
            return {"loss": torch.tensor(float("inf"), device=self.device)}

        scaler = GradScaler(self.device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=self.device.type):
            outputs = self.model(batch)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()

        self.grad_norm = compute_gradient_norm(self.model, norm_type=2.0)
        self.grad_norm_history.append(float(self.grad_norm))

        if self.grad_clip_enabled and self.grad_clip_value > 0:
            scaler.unscale_(optimizer)

            if self.grad_clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            elif self.grad_clip_type == "value":
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)

        scaler.step(optimizer)
        scaler.update()

        self.train_step_metrics = {"loss": loss, "grad_norm": self.grad_norm}

        return {"loss": loss, "grad_norm": self.grad_norm}

    @ensure_model_initialized
    def val_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Perform a single validation step with multimodal inputs."""
        if hasattr(self, "val_transform"):
            batch = self.apply_batch_transform(batch, self.val_transform)

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        targets = batch.get(self.target_key)
        model = self.model
        loss_fn = self.loss_fn

        if loss_fn is None:
            self.logger.error("Cannot perform validation step: loss_fn is None")
            return {"loss": torch.tensor(float("inf"), device=self.device)}

        with torch.no_grad():
            # Pass the entire batch to the model
            outputs = model(batch)
            loss = loss_fn(outputs, targets)

            metrics = {}
            if hasattr(self, "metrics_calculator"):
                metrics = self.metrics_calculator.compute_metrics(outputs, targets)
                metrics = {k: torch.tensor(v, device=self.device) for k, v in metrics.items()}

        return {"loss": loss, **metrics}
