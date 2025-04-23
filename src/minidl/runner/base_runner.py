import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from minidl.dataset import DatasetBuilder
from minidl.losses import build_loss
from minidl.metrics import MetricsBuilder, MetricsCalculator
from minidl.model.model_registry import ModelBuilder

T = TypeVar("T")


class PrioritizedHook(Protocol):
    priority: int


def ensure_model_exists(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure model exists before calling the function."""

    def wrapper(self: "BaseRunner", *args: Any, **kwargs: Any) -> T:
        if self.model is None:
            self.logger.error(f"Cannot execute {func.__name__}: model is None")
            if func.__name__ == "save_checkpoint":
                return None  # type: ignore
            elif func.__name__ == "load_checkpoint":
                return {}  # type: ignore
        return func(self, *args, **kwargs)

    return wrapper


class BaseRunner(ABC):
    """Base class for all runners.

    This class provides a framework for running experiments with high extensibility.
    It handles dataset creation, model initialization, training, validation, and testing.

    Attributes:
        config (dict[str, Any]): Configuration dictionary for the experiment
        device (torch.device): Device to run the experiment on
        logger (logging.Logger): Logger for the experiment
        model (nn.Module): Model for the experiment
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        test_dataset (Dataset): Test dataset
        train_dataloader (DataLoader): Training dataloader
        val_dataloader (DataLoader): Validation dataloader
        test_dataloader (DataLoader): Test dataloader
        optimizer (Optimizer): Optimizer for training
        scheduler (_LRScheduler): Learning rate scheduler
        loss_fn (nn.Module): Loss function
    """

    def __init__(self, config: dict[str, Any], device: torch.device | None = None):
        """Initialize the runner.

        Args:
            config: Configuration dictionary for the experiment
            device: Device to run the experiment on, defaults to cuda if available
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(self.__class__.__name__)

        self.hooks: list[PrioritizedHook] = []

        self.model: Module | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: _LRScheduler | None = None
        self.loss_fn: Module | None = None

        self.work_dir = self.config.get("work_dir", "work_dirs/default")
        os.makedirs(self.work_dir, exist_ok=True)

    def build_metrics(self) -> None:
        """Build metrics from config."""
        metrics_config = self.config.get("training", {})
        metrics_dict = MetricsBuilder.build_metrics(metrics_config.get("metrics", []))
        self.metrics_calculator = MetricsCalculator(metrics_dict)

    def build_loss(self) -> None:
        """Build loss function from config."""
        self.loss_fn = build_loss(self.config)

    def build_dataset(self) -> None:
        """Build datasets for training, validation, and testing."""
        dataset_config = self.config.get("dataset", {})
        train_indices = dataset_config.get("indices", {}).get("train", [])

        if "train" in dataset_config.get("indices", {}):
            self.train_dataset = DatasetBuilder.build_dataset(self.config, "train", train_indices)
        if "val" in dataset_config.get("indices", {}):
            self.val_dataset = DatasetBuilder.build_dataset(self.config, "val", train_indices)
        if "test" in dataset_config.get("indices", {}):
            self.test_dataset = DatasetBuilder.build_dataset(self.config, "test", train_indices)

    def build_dataloader(self) -> None:
        """Build dataloaders for training, validation, and testing."""
        dataloader_config = self.config.get("dataloader", {})

        train_loader_cfg = {"batch_size": 16, "shuffle": True, "num_workers": 4, "pin_memory": True, **dataloader_config.get("train", {})}
        val_loader_cfg = {"batch_size": 16, "shuffle": False, "num_workers": 4, "pin_memory": True, **dataloader_config.get("val", {})}
        test_loader_cfg = {"batch_size": 16, "shuffle": False, "num_workers": 4, "pin_memory": True, **dataloader_config.get("test", {})}

        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(self.train_dataset, **train_loader_cfg)

        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(self.val_dataset, **val_loader_cfg)

        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(self.test_dataset, **test_loader_cfg)

    def build_model(self) -> None:
        """Build model for the experiment."""
        self.model = ModelBuilder.build_model(self.config)
        if self.model is not None:
            self.model.to(self.device)

            model_name = self.config.get("model", {}).get("name")
            self.logger.info(f"Model: {model_name}")
            self.logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def build_optimizer(self) -> None:
        """Build optimizer for training."""
        if self.model is None:
            self.logger.error("Cannot build optimizer: model is None")
            return

        training_config = self.config.get("training", {})
        optim_config = training_config.get("optimizer", {})
        optim_name = optim_config.get("name", "Adam")

        optim_cls = getattr(torch.optim, optim_name)
        self.optimizer = optim_cls(self.model.parameters(), **optim_config.get("params", {}))

        scheduler_config = training_config.get("scheduler", {})
        if scheduler_config and self.optimizer is not None:
            scheduler_name = scheduler_config.get("name")
            if scheduler_name:
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
                self.scheduler = scheduler_cls(self.optimizer, **scheduler_config.get("params", {}))

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

    @ensure_model_exists
    def save_checkpoint(self, filename: str, extra_info: dict[str, Any] | None = None) -> None:
        """Save model checkpoint.

        Args:
            filename: Filename to save the checkpoint
            extra_info: Extra information to save in the checkpoint
        """
        if not filename.endswith(".pth"):
            filename = f"{filename}.pth"

        checkpoint_path = os.path.join(self.work_dir, filename)

        # Safe to use model since decorator ensures it's not None
        model = cast(Module, self.model)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
        }

        if extra_info:
            checkpoint.update(extra_info)

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    @ensure_model_exists
    def load_checkpoint(self, filename: str) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            filename: Filename to load the checkpoint from

        Returns:
            Checkpoint dictionary
        """
        if not filename.endswith(".pth"):
            filename = f"{filename}.pth"

        checkpoint_path = os.path.join(self.work_dir, filename)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Safe to use model since decorator ensures it's not None
        model = cast(Module, self.model)

        # Load model weights
        model.load_state_dict(checkpoint["model"])

        # Load optimizer state if available
        if self.optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state if available
        if self.scheduler and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def validate(self) -> dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        pass

    @abstractmethod
    def test(self) -> dict[str, float]:
        """Test the model.

        Returns:
            Dictionary of test metrics
        """
        pass

    def run(self) -> None:
        """Run the training, validation, and testing process."""
        # Build components
        self.build_model()
        self.build_dataset()
        self.build_dataloader()
        self.build_loss()
        self.build_metrics()
        self.build_optimizer()

        # Run training
        self.train()

        # Run validation
        val_metrics = self.validate()
        self.logger.info(f"Validation metrics: {val_metrics}")

        # Run testing
        test_metrics = self.test()
        self.logger.info(f"Test metrics: {test_metrics}")
