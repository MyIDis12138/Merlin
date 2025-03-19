import os
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter


class HookRegistry:
    """Registry for hook classes.

    This class provides a registry for hook classes, allowing them to be registered
    and retrieved by name.
    """

    _hooks: dict[str, type["Hook"]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a hook class.

        Args:
            name: Name of the hook

        Returns:
            Decorator function for registration
        """

        def decorator(hook_cls: type["Hook"]) -> type["Hook"]:
            cls._hooks[name] = hook_cls
            return hook_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type["Hook"]:
        """Get a hook class by name.

        Args:
            name: Name of the hook

        Returns:
            Hook class

        Raises:
            ValueError: If hook name is not registered
        """
        if name not in cls._hooks:
            raise ValueError(f"Hook '{name}' not found. Available hooks: {list(cls._hooks.keys())}")

        return cls._hooks[name]

    @classmethod
    def available_hooks(cls) -> dict[str, type["Hook"]]:
        """Get all available hooks.

        Returns:
            Dictionary of hook names and classes
        """
        return cls._hooks.copy()


class Hook:
    """Base class for hooks.

    Hooks are used to extend the functionality of runners without modifying their code.
    They can be registered to runners and will be called at specific stages of the training process.

    To create a custom hook, inherit from this class and implement the methods you need.
    """

    def __init__(self, priority: int = 50):
        """Initialize the hook.

        Args:
            priority: Priority of the hook, lower values mean higher priority
        """
        self.priority = priority

    def before_run(self, runner):
        """Called before the runner starts running."""
        pass

    def after_run(self, runner):
        """Called after the runner finishes running."""
        pass

    def before_train(self, runner):
        """Called before training starts."""
        pass

    def after_train(self, runner):
        """Called after training ends."""
        pass

    def before_train_epoch(self, runner):
        """Called before each training epoch."""
        pass

    def after_train_epoch(self, runner):
        """Called after each training epoch."""
        pass

    def before_train_step(self, runner):
        """Called before each training step."""
        pass

    def after_train_step(self, runner):
        """Called after each training step."""
        pass

    def before_val_epoch(self, runner):
        """Called before each validation epoch."""
        pass

    def after_val_epoch(self, runner):
        """Called after each validation epoch."""
        pass

    def before_val_step(self, runner):
        """Called before each validation step."""
        pass

    def after_val_step(self, runner):
        """Called after each validation step."""
        pass

    def before_test_epoch(self, runner):
        """Called before each test epoch."""
        pass

    def after_test_epoch(self, runner):
        """Called after each test epoch."""
        pass

    def before_test_step(self, runner):
        """Called before each test step."""
        pass

    def after_test_step(self, runner):
        """Called after each test step."""
        pass


@HookRegistry.register("tensorboard_logger_hook")
class TensorboardLoggerHook(Hook):
    """Hook for logging to TensorBoard."""

    def __init__(self, log_dir: str | None = None, interval: int = 10, priority: int = 50):
        """Initialize the hook.

        Args:
            log_dir: Directory to save TensorBoard logs, defaults to work_dir/tensorboard
            interval: Interval (in iterations) to log training metrics
            priority: Priority of the hook
        """
        super().__init__(priority)
        self.log_dir = log_dir
        self.interval = interval
        self.writer = None

    def before_train(self, runner):
        """Initialize TensorBoard writer."""
        if self.log_dir is None:
            self.log_dir = os.path.join(runner.work_dir, "tensorboard")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def after_train_step(self, runner):
        """Log training metrics to TensorBoard."""
        # Only log every interval iterations
        if runner.iter % self.interval != 0:
            return

        # Log learning rate
        if runner.optimizer is not None:
            for i, param_group in enumerate(runner.optimizer.param_groups):
                self.writer.add_scalar(f"train/lr_{i}", param_group["lr"], runner.iter)  # type: ignore

    def after_train_epoch(self, runner):
        """Log training metrics to TensorBoard."""
        # Log training metrics
        for k, v in runner.train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, runner.current_epoch)  # type: ignore

    def after_val_epoch(self, runner):
        """Log validation metrics to TensorBoard."""
        # Log validation metrics
        for k, v in runner.val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, runner.current_epoch)  # type: ignore

    def after_test_epoch(self, runner):
        """Log test metrics to TensorBoard."""
        # Log test metrics
        for k, v in runner.test_metrics.items():
            self.writer.add_scalar(f"test/{k}", v, runner.current_epoch)  # type: ignore

    def after_train(self, runner):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


@HookRegistry.register("checkpoint_hook")
class CheckpointHook(Hook):
    """Hook for saving checkpoints."""

    def __init__(
        self,
        interval: int = 1,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        out_dir: str | None = None,
        max_keep_ckpts: int = 5,
        save_best: bool = True,
        rule: str = "less",
        best_metric_name: str = "loss",
        priority: int = 50,
    ):
        """Initialize the hook.

        Args:
            interval: Interval (in epochs) to save checkpoints
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            out_dir: Directory to save checkpoints, defaults to runner.work_dir
            max_keep_ckpts: Maximum number of checkpoints to keep
            save_best: Whether to save the best checkpoint
            rule: Rule to determine the best checkpoint, 'less' or 'greater'
            best_metric_name: Name of the metric to determine the best checkpoint
            priority: Priority of the hook
        """
        super().__init__(priority)
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_best = save_best
        self.rule = rule
        self.best_metric_name = best_metric_name

        self.best_metric = float("inf") if rule == "less" else -float("inf")
        self.saved_ckpts: list[str] = []

    def before_train(self, runner):
        """Initialize checkpoint directory."""
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def after_train_epoch(self, runner):
        """Save checkpoint after training epoch."""
        # Save checkpoint every interval epochs
        if (runner.current_epoch + 1) % self.interval != 0:
            return

        # Save checkpoint
        ckpt_path = os.path.join(self.out_dir, f"epoch_{runner.current_epoch + 1}.pth")  # type: ignore
        self._save_checkpoint(runner, ckpt_path)
        self.saved_ckpts.append(ckpt_path)

        # Remove old checkpoints if max_keep_ckpts is reached
        if self.max_keep_ckpts > 0 and len(self.saved_ckpts) > self.max_keep_ckpts:
            ckpt_to_remove = self.saved_ckpts.pop(0)
            if os.path.exists(ckpt_to_remove):
                os.remove(ckpt_to_remove)

    def after_val_epoch(self, runner):
        """Save best checkpoint after validation epoch."""
        if not self.save_best:
            return

        # Get current metric
        if self.best_metric_name not in runner.val_metrics:
            return

        current_metric = runner.val_metrics[self.best_metric_name]

        # Check if current metric is better than best metric
        is_better = False
        if self.rule == "less":
            is_better = current_metric < self.best_metric
        else:
            is_better = current_metric > self.best_metric

        if is_better:
            self.best_metric = current_metric
            ckpt_path = os.path.join(self.out_dir, "best_model.pth")  # type: ignore
            self._save_checkpoint(runner, ckpt_path)
            runner.logger.info(f"Saved best model with {self.best_metric_name} = {self.best_metric:.4f}")

    def _save_checkpoint(self, runner, ckpt_path):
        """Save checkpoint to file."""
        checkpoint = {
            "epoch": runner.current_epoch + 1,
            "model": runner.model.state_dict(),
            "config": runner.config,
        }

        if self.save_optimizer and runner.optimizer is not None:
            checkpoint["optimizer"] = runner.optimizer.state_dict()

        if self.save_scheduler and runner.scheduler is not None:
            checkpoint["scheduler"] = runner.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)
        runner.logger.info(f"Checkpoint saved to {ckpt_path}")


@HookRegistry.register("early_stopping_hook")
class EarlyStoppingHook(Hook):
    """Hook for early stopping."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, rule: str = "less", metric_name: str = "loss", priority: int = 100):
        """Initialize the hook.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            rule: Rule to determine improvement, 'less' or 'greater'
            metric_name: Name of the metric to monitor
            priority: Priority of the hook
        """
        super().__init__(priority)
        self.patience = patience
        self.min_delta = min_delta
        self.rule = rule
        self.metric_name = metric_name

        self.best_metric = float("inf") if rule == "less" else -float("inf")
        self.counter = 0
        self.early_stop = False

    def after_val_epoch(self, runner):
        """Check for early stopping after validation epoch."""
        # Get current metric
        if self.metric_name not in runner.val_metrics:
            return

        current_metric = runner.val_metrics[self.metric_name]

        # Check if current metric is better than best metric
        is_better = False
        if self.rule == "less":
            is_better = current_metric < self.best_metric - self.min_delta
        else:
            is_better = current_metric > self.best_metric + self.min_delta

        if is_better:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            runner.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                runner.logger.info(f"Early stopping triggered after {runner.current_epoch + 1} epochs")
                # Set max_epochs to current epoch to stop training
                runner.max_epochs = runner.current_epoch + 1


class HookBuilder:
    """Builder for creating hook instances.

    This class provides a builder for creating hook instances based on configuration.
    """

    @staticmethod
    def build_hook(hook_config: dict[str, Any], work_dir: str | None = None) -> Hook:
        """Build a hook instance based on configuration.

        Args:
            hook_config: Configuration dictionary containing hook name and parameters
            work_dir: Working directory for the runner

        Returns:
            Hook instance

        Raises:
            ValueError: If hook name is not specified or not registered
        """
        hook_name = hook_config.get("name")

        if hook_name is None:
            raise ValueError("Hook name must be specified in config")

        # Get hook class
        hook_cls = HookRegistry.get(hook_name)

        # Extract hook parameters
        hook_params = hook_config.get("params", {}).copy()

        # Handle paths relative to work_dir
        if work_dir:
            if hook_name == "tensorboard_logger_hook" and "log_dir" in hook_params:
                hook_params["log_dir"] = os.path.join(work_dir, hook_params["log_dir"])
            elif hook_name == "checkpoint_hook" and "out_dir" in hook_params:
                hook_params["out_dir"] = os.path.join(work_dir, hook_params["out_dir"])

        # Create hook instance
        return hook_cls(**hook_params)

    @staticmethod
    def build_hooks(hooks_config: list[dict[str, Any]] | None = None, work_dir: str | None = None) -> list[Hook]:
        """Build multiple hook instances based on configuration.

        Args:
            hooks_config: list of hook configurations
            work_dir: Working directory for the runner

        Returns:
            list of hook instances

        Raises:
            ValueError: If any hook name is not specified or not registered
        """
        if not hooks_config:
            return []

        hooks = []
        for hook_config in hooks_config:
            hook = HookBuilder.build_hook(hook_config, work_dir)
            hooks.append(hook)

        return hooks
