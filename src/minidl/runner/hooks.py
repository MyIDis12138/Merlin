import glob
import os
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import wandb


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


@HookRegistry.register("checkpoint_saver_hook")
class CheckpointSaverHook(Hook):
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
            self.out_dir = os.path.join(runner.work_dir, "checkpoints")
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


@HookRegistry.register("checkpoint_loader_hook")
class CheckpointLoaderHook(Hook):
    """Hook for loading model checkpoints from a file or directory.

    This hook loads a model checkpoint before training. If a directory is provided,
    it will find the latest checkpoint file in that directory. If a file is provided,
    it will load that specific checkpoint file.

    Attributes:
        checkpoint_path (str): Path to a checkpoint file or directory containing checkpoint files
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys
                      returned by this module's state_dict()
        map_location (str, optional): Device to map tensors to when loading the checkpoint
        load_optimizer (bool): Whether to load optimizer state
        load_scheduler (bool): Whether to load scheduler state
        priority (int): Priority of the hook
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        enable=True,
        strict: bool = True,
        map_location: str | None = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        priority: int = 0,
    ):
        """Initialize the hook.

        Args:
            checkpoint_path: Path to checkpoint file or directory containing checkpoint files
            enable: whether enable checkpoint loading.
            strict: Whether to strictly enforce that the keys in state_dict match the keys
                   returned by this module's state_dict()
            map_location: Device to map tensors to when loading the checkpoint
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            priority: Priority of the hook
        """
        super().__init__(priority)
        self.checkpoint_path = checkpoint_path
        self.strict = strict
        self.map_location = map_location
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.enable = enable

    def _find_latest_checkpoint(self, dir_path: str) -> str | None:
        """Find the latest checkpoint file in the directory.

        Args:
            dir_path: Directory path to search for checkpoint files

        Returns:
            Path to the latest checkpoint file, or None if no checkpoint files found
        """
        checkpoint_files = []

        # Look for epoch_*.pth files
        epoch_files = glob.glob(os.path.join(dir_path, "epoch_*.pth"))
        checkpoint_files.extend(epoch_files)

        # Also look for best_model.pth files
        best_model_path = os.path.join(dir_path, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint_files.append(best_model_path)

        # Look for other .pth files as a fallback
        if not checkpoint_files:
            checkpoint_files = glob.glob(os.path.join(dir_path, "*.pth"))

        if not checkpoint_files:
            return None

        # Sort files by modification time (latest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        return checkpoint_files[0]

    def before_train(self, runner):
        """Load checkpoint before the runner starts.

        Args:
            runner: Runner instance
        """
        if not self.enable:
            runner.logger.info("Skipping checkpoint loading")
            return

        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
        else:
            checkpoint_path = runner.work_dir
            runner.logger.warning("No checkpoint path provided, checking work dir")

        # Determine the actual checkpoint path
        if os.path.isdir(checkpoint_path):
            checkpoint_path = self._find_latest_checkpoint(os.path.join(checkpoint_path, "checkpoints"))
            if not checkpoint_path:
                runner.logger.warning(f"No checkpoint files found in {checkpoint_path}")
                return
            runner.logger.info(f"Using latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_path
            if not os.path.isfile(checkpoint_path):
                runner.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                return

        # Load the checkpoint
        try:
            map_location = self.map_location
            if map_location is None and hasattr(runner, "device"):
                map_location = runner.device

            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # Load model state
            if "model" in checkpoint:
                if runner.model is not None:
                    runner.model.load_state_dict(checkpoint["model"], strict=self.strict)
                    runner.logger.info(f"Loaded model state from {checkpoint_path}")
                else:
                    runner.logger.warning("Cannot load model state: model is None")
            else:
                runner.logger.warning(f"Checkpoint does not contain model state: {checkpoint_path}")

            # Load optimizer state if requested
            if self.load_optimizer and "optimizer" in checkpoint and runner.optimizer is not None:
                runner.optimizer.load_state_dict(checkpoint["optimizer"])
                runner.logger.info(f"Loaded optimizer state from {checkpoint_path}")

            # Load scheduler state if requested
            if self.load_scheduler and "scheduler" in checkpoint and runner.scheduler is not None:
                runner.scheduler.load_state_dict(checkpoint["scheduler"])
                runner.logger.info(f"Loaded scheduler state from {checkpoint_path}")

            # Load epoch/iteration info if available
            if "epoch" in checkpoint and hasattr(runner, "current_epoch"):
                runner.current_epoch = checkpoint["epoch"]
                runner.logger.info(f"Resuming from epoch {runner.current_epoch}")

            if "iter" in checkpoint and hasattr(runner, "iter"):
                runner.iter = checkpoint["iter"]
                runner.logger.info(f"Resuming from iteration {runner.iter}")

            # Load additional metadata
            for key, value in checkpoint.items():
                if key not in ["model", "optimizer", "scheduler", "epoch", "iter", "config"]:
                    setattr(runner, f"loaded_{key}", value)

            runner.logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")

        except Exception as e:
            runner.logger.error(f"Failed to load checkpoint: {e}")
            raise


@HookRegistry.register("wandb_logger_hook")
class WandbLoggerHook(Hook):
    """Hook for logging to Weights & Biases (wandb)."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        dir: str | None = None,
        tags: list | None = None,
        notes: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        save_code: bool = True,
        log_artifacts: bool = False,
        interval: int = 10,
        priority: int = 50,
        log_grad_norm: bool = True,
        grad_norm_window: int = 100,
    ):
        """Initialize the wandb logging hook.

        Args:
            project: Name of the wandb project
            name: Name of the run (defaults to a randomly generated name if not provided)
            entity: wandb team/username (defaults to your username if not provided)
            config: Dictionary of configuration parameters to track
            dir: Directory to store the run data
            tags: List of tags for organizing and filtering runs
            notes: Notes about the run to be stored with it
            group: Specify a group to organize multiple runs
            job_type: Specify the type of job (e.g., 'train', 'eval', etc.)
            save_code: Save a copy of the code that created the run
            log_artifacts: Whether to log model checkpoints as artifacts
            interval: Interval (in iterations) to log training metrics
            priority: Priority of the hook
            log_grad_norm: Whether to log gradient norms
            grad_norm_window: Number of recent gradient norms to consider for tracking
        """
        super().__init__(priority)
        self.project = project
        self.name = name
        self.entity = entity
        self.config = config
        self.dir = dir
        self.tags = tags
        self.notes = notes
        self.group = group
        self.job_type = job_type
        self.save_code = save_code
        self.log_artifacts = log_artifacts
        self.interval = interval
        self.run = None
        self.log_grad_norm = log_grad_norm
        self.grad_norm_window = grad_norm_window
        self.recent_grad_norms = []  # type: ignore

    def before_run(self, runner):
        """Initialize wandb run before the runner starts."""
        # Update config with runner's config if available
        if hasattr(runner, "config") and runner.config is not None:
            if self.config is None:
                self.config = runner.config
            else:
                # Merge configs, prioritizing explicit hook config
                merged_config = runner.config.copy()
                merged_config.update(self.config)
                self.config = merged_config

        # Set up the directory
        if self.dir is None and hasattr(runner, "work_dir"):
            self.dir = os.path.join(runner.work_dir, "wandb")
            os.makedirs(self.dir, exist_ok=True)

        # If name is not specified, generate one based on datetime and config
        if self.name is None:
            import hashlib
            import time

            # Generate a short hash from config
            config_str = str(sorted(self.config.items())) if self.config else ""
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
            self.name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{config_hash}"

        # Initialize wandb run
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            config=self.config,
            dir=self.dir,
            tags=self.tags,
            notes=self.notes,
            group=self.group,
            job_type=self.job_type,
            save_code=self.save_code,
            reinit=True,
        )

        # Watch the model to track gradients, parameters, etc.
        if hasattr(runner, "model") and runner.model is not None:
            try:
                # Watch model with wandb to track gradients and parameters
                wandb.watch(
                    runner.model,
                    log="all",  # Log gradients and parameters
                    log_freq=self.interval,  # Log every interval iterations
                    log_graph=True,  # Log model graph
                )
            except Exception as e:
                if hasattr(runner, "logger"):
                    runner.logger.warning(f"Failed to watch model with wandb: {e}")

        # Log system info
        try:
            import platform

            sys_info = {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }

            if torch.cuda.is_available():
                sys_info.update(
                    {"cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()}
                )

            wandb.run.summary.update(sys_info)

            # Create custom charts for gradient tracking
            if self.log_grad_norm:
                wandb.define_metric("train/grad_norm", summary="max")
                wandb.define_metric("train/grad_norm_mean", summary="mean")
                wandb.define_metric("train/grad_norm_std", summary="std")

        except Exception as e:
            if hasattr(runner, "logger"):
                runner.logger.warning(f"Failed to log system info: {e}")

    def after_train_step(self, runner):
        """Log training metrics to wandb."""
        # Only log every interval iterations
        if runner.iter % self.interval != 0:
            return

        # Prepare metrics dictionary
        metrics_dict = {}

        # Log learning rate
        if runner.optimizer is not None:
            for i, param_group in enumerate(runner.optimizer.param_groups):
                metrics_dict[f"train/lr_{i}"] = param_group["lr"]

        # Log gradient norm if available
        if hasattr(runner, "grad_norm"):
            metrics_dict["train/grad_norm"] = runner.grad_norm.item()

        # Log step metrics if available
        if hasattr(runner, "train_step_metrics") and runner.train_step_metrics:
            for k, v in runner.train_step_metrics.items():
                if k != "grad_norm":  # Already added above
                    metrics_dict[f"train/step_{k}"] = v.item()

        # Add current batch metrics if available in the tqdm progress bar
        if hasattr(runner, "train_dataloader") and hasattr(runner.train_dataloader, "postfix"):
            for k, v in runner.train_dataloader.postfix.items():
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()):
                    try:
                        metrics_dict[f"train/batch_{k}"] = float(v)
                    except ValueError:
                        pass

        # Log all metrics
        if metrics_dict:
            wandb.log(metrics_dict, step=runner.iter)

    def after_train_epoch(self, runner):
        """Log training metrics to wandb after each training epoch."""
        # Log training metrics
        if hasattr(runner, "train_metrics") and runner.train_metrics:
            metrics = {f"train/{k}": v for k, v in runner.train_metrics.items()}
            metrics["epoch"] = runner.current_epoch

            # Add gradient norm histogram if available
            if hasattr(runner, "grad_norm_history") and runner.grad_norm_history:
                # Convert to tensor for wandb
                grad_norm_tensor = torch.tensor(runner.grad_norm_history[-len(runner.train_dataloader) :])
                metrics["train/grad_norm_hist"] = wandb.Histogram(grad_norm_tensor.numpy())

                # Also log min, max, mean, std of gradient norms for this epoch
                if len(grad_norm_tensor) > 0:
                    metrics["train/grad_norm_min"] = grad_norm_tensor.min().item()
                    metrics["train/grad_norm_max"] = grad_norm_tensor.max().item()
                    metrics["train/grad_norm_mean"] = grad_norm_tensor.mean().item()
                    metrics["train/grad_norm_std"] = grad_norm_tensor.std().item()

            wandb.log(metrics, step=runner.iter if hasattr(runner, "iter") else None)

    def after_val_epoch(self, runner):
        """Log validation metrics to wandb after each validation epoch."""
        # Log validation metrics
        if hasattr(runner, "val_metrics") and runner.val_metrics:
            metrics = {f"val/{k}": v for k, v in runner.val_metrics.items()}
            metrics["epoch"] = runner.current_epoch
            wandb.log(metrics, step=runner.iter if hasattr(runner, "iter") else None)

    def after_test_epoch(self, runner):
        """Log test metrics to wandb after test epoch."""
        # Log test metrics
        if hasattr(runner, "test_metrics") and runner.test_metrics:
            metrics = {f"test/{k}": v for k, v in runner.test_metrics.items()}
            metrics["epoch"] = runner.current_epoch
            wandb.log(metrics, step=runner.iter if hasattr(runner, "iter") else None)


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
