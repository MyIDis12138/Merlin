from typing import Any, Callable, Dict, Optional, Type

import torch

from .base_runner import BaseRunner
from .hooks import HookBuilder


class RunnerRegistry:
    """Registry for runner classes.

    This class provides a registry for runner classes, allowing them to be registered
    and retrieved by name.
    """

    _runners: Dict[str, Type[BaseRunner]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a runner class.

        Args:
            name: Name of the runner

        Returns:
            Decorator function for registration
        """

        def decorator(runner_cls: Type[BaseRunner]) -> Type[BaseRunner]:
            cls._runners[name] = runner_cls
            return runner_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseRunner]:
        """Get a runner class by name.

        Args:
            name: Name of the runner

        Returns:
            Runner class

        Raises:
            ValueError: If runner name is not registered
        """
        if name not in cls._runners:
            raise ValueError(f"Runner '{name}' not found. Available runners: {list(cls._runners.keys())}")

        return cls._runners[name]

    @classmethod
    def available_runners(cls) -> Dict[str, Type[BaseRunner]]:
        """Get all available runners.

        Returns:
            Dictionary of runner names and classes
        """
        return cls._runners.copy()


class RunnerBuilder:
    """Builder for creating runner instances.

    This class provides a builder for creating runner instances based on configuration.
    """

    @staticmethod
    def build_runner(config: Dict[str, Any], device: Optional[torch.device] = None) -> BaseRunner:
        """Build a runner instance based on configuration.

        Args:
            config: Configuration dictionary
            device: Device to run on

        Returns:
            Runner instance

        Raises:
            ValueError: If runner type is not specified or not registered
        """
        runner_type = config.get("runner", {}).get("name")

        if runner_type is None:
            raise ValueError("Runner name must be specified in config under 'runner.name'")

        runner_cls = RunnerRegistry.get(runner_type)

        # Create runner instance
        runner = runner_cls(config, device)

        # Build and register hooks if specified in config
        hooks_config = config.get("runner", {}).get("hooks")
        if hooks_config:
            work_dir = config.get("work_dir")
            hooks = HookBuilder.build_hooks(hooks_config, work_dir)
            for hook in hooks:
                if hasattr(runner, "register_hook"):
                    runner.register_hook(hook)
                else:
                    raise AttributeError(f"Runner {runner_type} does not have a register_hook method")

        return runner
