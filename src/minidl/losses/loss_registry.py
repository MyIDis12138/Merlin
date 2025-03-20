from collections.abc import Callable

import torch.nn as nn


class LossRegistry:
    """Registry for loss functions.

    This class provides functionality to register custom loss functions
    and retrieve them by name.
    """

    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a loss function.

        Args:
            name: Name of the loss function

        Returns:
            Decorator function
        """

        def _register(loss_class: type[nn.Module]) -> type[nn.Module]:
            if name in cls._registry:
                raise ValueError(f"Loss function {name} already registered")
            cls._registry[name] = loss_class
            return loss_class

        return _register

    @classmethod
    def get(cls, name: str) -> type[nn.Module]:
        """Get a loss function by name.

        Args:
            name: Name of the loss function

        Returns:
            Loss class

        Raises:
            ValueError: If loss function not found
        """
        if name not in cls._registry:
            raise ValueError(f"Loss function {name} not found in registry")
        return cls._registry[name]

    @classmethod
    def available_losses(cls) -> dict[str, type[nn.Module]]:
        """Get all registered loss functions.

        Returns:
            Dictionary of registered loss functions
        """
        return cls._registry.copy()
