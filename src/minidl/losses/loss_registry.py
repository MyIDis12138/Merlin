from typing import Callable, Dict, Type

import torch.nn as nn


class LossRegistry:
    """Registry for loss functions.

    This class provides functionality to register custom loss functions
    and retrieve them by name.
    """

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a loss function.

        Args:
            name: Name of the loss function

        Returns:
            Decorator function
        """

        def _register(loss_class: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._registry:
                raise ValueError(f"Loss function {name} already registered")
            cls._registry[name] = loss_class
            return loss_class

        return _register

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
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
    def available_losses(cls) -> Dict[str, Type[nn.Module]]:
        """Get all registered loss functions.

        Returns:
            Dictionary of registered loss functions
        """
        return cls._registry.copy()
