from collections.abc import Callable
from typing import Any


class BaseTransform:
    """Base class for all transforms"""

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self) -> str:
        return self.__class__.__name__


class TransformRegistry:
    """Registry for transform classes.

    This class provides a registry for registering and retrieving transform classes.
    """

    _transforms: dict[str, type[BaseTransform]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a transform class.

        Args:
            name: Name of the transform

        Returns:
            Decorator function
        """

        def decorator(transform_cls: type[BaseTransform]) -> type[BaseTransform]:
            if name in cls._transforms:
                raise ValueError(f"Transform '{name}' already registered")

            cls._transforms[name] = transform_cls
            return transform_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseTransform]:
        """Get a transform class by name.

        Args:
            name: Name of the transform

        Returns:
            Transform class

        Raises:
            ValueError: If transform name is not registered
        """
        if name not in cls._transforms:
            raise ValueError(f"Transform '{name}' not found. Available transforms: {list(cls._transforms.keys())}")

        return cls._transforms[name]

    @classmethod
    def available_transforms(cls) -> dict[str, type[BaseTransform]]:
        """Get all available transforms.

        Returns:
            Dictionary of transform names and classes
        """
        return cls._transforms.copy()


class TransformBuilder:
    """Builder for creating transform instances.

    This class provides a builder for creating transform instances based on configuration.
    """

    @staticmethod
    def build_transform(config: dict[str, Any]) -> BaseTransform:
        """Build a transform instance based on configuration.

        Args:
            config: Configuration dictionary containing transform type and parameters

        Returns:
            Transform instance

        Raises:
            ValueError: If transform type is not specified or not registered
        """
        transform_type = config.get("name")

        if transform_type is None:
            raise ValueError("Transform type must be specified in config (type)")

        # Get transform class
        transform_cls = TransformRegistry.get(transform_type)

        # Extract transform parameters
        transform_args = {k: v for k, v in config.items() if k != "name"}

        # Create transform instance
        return transform_cls(**transform_args)

    @classmethod
    def register_transform(cls, name: str, transform_cls: type[BaseTransform]) -> None:
        """Register a new transform.

        Args:
            name: Name of the transform
            transform_cls: Transform class to register
        """
        TransformRegistry._transforms[name] = transform_cls
