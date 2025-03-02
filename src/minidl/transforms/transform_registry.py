from typing import Any, Callable, Dict, List, Optional, Type

from .mri_transforms import BaseTransform, MRITransformPipeline


class TransformRegistry:
    """Registry for transform classes.

    This class provides a registry for registering and retrieving transform classes.
    """

    _transforms: Dict[str, Type[BaseTransform]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a transform class.

        Args:
            name: Name of the transform

        Returns:
            Decorator function
        """

        def decorator(transform_cls: Type[BaseTransform]) -> Type[BaseTransform]:
            if name in cls._transforms:
                raise ValueError(f"Transform '{name}' already registered")

            cls._transforms[name] = transform_cls
            return transform_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseTransform]:
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
    def available_transforms(cls) -> Dict[str, Type[BaseTransform]]:
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
    def build_transform(config: Dict[str, Any]) -> BaseTransform:
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

    @staticmethod
    def build_transform_pipeline(transform_configs: Optional[List[Dict[str, Any]]] = None) -> Optional[MRITransformPipeline]:
        """Build a transform pipeline based on configuration.

        Args:
            transform_configs: List of transform configurations

        Returns:
            Transform pipeline instance, or None if no transforms are specified
        """
        if not transform_configs:
            return None

        transforms = []
        for config in transform_configs:
            transform = TransformBuilder.build_transform(config)
            transforms.append(transform)

        return MRITransformPipeline(transforms)

    @classmethod
    def register_transform(cls, name: str, transform_cls: Type[BaseTransform]) -> None:
        """Register a new transform.

        Args:
            name: Name of the transform
            transform_cls: Transform class to register
        """
        TransformRegistry._transforms[name] = transform_cls

    @classmethod
    def build_transforms(cls, transform_configs: Optional[list] = None) -> Optional[MRITransformPipeline]:
        """Build a transform pipeline based on configuration (legacy method).

        Args:
            transform_configs: List of transform configurations

        Returns:
            Transform pipeline instance, or None if no transforms are specified
        """
        if transform_configs is None or len(transform_configs) == 0:
            return None

        return cls.build_transform_pipeline(transform_configs)
