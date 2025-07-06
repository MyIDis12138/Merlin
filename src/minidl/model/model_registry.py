from collections.abc import Callable
from typing import Any

import torch.nn as nn

_MODEL_REGISTRY = {}


class ModelRegistry:
    """
    model registry

    This class provides a registry for models.
    """

    @staticmethod
    def register(name: str) -> Callable:
        """register a model class

        Args:
            name: model name

        Returns:
            decorator function
        """

        def decorator(model_cls: type[nn.Module]) -> type[nn.Module]:
            if name in _MODEL_REGISTRY:
                raise ValueError(f"model '{name}' already registered")

            _MODEL_REGISTRY[name] = model_cls
            return model_cls

        return decorator

    @staticmethod
    def get(name: str) -> type[nn.Module]:
        """get model class by name

        Args:
            name: model name

        Returns:
            model class

        Raises:
            ValueError: model not found
        """
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' not found, model list: {list(_MODEL_REGISTRY.keys())}")

        return _MODEL_REGISTRY[name]

    @staticmethod
    def available_models() -> dict[str, type[nn.Module]]:
        """get all available models

        Returns:
            model name and class dictionary
        """
        return _MODEL_REGISTRY.copy()


class ModelBuilder:
    """
    model builder

    This class provides a builder for building model instances.
    """

    @staticmethod
    def build_model(config: dict[str, Any]) -> nn.Module:
        """build model instance by config

        Args:
            config: model config dict

        Returns:
            model instance

        Raises:
            ValueError: model name not specified or not registered
        """
        model_config = config.get("model", {})
        model_name = model_config.get("name")

        if model_name is None:
            raise ValueError("Config must specify model name (model.name)")

        model_cls = ModelRegistry.get(model_name)

        return model_cls(**model_config.get("params", {}))
