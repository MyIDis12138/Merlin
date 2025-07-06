from collections.abc import Callable
from typing import Any


class MetricRegistry:
    """Registry for metrics functions.

    This class provides functionality to register custom metrics functions
    and retrieve them by name.
    """

    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a metric function.

        Args:
            name: Name of the metric function

        Returns:
            Decorator function
        """

        def _register(metric_func: Callable) -> Callable:
            if name in cls._registry:
                raise ValueError(f"Metric function {name} already registered")
            cls._registry[name] = metric_func
            return metric_func

        return _register

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a metric function by name.

        Args:
            name: Name of the metric function

        Returns:
            Metric function

        Raises:
            ValueError: If metric function not found
        """
        if name not in cls._registry:
            raise ValueError(f"Metric function {name} not found in registry")
        return cls._registry[name]

    @classmethod
    def available_metrics(cls) -> dict[str, Callable]:
        """Get all registered metric functions.

        Returns:
            dictionary of registered metric functions
        """
        return cls._registry.copy()


class MetricsBuilder:
    """Builder for creating metric instances.

    This class provides a builder for creating metric instances based on configuration.
    """

    @staticmethod
    def build_metrics(config: dict[str, Any]) -> dict[str, Callable]:
        """Build metric functions from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            dictionary of metric functions
        """
        metrics_dict = {}
        for metric_name in config:
            if isinstance(metric_name, str):
                try:
                    metrics_dict[metric_name] = MetricRegistry.get(metric_name)
                except ValueError:
                    pass
            elif isinstance(metric_name, dict):
                name = metric_name.get("name")
                if name:
                    try:
                        metric_func = MetricRegistry.get(name)
                        metrics_dict[name] = metric_func
                    except ValueError:
                        pass

        return metrics_dict
