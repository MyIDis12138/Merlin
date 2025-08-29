from typing import Any

import torch.nn as nn

from .loss_registry import LossRegistry


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Build a loss function from configuration.

    Args:
        config: Loss configuration dictionary

    Returns:
        A PyTorch loss function
    """
    # Check if loss is directly in config or nested under 'training'
    if "loss" in config:
        loss_config = config.get("loss", {})
    elif "training" in config and "loss" in config["training"]:
        loss_config = config["training"].get("loss", {})
    else:
        loss_config = {}

    loss_type = loss_config.get("name", "CrossEntropyLoss")

    # Try to get the loss from the registry first
    try:
        loss_class = LossRegistry.get(loss_type)
    except ValueError:
        # If not in registry, try to get from torch.nn
        if hasattr(nn, loss_type):
            loss_class = getattr(nn, loss_type)
        else:
            raise ValueError(f"Loss type {loss_type} not found in registry or torch.nn")

    # Instantiate the loss function
    return loss_class(**loss_config.get("params", {}))
