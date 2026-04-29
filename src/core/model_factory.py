"""Model factory for matching/mining stages."""

from __future__ import annotations

import torch

from src.core import models, utils

__all__ = [
    "build_from_args",
    "load_state_dict_if_needed",
]


def build_from_args(args, *, for_inference: bool = False):
    """Build a model instance from CLI args.

    This keeps model construction logic centralized and consistent.
    """
    if args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)

    model.to(utils.get_device())
    if for_inference:
        model.eval()
    return model


def load_state_dict_if_needed(model, model_path: str | None):
    """Load model weights when a path is provided."""
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=utils.get_device()))
    return model
