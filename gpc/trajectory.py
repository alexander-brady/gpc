"""Helpers for time-indexed tensors laid out as [B, K, ...]."""

from __future__ import annotations

import torch
from shape_extensions import Elements, Int, IntTuple, IntVar
from torch import Tensor


def fit_horizon[B: IntVar, K: IntVar, H: IntVar, Ds: IntTuple](
    history: Tensor[[B, K, *Elements[Ds]]], horizon: Int[H]
) -> Tensor[[B, H, *Elements[Ds]]]:
    """Fit a [B, K, ...] history to exactly ``horizon`` steps.

    Keeps the latest ``horizon`` steps of a longer history, and left-pads a shorter one by
    repeating its earliest step.

    Args:
        history (Tensor[B, K, ...]): History with at least one step.
        horizon (int > 0): Number of steps to fit the history to.

    Returns:
        Tensor[B, horizon, ...]: The fitted history.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if history.shape[1] == 0:
        raise ValueError("history must contain at least one step.")

    fitted = history[:, -horizon:]
    missing = horizon - fitted.shape[1]
    if missing == 0:
        return fitted
    padding = fitted[:, :1].expand(-1, missing, *fitted.shape[2:])
    return torch.cat([padding, fitted], dim=1)
