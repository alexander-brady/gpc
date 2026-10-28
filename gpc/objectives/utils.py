"""Helpers for marking and inspecting objectives."""

from __future__ import annotations

import inspect
from typing import TypeGuard

from shape_extensions import IntTuple
from torch import nn
from typing_extensions import TypeIs

from gpc.objectives.protocol import Objective, PlanObjective


def frame_objective[O: Objective[[int, int, int]] | type[Objective[[int, int, int]]]](
    objective: O,
) -> O:
    """Mark an objective (function or class) as scoring frames [B, N, T, C, H, W], decoded
    from the world model's latents. Equivalent to setting ``frame_objective = True``.

    Example:
        ```python
        @frame_objective
        def brightness(futures, goal):
            return futures[:, :, -1].mean(dim=(2, 3, 4))
        ```
    """
    objective.frame_objective = True  # pyrefly: ignore[missing-attribute]
    return objective


def latent_objective[O: Objective | type[Objective]](objective: O) -> O:
    """Mark an objective (function or class) as scoring latents [B, N, T, D...].
    This is already the default, so the marker only makes it explicit.
    Equivalent to setting ``frame_objective = False``.

    Example:
        ```python
        @latent_objective
        def smoothness(futures, goal):
            return (futures[:, :, 1:] - futures[:, :, :-1]).abs().mean(dim=(2, 3))
        ```"""
    objective.frame_objective = False  # pyrefly: ignore[missing-attribute]
    return objective


def is_frame_objective(objective: Objective) -> TypeGuard[Objective[[int, int, int]]]:
    """Whether the objective scores frames rather than latents, narrowing it to a frame
    objective. Only the true branch narrows: an unmarked objective may still be typed for
    frames."""
    return getattr(objective, "frame_objective", False)


def accepts_plans[Ds: IntTuple](objective: Objective[Ds]) -> TypeIs[PlanObjective[Ds]]:
    """Whether the objective takes a keyword ``plans`` argument (or ``**kwargs``), narrowing
    it to a [`PlanObjective`](gpc.objectives.PlanObjective), and otherwise to a [`FutureObjective`](gpc.objectives.FutureObjective).
    For an ``nn.Module``, its ``forward`` signature is inspected."""
    target = objective.forward if isinstance(objective, nn.Module) else objective
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind is p.VAR_KEYWORD
        or (p.name == "plans" and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD))
        for p in parameters
    )
