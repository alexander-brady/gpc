from __future__ import annotations

from typing import Protocol

from shape_extensions import IntVar
from torch import Tensor


class Objective(Protocol):
    """Function that scores a future against a goal.
    Higher scores indicate better alignment with the goal."""

    def __call__[B: IntVar, N: IntVar](
        self,
        frames: Tensor[[B, int, int, int, int]],
        plans: Tensor[[B, N, int, int]],
        goal: Tensor[[B, int]] | None,
    ) -> Tensor[[B, N]]:
        """Score candidate plans by the objective of their futures,
        simulated from the observed frames.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition on
            plans (Tensor[B, N, T, A]): candidate action sequences to evaluate
            goal (Tensor[B, GoalDim] | None): the optional goal to evaluate against

        Returns:
            Tensor[B, N]: Scores of the candidate plans, one per batch and plan.
        """
