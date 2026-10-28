from __future__ import annotations

from typing import Protocol

from shape_extensions import Elements, IntTuple, IntVar
from torch import Tensor


class FutureObjective[Ds: IntTuple](Protocol):
    """Function that scores simulated futures against a goal.
    Higher scores indicate better alignment with the goal."""

    def __call__[B: IntVar, N: IntVar](
        self,
        futures: Tensor[[B, N, int, *Elements[Ds]]],
        goal: Tensor[[B, int]] | None,
    ) -> Tensor[[B, N]]:
        """Score candidate futures, each simulated by the world model from one plan.

        Args:
            futures (Tensor[B, N, T, ...]): predicted trajectory of each candidate plan,
                as latents [B, N, T, D...] or as frames [B, N, T, C, H, W]
            goal (Tensor[B, GoalDim] | None): the optional goal to evaluate against

        Returns:
            Tensor[B, N]: Scores of the candidate futures, one per batch and plan.
                Higher scores indicate better alignment with the goal.
        """


class PlanObjective[Ds: IntTuple](Protocol):
    """[`FutureObjective`](gpc.objectives.FutureObjective) that also receives the candidate
    plans, e.g. to penalize action magnitude."""

    def __call__[B: IntVar, N: IntVar](
        self,
        futures: Tensor[[B, N, int, *Elements[Ds]]],
        goal: Tensor[[B, int]] | None,
        *,
        plans: Tensor[[B, N, int, int]],
    ) -> Tensor[[B, N]]:
        """Score candidate futures together with the plans that produced them.

        Args:
            futures (Tensor[B, N, T, ...]): predicted trajectory of each candidate plan,
                as latents [B, N, T, D...] or frames [B, N, T, C, H, W]
            goal (Tensor[B, GoalDim] | None): the optional goal to evaluate against
            plans (Tensor[B, N, T, A]): the candidate plans that produced the futures

        Returns:
            Tensor[B, N]: Scores of the candidate futures, one per batch and plan.
                Higher scores indicate better alignment with the goal.
        """


type Objective[Ds: IntTuple] = FutureObjective[Ds] | PlanObjective[Ds]
"""Function that scores simulated futures, and optionally their plans, against a goal.

An objective that takes a keyword ``plans`` argument (or ``**kwargs``) is a
[`PlanObjective`](gpc.objectives.PlanObjective) and receives the candidate plans; any other
is a [`FutureObjective`](gpc.objectives.FutureObjective).

By default an objective scores futures in the world model's latent space, with ``Ds`` the
latent shape. Mark it with [`frame_objective`](gpc.objectives.frame_objective) (or set a
``frame_objective = True`` attribute) to have the controller decode the futures into frames
first, which makes it an ``Objective[[C, H, W]]``.
"""
