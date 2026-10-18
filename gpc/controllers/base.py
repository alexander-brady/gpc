from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import torch
from shape_extensions import IntVar
from torch import Tensor

from gpc.models import Policy, WorldModel
from gpc.objectives import Objective


class GPC(ABC):
    """Abstract base class for a Generative Predictive Control (GPC) controller.

    Attributes:
        policy (Policy): The generative policy model used for action selection.
        world_model (WorldModel): World model used for predicting future states.
        objective (Objective): The objective function used to evaluate action plans.
        action_horizon (int > 0): The number of action steps to execute before replanning.
    """

    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        objective: Objective,
        *,
        action_horizon: int = 1,
    ):
        if action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer.")

        self.policy = policy
        self.world_model = world_model
        self.objective = objective
        self.action_horizon = action_horizon

        self.history_size = max(self.policy.observation_horizon, self.world_model.context_horizon)

        self.reset()

    def reset(self) -> None:
        """Reset the internal state of the GPC controller."""
        self._frames = deque(maxlen=self.history_size)
        self._proprio = deque(maxlen=self.history_size)

        self._pending: Tensor | None = None
        self._action_index: int = 0

    def act[B: IntVar](
        self,
        frames: Tensor[[B, 1, int, int, int]],
        *,
        proprio: Tensor[[B, 1, int]],
        goal: Tensor[[B, int]] | None = None,
    ) -> Tensor[[B, int]]:
        """Consume an observation and return the next action.

        Args:
            frames (Tensor[B, K, C, H, W]): Observed frames to condition
            proprio (Tensor[B, K, P]): Observed proprioception
            goal (Tensor[B, GoalDim]): Goal to plan towards.

        Returns:
            Tensor[B, A]: the next action to take.
        """
        self._frames.extend(frames.unbind(dim=1))
        self._proprio.extend(proprio.unbind(dim=1))

        # Replan when the action chunk is exhausted
        if self._pending is None or self._action_index >= len(self._pending):
            frames = torch.stack(list(self._frames), dim=1)
            proprio = torch.stack(list(self._proprio), dim=1)

            plan = self.plan(frames, proprio=proprio, goal=goal)

            self._pending = plan[:, : self.action_horizon]
            self._action_index = 0

        action = self._pending[:, self._action_index]
        self._action_index += 1

        return action

    @abstractmethod
    def plan[B: IntVar, K: IntVar](
        self,
        frames: Tensor[[B, K, int, int, int]],
        proprio: Tensor[[B, K, int]],
        goal: Tensor[[B, int]] | None = None,
    ) -> Tensor[[B, int, int]]:
        """Plan from an explicit observation history. Not stateful.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition
            proprio (Tensor[B, K, P]): observed proprioception
            goal (Tensor[B, GoalDim]): the goal to plan towards.

        Returns:
            Tensor[B, T, A]: the next action(s) to take.
        """
