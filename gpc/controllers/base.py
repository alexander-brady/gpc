from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import torch
import torch.nn as nn
from shape_extensions import Elements, IntTuple, IntVar
from torch import Tensor

from gpc.models import Policy, WorldModel
from gpc.objectives import Objective, accepts_plans, is_frame_objective
from gpc.trajectory import fit_horizon


class GPC(ABC, nn.Module):
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
        super().__init__()

        if action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer.")
        if is_frame_objective(objective) and type(world_model).decode is WorldModel.decode:
            raise ValueError("A frame objective requires a world model that implements decode.")

        self.policy = policy
        self.world_model = world_model
        self.objective = objective
        self.action_horizon = action_horizon

        self.history_size = max(policy.observation_horizon, world_model.context_horizon)

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
        if self._pending is None or self._action_index >= self._pending.shape[1]:
            frames = torch.stack(list(self._frames), dim=1)
            proprio = torch.stack(list(self._proprio), dim=1)

            plan = self.plan(frames, proprio=proprio, goal=goal)

            self._pending = plan[:, : self.action_horizon]
            self._action_index = 0

        action = self._pending[:, self._action_index]
        self._action_index += 1

        return action

    def encode_context[B: IntVar](self, frames: Tensor[[B, int, int, int, int]]) -> Tensor:
        """Encode the latest observed frames into the world model's context latents.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition on

        Returns:
            Tensor[B, K', D...]: latents of the ``context_horizon`` latest frames.
        """
        return self.world_model.encode(fit_horizon(frames, self.world_model.context_horizon))

    def simulate[B: IntVar, N: IntVar, T: IntVar, D: IntTuple](
        self,
        context: Tensor[[B, int, *Elements[D]]],
        plans: Tensor[[B, N, T, int]],
    ) -> Tensor[[B, N, T, *Elements[D]]]:
        """Roll out each candidate plan with the world model.

        Args:
            context (Tensor[B, K, D...]): context latents from [`encode_context`](gpc.controllers.base.GPC.encode_context)
            plans (Tensor[B, N, T, A]): candidate action sequences

        Returns:
            Tensor[B, N, T, D...]: predicted latents.
        """
        batch, num_plans = plans.shape[:2]
        context = context.repeat_interleave(num_plans, dim=0)
        futures = self.world_model.rollout(context, plans.flatten(0, 1))
        return futures.unflatten(0, (batch, num_plans))

    def score[B: IntVar, N: IntVar](
        self,
        futures: Tensor,
        goal: Tensor[[B, int]] | None,
        plans: Tensor[[B, N, int, int]],
    ) -> Tensor[[B, N]]:
        """Score simulated futures with the objective, decoding them into frames for a frame
        objective and passing the plans if it accepts them.

        Args:
            futures (Tensor[B, N, T, D...]): latent futures from [`simulate`](gpc.controllers.base.GPC.simulate)
            goal (Tensor[B, GoalDim] | None): the optional goal to evaluate against
            plans (Tensor[B, N, T, A]): the candidate plans that produced the futures

        Returns:
            Tensor[B, N]: one score per batch and plan.
        """
        objective = self.objective
        if is_frame_objective(objective):
            futures = self.world_model.decode(futures.flatten(0, 1)).unflatten(0, plans.shape[:2])
        if accepts_plans(objective):
            return objective(futures, goal, plans=plans)
        return objective(futures, goal)

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
