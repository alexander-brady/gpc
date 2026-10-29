from collections.abc import Callable

import torch
from shape_extensions import Int, IntVar
from torch import Tensor
from torch.optim import SGD, Optimizer

from gpc.controllers.base import GPC
from gpc.controllers.opt import GPCOpt
from gpc.controllers.rank import GPCRank
from gpc.models import Policy, WorldModel
from gpc.objectives import Objective


class GPCRankOpt[N: IntVar](GPCRank[N], GPCOpt):
    """Samples action proposals from a policy, refines each by gradient ascent on the objective
    through the world model, and returns the refined proposal whose simulated future scores
    highest.

    This solves the optimization of [`GPCOpt`](gpc.GPCOpt) from ``num_samples``
    initializations, at the cost of ``num_samples * (num_steps + 1)`` world model rollouts per
    replanning step. The same differentiability requirements apply.

    Attributes:
        policy (Policy): Policy model used to propose initial plans.
        world_model (WorldModel): World model used for predicting future states.
        objective (Objective): Objective function plans are optimized and ranked by.
        action_horizon (int): Number of action steps to execute before replanning.
        num_samples (Int[N] > 0): Number of action sequences to sample and refine.
        spread (float): Factor by which proposals are pushed away from their mean, for diversity.
            Spread proposals are clamped to the policy's action bounds.
        num_steps (int >= 0): Number of gradient steps applied to each proposal.
        step_size (float > 0): Learning rate passed to the optimizer.
        optimizer (Callable[..., Optimizer]): Optimizer factory, called as
            ``optimizer([plans], lr=step_size)``. Defaults to plain gradient ascent.
        rank_first_step (bool): Whether to rank when only a single frame has been observed.
            Otherwise, the first proposal is returned.
        optimize_first_step (bool): Whether to optimize when only a single frame has been
            observed. Otherwise, the proposals are ranked unchanged.
    """

    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        objective: Objective,
        *,
        action_horizon: int = 1,
        num_samples: Int[N] = 16,
        spread: float = 1.01,
        num_steps: int = 25,
        step_size: float = 5e-3,
        optimizer: Callable[..., Optimizer] = SGD,
        rank_first_step: bool = False,
        optimize_first_step: bool = False,
    ):
        GPC.__init__(self, policy, world_model, objective, action_horizon=action_horizon)
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if num_steps < 0:
            raise ValueError("num_steps must be a non-negative integer.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        self.num_samples = num_samples
        self.spread = spread
        self.num_steps = num_steps
        self.step_size = step_size
        self.optimizer = optimizer
        self.rank_first_step = rank_first_step
        self.optimize_first_step = optimize_first_step

    @torch.no_grad()
    def plan[B: IntVar, K: IntVar](
        self,
        frames: Tensor[[B, K, int, int, int]],
        proprio: Tensor[[B, K, int]],
        goal: Tensor[[B, int]] | None = None,
    ) -> Tensor[[B, int, int]]:
        """Generate a plan of actions based on the current observations and goal.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition the plan on.
            proprio (Tensor[B, K, P]): observed proprioceptive data to condition the plan on.
            goal (Tensor[B, G] | None): optional goal to condition the plan on.

        Returns:
            Tensor[B, T, A]: the best optimized plan of actions.
        """
        plans = self._propose(frames, proprio)
        first_step = frames.shape[1] == 1
        if first_step and not self.rank_first_step:
            return plans[:, 0]
        if not first_step or self.optimize_first_step:
            plans = self.optimize(frames, plans, goal)
        return self.rank(frames, plans, goal)
