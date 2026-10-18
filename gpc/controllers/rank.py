import torch
from shape_extensions import Int, IntVar
from torch import Tensor

from gpc.controllers.base import GPC
from gpc.models import Policy, WorldModel
from gpc.objectives import Objective


class GPCRank[N: IntVar](GPC):
    """Samples action proposals from a policy, simulates each with the world model, and returns
    the proposal whose simulated future scores highest.

    Attributes:
        policy (Policy): Policy model used for action selection.
        world_model (WorldModel): World model used for predicting future states.
        objective (Objective): Objective function used to evaluate action plans.
        action_horizon (int): Number of action steps to execute before replanning.
        num_samples (Int[N] > 0): Number of action sequences to sample for planning.
        spread (float): Factor by which proposals are pushed away from their mean, for diversity.
        rank_first_step (bool): Whether to rank when only a single frame has been observed.
            Otherwise, the first proposal generated is returned.
    """

    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        objective: Objective,
        *,
        action_horizon: int = 1,
        num_samples: Int[N] = 64,
        spread: float = 1.01,
        rank_first_step: bool = False,
    ):
        super().__init__(policy, world_model, objective, action_horizon=action_horizon)
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        self.num_samples = num_samples
        self.spread = spread
        self.rank_first_step = rank_first_step

    @torch.no_grad()
    def _propose[B: IntVar, K: IntVar](
        self,
        frames: Tensor[[B, K, int, int, int]],
        proprio: Tensor[[B, K, int]],
    ) -> Tensor[[B, N, int, int]]:
        """Sample ``num_samples`` candidate action trajectories [B, N, T, A]."""
        n = self.num_samples
        plans = self.policy.sample(frames.repeat_interleave(n, 0), proprio.repeat_interleave(n, 0))
        plans = plans.reshape(-1, n, *plans.shape[1:])
        mean = plans.mean(dim=1, keepdim=True)
        return mean + self.spread * (plans - mean)

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
            Tensor[B, T, A]: the generated plan of actions.
        """
        plans = self._propose(frames, proprio)
        if frames.shape[1] == 1 and not self.rank_first_step:
            return plans[:, 0]
        # The first maximum wins ties.
        best = self.objective(frames, plans, goal).argmax(dim=1)
        return plans[torch.arange(plans.shape[0]), best]
