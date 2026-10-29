from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
from shape_extensions import IntVar
from torch import Tensor, autograd, nn
from torch.optim import SGD, Optimizer

from gpc.controllers.base import GPC
from gpc.models import Policy, WorldModel
from gpc.objectives import Objective


@contextmanager
def frozen(module: nn.Module) -> Generator[None]:
    """Disable gradients on the module's trainable parameters, restoring them on exit."""
    params = [p for p in module.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


class GPCOpt(GPC):
    """Samples one action proposal from a policy as a warm start, then refines it by gradient
    ascent on the objective, differentiating through the world model.

    The world model, its decoder for a frame objective, and the objective must be
    differentiable with respect to the actions. A stochastic world model should be made
    deterministic at inference time (e.g. by freezing its sampling noise), or its noisy
    gradients destabilize the optimization.

    Attributes:
        policy (Policy): Policy model used to warm-start the plan.
        world_model (WorldModel): World model used for predicting future states.
        objective (Objective): Objective function the plan is optimized to maximize.
        action_horizon (int): Number of action steps to execute before replanning.
        num_steps (int >= 0): Number of gradient steps per replanning step.
        step_size (float > 0): Learning rate passed to the optimizer.
        optimizer (Callable[..., Optimizer]): Optimizer factory, called as
            ``optimizer([plan], lr=step_size)``. Defaults to plain gradient ascent;
            pass e.g. ``torch.optim.Adam`` or a ``functools.partial`` for other settings.
        optimize_first_step (bool): Whether to optimize when only a single frame has been
            observed. Otherwise, the policy's proposal is returned unchanged.
    """

    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        objective: Objective,
        *,
        action_horizon: int = 1,
        num_steps: int = 25,
        step_size: float = 5e-3,
        optimizer: Callable[..., Optimizer] = SGD,
        optimize_first_step: bool = False,
    ):
        super().__init__(policy, world_model, objective, action_horizon=action_horizon)
        if num_steps < 0:
            raise ValueError("num_steps must be a non-negative integer.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        self.num_steps = num_steps
        self.step_size = step_size
        self.optimizer = optimizer
        self.optimize_first_step = optimize_first_step

    def optimize[B: IntVar, N: IntVar](
        self,
        frames: Tensor[[B, int, int, int, int]],
        plans: Tensor[[B, N, int, int]],
        goal: Tensor[[B, int]] | None = None,
    ) -> Tensor[[B, N, int, int]]:
        """Refine each candidate plan independently by ``num_steps`` steps of gradient ascent
        on its score, projecting them back onto the policy's action bounds after each step.

        Only the plans are optimized: the controller's parameters are frozen meanwhile, so
        autograd neither accumulates their gradients nor keeps activations to compute them.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition on
            plans (Tensor[B, N, T, A]): candidate plans to start from
            goal (Tensor[B, GoalDim] | None): the optional goal to optimize towards

        Returns:
            Tensor[B, N, T, A]: the optimized plans, detached from the graph.
        """
        with torch.no_grad():
            context = self.encode_context(frames)
        plans = plans.detach().clone().requires_grad_(True)
        optimizer = self.optimizer([plans], lr=self.step_size)
        with torch.enable_grad(), frozen(self):
            for _ in range(self.num_steps):
                # Plans are scored independently, so the gradient of the summed score with
                # respect to each plan is that plan's own score gradient.
                score = self.score(self.simulate(context, plans), goal, plans).sum()
                (grad,) = autograd.grad(score, plans)
                plans.grad = -grad  # optimizers minimize
                optimizer.step()
                with torch.no_grad():
                    plans.copy_(self.policy.clamp_actions(plans))
        return plans.detach()

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
            Tensor[B, T, A]: the optimized plan of actions.
        """
        plans = self._sample(frames, proprio)
        if frames.shape[1] == 1 and not self.optimize_first_step:
            return plans
        return self.optimize(frames, plans[:, None], goal)[:, 0]
