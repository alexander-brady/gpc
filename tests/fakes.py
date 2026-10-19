"""Shared minimal fakes for the abstract GPC components."""

import torch
from torch import Tensor

from gpc.controllers.base import GPC
from gpc.models import Policy, PolicyConfig, WorldModel, WorldModelConfig

B, C, H, W, P, A = 2, 3, 4, 4, 5, 2


class FakePolicy(Policy):
    """Returns a distinct, deterministic plan per batch row: arange(B*T*A).
    Records the observations it is sampled with."""

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.calls: list[tuple[Tensor, Tensor]] = []

    def sample(self, frames: Tensor, proprio: Tensor) -> Tensor:
        self.calls.append((frames, proprio))
        b, t = frames.shape[0], self.prediction_horizon
        return torch.arange(b * t * A, dtype=torch.float).reshape(b, t, A)


class FakeWorldModel(WorldModel):
    """Latent = flattened frame mean; prediction echoes the actions, recording each context."""

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.contexts: list[Tensor] = []

    def encode(self, frames: Tensor) -> Tensor:
        return frames.flatten(2).mean(-1, keepdim=True)

    def predict(self, history: Tensor, actions: Tensor) -> Tensor:
        self.contexts.append(history)
        return actions


class SumObjective:
    """Scores each plan by the sum of its actions, recording the calls it receives."""

    def __init__(self):
        self.calls: list[tuple[Tensor, Tensor, Tensor | None]] = []

    def __call__(self, frames: Tensor, plans: Tensor, goal: Tensor | None) -> Tensor:
        self.calls.append((frames, plans, goal))
        return plans.sum(dim=(2, 3))


class StubGPC(GPC):
    """Concrete GPC whose plan returns plan_len steps numbered by the replan count."""

    plan_len = 4

    def __init__(self, *args, **kwargs):
        self.plan_calls: list[tuple[Tensor, Tensor, Tensor | None]] = []
        super().__init__(*args, **kwargs)

    def plan(self, frames: Tensor, proprio: Tensor, goal: Tensor | None = None) -> Tensor:
        self.plan_calls.append((frames, proprio, goal))
        n = len(self.plan_calls)
        steps = torch.arange(self.plan_len, dtype=torch.float) + 100 * n
        return steps.view(1, -1, 1).expand(frames.shape[0], -1, A)


def make_obs(b: int = B, k: int = 1, fill: float = 0.0) -> tuple[Tensor, Tensor]:
    """Create a fake observation with `b` batch rows and `k` frames per row, filled with `fill`."""
    return torch.full((b, k, C, H, W), fill), torch.full((b, k, P), fill)


def step(gpc: GPC, fill: float = 0.0, goal: Tensor | None = None) -> Tensor:
    """Feed a single-frame observation to the controller and return its action."""
    frames, proprio = make_obs(fill=fill)
    return gpc.act(frames, proprio=proprio, goal=goal)
