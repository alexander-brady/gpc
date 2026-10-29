"""Shared minimal fakes for the abstract GPC components."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gpc.controllers.base import GPC
from gpc.models import Config, Policy, PolicyConfig, WorldModel, WorldModelConfig
from gpc.objectives import ObjectiveModule

B, C, H, W, P, A = 2, 3, 4, 4, 5, 2


class FakePolicy(Policy):
    """Returns a distinct, deterministic plan per batch row: arange(B*T*A) + offset.
    Records the observations it is sampled with."""

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.offset = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[Tensor, Tensor]] = []

    def sample(self, frames: Tensor, proprio: Tensor) -> Tensor:
        self.calls.append((frames, proprio))
        b, t = frames.shape[0], self.prediction_horizon
        return torch.arange(b * t * A, dtype=torch.float).reshape(b, t, A) + self.offset


class FakeWorldModel(WorldModel):
    """Latent = flattened frame mean; prediction = per-step action sum, recording each context."""

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.contexts: list[Tensor] = []

    def encode(self, frames: Tensor) -> Tensor:
        return frames.flatten(2).mean(-1, keepdim=True)

    def predict(self, history: Tensor, actions: Tensor) -> Tensor:
        self.contexts.append(history)
        return actions.sum(dim=-1, keepdim=True)


class DecodingWorldModel(FakeWorldModel):
    """FakeWorldModel that decodes each latent into a frame filled with its first element."""

    def decode(self, latents: Tensor) -> Tensor:
        return latents[..., :1, None, None].expand(*latents.shape[:-1], C, H, W)


class SumObjective:
    """Scores each future by the sum of its elements, recording the calls it receives.
    With FakeWorldModel, the latent future sums each step's actions, so this sums the plan."""

    def __init__(self):
        self.calls: list[tuple[Tensor, Tensor | None, Tensor | None]] = []

    def __call__(
        self, futures: Tensor, goal: Tensor | None, *, plans: Tensor | None = None
    ) -> Tensor:
        self.calls.append((futures, goal, plans))
        return futures.flatten(2).sum(dim=-1)


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


def total_objective(futures: Tensor, goal: Tensor | None) -> Tensor:
    """Module-level objective, saved by reference."""
    return futures.flatten(2).sum(dim=-1)


def scaled_objective(futures: Tensor, goal: Tensor | None, *, scale: float = 1.0) -> Tensor:
    """Parameterized objective, saved as a ``functools.partial``."""
    return scale * futures.flatten(2).sum(dim=-1)


@dataclass(kw_only=True)
class WeightedObjectiveConfig(Config):
    scale: float = 1.0


class WeightedObjective(ObjectiveModule):
    """Module objective with a setting and a weight, saved to ``objective/``."""

    config_class = WeightedObjectiveConfig

    def __init__(self, config: WeightedObjectiveConfig | None = None):
        super().__init__(config)
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, futures: Tensor, goal: Tensor | None) -> Tensor:
        return self.config.scale * self.weight * futures.flatten(2).sum(dim=-1)
