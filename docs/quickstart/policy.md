# 1. Policy

A policy subclasses [`Policy`][gpc.models.Policy] and implements `sample`, which proposes one
plan of `T` actions per batch element from the latest observations.

```python
import torch
from torch import Tensor

from gpc.models import Policy, PolicyConfig


class MyPolicy(Policy):
    def __init__(self, config: PolicyConfig, action_dim: int):
        super().__init__(config)
        self.action_dim = action_dim

    def sample(self, frames: Tensor, proprio: Tensor) -> Tensor:
        # frames: [B, K, C, H, W], proprio: [B, K, P] -> actions: [B, T, A]
        batch = frames.shape[0]
        return torch.randn(batch, self.prediction_horizon, self.action_dim)


policy = MyPolicy(PolicyConfig(observation_horizon=2, prediction_horizon=8), action_dim=2)
```

`PolicyConfig` sets two horizons:

- `observation_horizon` (`K`): how many recent observations the policy conditions on.
- `prediction_horizon` (`T`): how many actions each sampled plan contains.

`sample` always receives exactly `observation_horizon` observations. Early in an
episode, when fewer have been observed, the controller pads the history by repeating
the earliest observation.

!!! tip "Sampling should be stochastic"

    The controller calls `sample` once per candidate plan and relies on the variety
    between them. A deterministic policy gives the controller nothing to choose from.

Next, build a [world model](world-model.md) to simulate these plans.
