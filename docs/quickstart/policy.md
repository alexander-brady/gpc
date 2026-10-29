# 1. Policy

A policy subclasses [`Policy`][gpc.models.Policy] and implements `sample`, which proposes
`num_samples` plans of `T` actions per batch element from the latest observations.

```python
import torch
from torch import Tensor

from gpc.models import Policy, PolicyConfig


class MyPolicy(Policy):
    def __init__(self, config: PolicyConfig, action_dim: int):
        super().__init__(config)
        self.action_dim = action_dim

    def sample(self, frames: Tensor, proprio: Tensor, *, num_samples: int = 1) -> Tensor:
        # frames: [B, K, C, H, W], proprio: [B, K, P] -> actions: [B, N, T, A]
        batch = frames.shape[0]
        return torch.randn(batch, num_samples, self.prediction_horizon, self.action_dim)


policy = MyPolicy(PolicyConfig(observation_horizon=2, prediction_horizon=8), action_dim=2)
```

`PolicyConfig` sets two horizons:

- `observation_horizon` (`K`): how many recent observations the policy conditions on.
- `prediction_horizon` (`T`): how many actions each sampled plan contains.

Optionally, `action_low` and `action_high` bound the action space, each as one value shared
by every action dimension or a list with one value per dimension. Controllers clamp the plans
they perturb or optimize to these bounds; leave them unset for an unbounded action space.

```python
PolicyConfig(observation_horizon=2, prediction_horizon=8, action_low=-1.0, action_high=1.0)
```

`sample` always receives exactly `observation_horizon` observations. Early in an
episode, when fewer have been observed, the controller pads the history by repeating
the earliest observation.

!!! tip "Sampling should be stochastic"

    The controller draws its candidate plans with `num_samples` and relies on the variety
    between them. A deterministic policy gives the controller nothing to choose from.

!!! tip "Encode once, sample many"

    `GPCRank` asks for `num_samples` plans (64 by default) per replanning step. Encode the
    observations once per batch element and draw every sample from that encoding, rather
    than repeating the observations `num_samples` times:

    ```python
    def sample(self, frames, proprio, *, num_samples=1):
        cond = self.encoder(frames, proprio)  # [B, D], computed once
        cond = cond.repeat_interleave(num_samples, dim=0)  # [B * N, D]
        noise = torch.randn(len(cond), self.prediction_horizon, self.action_dim)
        actions = self.denoise(noise, cond)  # [B * N, T, A]
        return actions.unflatten(0, (-1, num_samples))  # [B, N, T, A]
    ```

Next, build a [world model](world-model.md) to simulate these plans.
