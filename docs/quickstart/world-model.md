# 2. World model

A world model subclasses [`WorldModel`][gpc.models.WorldModel] and implements two methods:

- `encode` maps frames `[B, F, C, H, W]` to latents `[B, F, D...]`.
- `predict` maps `K` context latents and `T` actions to the next `T` latents.

```python
import torch.nn as nn

from gpc.models import WorldModel, WorldModelConfig


class MyWorldModel(WorldModel):
    def __init__(self, config: WorldModelConfig, latent_dim: int, action_dim: int):
        super().__init__(config)
        self.encoder = nn.LazyLinear(latent_dim)
        self.dynamics = nn.Linear(latent_dim + action_dim, latent_dim)

    def encode(self, frames: Tensor) -> Tensor:
        return self.encoder(frames.flatten(2))  # [B, F, C, H, W] -> [B, F, D]

    def predict(self, history: Tensor, actions: Tensor) -> Tensor:
        state = history[:, -1]
        predicted = []
        for action in actions.unbind(dim=1):
            state = self.dynamics(torch.cat([state, action], dim=-1))
            predicted.append(state)
        return torch.stack(predicted, dim=1)  # [B, T, D]


world_model = MyWorldModel(
    WorldModelConfig(context_horizon=4, prediction_horizon=4), latent_dim=16, action_dim=2
)
```

`WorldModelConfig` sets `context_horizon` (`K`), the number of latents `predict` conditions
on, and `prediction_horizon` (`T`), the number of latents it predicts per call.

The base class also provides:

| Method | Default behaviour |
| ------ | ----------------- |
| `rollout(history, actions)` | Predicts any number of steps by calling `predict` on chunks of `prediction_horizon` actions, each conditioned on the latest `context_horizon` latents. |
| `decode(latents)` | Raises `NotImplementedError`. Override it to turn latents back into frames, e.g. for visualization. |

Next, define an [objective](objective.md) that scores the simulated futures.
