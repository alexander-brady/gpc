# 3. Objective

An objective is any callable matching the [`Objective`][gpc.objectives.Objective] protocol.
It receives the observed frames, the candidate plans and an optional goal, and returns one
score per plan. **Higher scores are better.**

The objective is responsible for simulating each plan, so it typically holds a reference
to the world model:

```python
from gpc.trajectory import fit_horizon


class GoalDistance:
    """Scores each plan by how close its final predicted latent is to the goal."""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def __call__(self, frames: Tensor, plans: Tensor, goal: Tensor | None) -> Tensor:
        # frames: [B, K, C, H, W], plans: [B, N, T, A], goal: [B, D] -> scores: [B, N]
        assert goal is not None, "GoalDistance requires a goal"
        batch, num_plans = plans.shape[:2]

        frames = fit_horizon(frames, self.world_model.context_horizon)  # [B, K, C, H, W]
        history = self.world_model.encode(frames)  # [B, K, D]
        history = history.repeat_interleave(num_plans, dim=0)  # [B * N, K, D]
        actions = plans.flatten(0, 1)  # [B * N, T, A]

        future = self.world_model.rollout(history, actions)  # [B * N, T, D]
        final = future[:, -1].unflatten(0, (batch, num_plans))  # [B, N, D]
        return -torch.linalg.vector_norm(final - goal[:, None], dim=-1)


objective = GoalDistance(world_model)
```

The objective receives the controller's full observation history, which may be shorter
than the world model's `context_horizon` early in an episode.
[`fit_horizon`][gpc.trajectory.fit_horizon] trims or pads it to exactly that many steps,
repeating the earliest frame as padding, just as the controller does for the policy.

With all three components in place, wire them into a [controller](controller.md).
