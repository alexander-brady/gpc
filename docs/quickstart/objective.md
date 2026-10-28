# 3. Objective

An [`Objective`][gpc.objectives.Objective] is any callable that scores futures: either a
[`FutureObjective`][gpc.objectives.FutureObjective], or a
[`PlanObjective`][gpc.objectives.PlanObjective] that also takes the plans. The controller encodes the observed frames, rolls out every candidate plan with the world
model, and passes the predicted futures to the objective along with an optional goal. The
objective returns one score per plan. **Higher scores are better.**

By default, futures arrive in the world model's latent space:

```python
class GoalDistance:
    """Scores each plan by how close its final predicted latent is to the goal."""

    def __call__(self, futures: Tensor, goal: Tensor | None) -> Tensor:
        # futures: [B, N, T, D], goal: [B, D] -> scores: [B, N]
        assert goal is not None, "GoalDistance requires a goal"
        final = futures[:, :, -1]  # [B, N, D]
        return -torch.linalg.vector_norm(final - goal[:, None], dim=-1)


objective = GoalDistance()
```

Because the objective sees the whole predicted trajectory, it can also score the path
taken, for example by summing a per-step reward or penalizing unsafe intermediate states.

## Action costs

To penalize the actions themselves, write a [`PlanObjective`][gpc.objectives.PlanObjective]
by adding a keyword `plans` argument (or `**kwargs`). The controller sees it and passes the
candidate plans `[B, N, T, A]` that produced the futures. A `FutureObjective` like
`GoalDistance` is called with just the futures and goal. For an `nn.Module` objective, the
controller checks the signature of `forward`.

```python
class EffortPenalized(GoalDistance):
    """GoalDistance minus a penalty on the squared action magnitude."""

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    def __call__(self, futures: Tensor, goal: Tensor | None, *, plans: Tensor) -> Tensor:
        effort = plans.square().sum(dim=(2, 3))
        return super().__call__(futures, goal) - self.weight * effort
```

## Scoring frames

If your objective needs pixels, mark it with
[`frame_objective`][gpc.objectives.frame_objective]. The controller then decodes the futures
into frames `[B, N, T, C, H, W]` before scoring. This works on both functions and classes,
and requires a world model that implements `decode`:

```python
from gpc.objectives import frame_objective


@frame_objective
def brightness(futures: Tensor, goal: Tensor | None) -> Tensor:
    return futures[:, :, -1].mean(dim=(2, 3, 4))
```

Setting a `frame_objective = True` attribute on a class has the same effect. Objectives
score latents by default; [`latent_objective`][gpc.objectives.latent_objective] (or
`frame_objective = False`) states that explicitly.

## Shape checking

`Objective[Ds]` is generic over the latent shape `Ds`, and a frame objective is an
`Objective[[C, H, W]]`.
If you annotate your objective's shapes, the type checker catches an objective written for
the wrong representation:

```python
def latent_distance(
    futures: Tensor[[int, int, int, int]],  # [B, N, T, D] for a world model with D = [int]
    goal: Tensor[[int, int]] | None,
) -> Tensor[[int, int]]: ...


GPCRank(policy, world_model, latent_distance)  # ok: matches the world model's latents
frame_objective(latent_distance)  # error: typed for latents, not frames
```

The controller also rejects an objective whose futures match neither the world model's
latent shape nor frames. The `frame_objective` marker itself is only checked at runtime,
so an unmarked objective typed for frames still type-checks, then receives latents.

Pyrefly currently checks concrete shapes like these, but not objectives that are generic in
the batch or plan dimensions.

With all three components in place, wire them into a [controller](controller.md).
