# 4. Controller

A controller decides how the policy, world model and objective work together to pick a
plan. GPC defines a few strategies:

| Controller | Strategy | Status |
| ---------- | -------- | ------ |
| [`GPCRank`][gpc.GPCRank] | Sample many plans from the policy and keep the best-scoring one. | Available |
| [`GPCOpt`][gpc.GPCOpt] | Refine a sampled plan by gradient ascent on the objective through the world model. | Available |
| `GPCRankOpt` | Rank sampled plans, then refine the best one. | Planned |

This guide uses `GPCRank`, the simplest of the three.

[`GPCRank`][gpc.GPCRank] samples `num_samples` plans from the policy, scores them with the
objective, and keeps the best one.

```python
from gpc import GPCRank

controller = GPCRank(
    policy,
    world_model,
    objective,
    action_horizon=4,
    num_samples=32,
)
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `action_horizon` | `1` | Actions to execute from each plan before replanning. |
| `num_samples` | `64` | Candidate plans sampled per replanning step. |
| `spread` | `1.01` | Scales each candidate's distance from the candidates' mean. Values above `1` increase diversity; `1` leaves the samples unchanged. |
| `rank_first_step` | `False` | Whether to rank when only one frame has been observed. If `False`, the first sampled plan is used. |

### Optimizing a plan

[`GPCOpt`][gpc.GPCOpt] samples a single plan from the policy as a warm start, then refines
it with `num_steps` gradient steps that maximize the objective, differentiating through the
world model (and its decoder, for a frame objective). It suits objectives that are smooth
numerical rewards; [`GPCRank`][gpc.GPCRank] also works with non-differentiable ones.

```python
from gpc import GPCOpt

controller = GPCOpt(
    policy,
    world_model,
    objective,
    action_horizon=4,
    num_steps=25,
    step_size=5e-3,
)
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `action_horizon` | `1` | Actions to execute from each plan before replanning. |
| `num_steps` | `25` | Gradient steps per replanning step. `0` returns the policy's plan unchanged. |
| `step_size` | `5e-3` | Learning rate passed to the optimizer. |
| `optimizer` | `torch.optim.SGD` | Optimizer factory, called as `optimizer([plan], lr=step_size)`. Use e.g. `torch.optim.Adam`, or a `functools.partial` for other settings. |
| `optimize_first_step` | `False` | Whether to optimize when only one frame has been observed. If `False`, the sampled plan is used. |

!!! warning "Make the world model deterministic"

    The gradients are only as reliable as the world model. Freeze any sampling noise at
    inference time (e.g. fix a diffusion world model's initial noise), or the noisy gradients
    destabilize the optimization.

!!! note "Action horizon vs. prediction horizon"

    The controller's `action_horizon` is how many actions are *executed* before replanning.
    The policy's `prediction_horizon` is how many actions each plan *contains*. If the
    action horizon is longer, the controller replans as soon as the plan runs out.

## Control loop

`act` consumes one observation per step and returns one action. It keeps the observation
history and the current plan internally, so call `reset` at the start of every episode.

Here it drives a vectorized [Gymnasium](https://gymnasium.farama.org/)-style environment
whose observations hold an image and a proprioceptive state:

```python
import gymnasium as gym


def to_tensors(obs: dict) -> tuple[Tensor, Tensor]:
    """Converts a batch of observations to frames [B, 1, C, H, W] and proprio [B, 1, P]."""
    frames = torch.as_tensor(obs["pixels"]).permute(0, 3, 1, 2).float() / 255  # from [B, H, W, C]
    proprio = torch.as_tensor(obs["proprio"]).float()  # [B, P]
    return frames[:, None], proprio[:, None]


batch = 3
envs = gym.make_vec("YourEnv-v0", num_envs=batch)  # stand-in for your environment
goal = torch.zeros(batch, 16)  # target latent, e.g. the encoding of a goal image

obs, info = envs.reset(seed=0)
controller.reset()
for step in range(100):
    frames, proprio = to_tensors(obs)
    action = controller.act(frames, proprio=proprio, goal=goal)  # [B, A]
    obs, reward, terminated, truncated, info = envs.step(action.numpy())
    if (terminated | truncated).any():
        break
```

The controller keeps one history for the whole batch, so the environments run their
episodes in lockstep: stop (or `reset` the controller) as soon as any of them finishes.

## Planning without state

To plan once from an explicit observation history, call `plan` instead. It does not read or
modify the controller's internal state, and returns the whole plan.

```python
frames = torch.rand(batch, 4, 3, 64, 64)  # [B, K, C, H, W]
proprio = torch.rand(batch, 4, 5)  # [B, K, P]
plan = controller.plan(frames, proprio, goal)  # [B, T, A]
```

## Next steps

See the [API reference](../api/gpc/index.md) for every class and option used above.
