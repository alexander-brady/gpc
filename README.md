# GPC: A Framework for World-Model-Based Generative Control

**GPC** is a framework for world-model-based Generative Predictive Control.

A GPC controller combines three components:

- a generative **policy** that proposes action plans from recent observations,
- a **world model** that simulates the future each plan would lead to, and
- an **objective** that scores those simulated futures against a goal.

At each replanning step, the controller picks the plan whose predicted future scores best,
then executes it in the environment.

## Installation

GPC requires Python 3.12 or newer.

```bash
uv add gpclib    # or: pip install gpclib
```

## Quick start

Implement a [`Policy`](docs/quickstart/policy.md), a [`WorldModel`](docs/quickstart/world-model.md)
and an [`Objective`](docs/quickstart/objective.md), then wire them into a controller:

```python
from gpc import GPCRank

controller = GPCRank(policy, world_model, objective, action_horizon=4, num_samples=32)

controller.reset()
for step in range(100):
    action = controller.act(frames, proprio=proprio, goal=goal)  # [B, A]
```

See the [quickstart](docs/quickstart/index.md) for full examples and the
[API reference](https://alexander-brady.github.io/gpc/api/gpc/) for details.

## Development

```bash
just test       # run the test suite with coverage
just validate   # format check, lint and type check
just docs       # serve the docs locally
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for CI checks, testing, tensor shape annotations and documentation guidelines.
