# GPC: A Framework for World-Model-Based Generative Control

[![PyPI](https://img.shields.io/pypi/v/gpclib)](https://pypi.org/project/gpclib/)
[![CI](https://img.shields.io/github/actions/workflow/status/alexander-brady/gpc/ci.yml?branch=main&label=CI)](https://github.com/alexander-brady/gpc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/gpclib)](https://pypi.org/project/gpclib/)
[![License: MIT](https://img.shields.io/github/license/alexander-brady/gpc)](https://github.com/alexander-brady/gpc/blob/main/LICENSE)

**GPC** is a framework for world-model-based Generative Predictive Control.

A GPC controller combines three components:

- a generative **policy** that proposes action plans from recent observations,
- a **world model** that simulates the future each plan would lead to, and
- an **objective** that scores those simulated futures against a goal.

At each replanning step, the controller picks the plan whose predicted future scores best,
then executes it in the environment.

## Installation

GPC requires Python 3.12 or newer. It is lightweight, depending only on PyTorch and the
Hugging Face Hub client, so it drops into existing training and robotics stacks easily.

```bash
uv add gpclib    # or: pip install gpclib
```

## Quick start

Implement a [`Policy`](https://alexander-brady.github.io/gpc/quickstart/policy/), a [`WorldModel`](https://alexander-brady.github.io/gpc/quickstart/world-model/)
and an [`Objective`](https://alexander-brady.github.io/gpc/quickstart/objective/), then wire them into a controller:

```python
from gpc import GPCRank

controller = GPCRank(policy, world_model, objective, action_horizon=4, num_samples=32)

controller.reset()
for step in range(100):
    action = controller.act(frames, proprio=proprio, goal=goal)  # [B, A]
```

See the [quickstart](https://alexander-brady.github.io/gpc/quickstart/) for full examples and the
[API reference](https://alexander-brady.github.io/gpc/api/gpc/) for details.

## Controllers

Each controller combines the three components with a different planning strategy:

| Controller | Strategy |
| ---------- | -------- |
| `GPCRank` | Sample many plans from the policy and keep the best-scoring one. |
| `GPCOpt` | Refine a sampled plan by gradient ascent on the objective through the world model. |
| `GPCRankOpt` | Refine many sampled plans by gradient ascent, then keep the best-scoring one. |

See [Controller](https://alexander-brady.github.io/gpc/quickstart/controller/) for when to use each and their hyperparameters.

## Sharing on the Hub

Policies, world models and whole controllers can be saved with `save_pretrained`, pushed to
the [Hugging Face Hub](https://huggingface.co/docs/hub) with `push_to_hub`, and loaded back
with `from_pretrained`:

```python
controller.push_to_hub("me/push-t")

controller = GPCRank.from_pretrained("me/push-t")  # any controller can load the repo
```

See [Sharing on the Hub](https://alexander-brady.github.io/gpc/quickstart/hub/) for details.

## Development

```bash
just test       # run the test suite with coverage
just validate   # format check, lint and type check
just docs       # serve the docs locally
```

See [CONTRIBUTING.md](https://github.com/alexander-brady/gpc/blob/main/CONTRIBUTING.md) for CI checks, testing, tensor shape annotations and documentation guidelines.
