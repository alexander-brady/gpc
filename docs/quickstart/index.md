# Quickstart

This series builds a working GPC controller from scratch in four steps, then shows how to share it:

1. [Policy](policy.md): propose candidate action plans from recent observations.
2. [World model](world-model.md): simulate the future each plan leads to.
3. [Objective](objective.md): score those simulated futures against a goal.
4. [Controller](controller.md): wire the three together and run a control loop.
5. [Sharing on the Hub](hub.md): push and pull models and controllers.

Each step builds on the previous one, so the code snippets are meant to be run in order,
in a single script or notebook.

## Tensor conventions

All tensors are batch-first. The letters below are used throughout.

| Symbol | Meaning |
| ------ | ------- |
| `B` | Batch size (e.g. parallel environments) |
| `K` | Number of observed (context) timesteps |
| `T` | Number of planned or predicted timesteps |
| `N` | Number of candidate plans |
| `C, H, W` | Frame channels, height and width |
| `P` | Proprioception dimension (the agent's own low-dimensional state) |
| `A` | Action dimension |
| `D...` | Latent dimensions of the world model, one or more |

Ready? Start with the [policy](policy.md).
