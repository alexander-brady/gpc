# GPC

**GPC** is a framework for world-model-based Generative Predictive Control.

A GPC controller combines three components:

- a generative **policy** that proposes action plans from recent observations,
- a **world model** that simulates the future each plan would lead to, and
- an **objective** that scores those simulated futures against a goal.

The controller uses them to pick, at each replanning step, the plan whose
predicted future scores best, then executes it in the environment.

## Installation

GPC requires Python 3.12 or newer.

=== "uv"

    ```bash
    uv add gpc
    ```

=== "pip"

    ```bash
    pip install gpc
    ```

## Next steps

Follow the [Quickstart](quickstart/index.md) to implement the three components and run a
controller, or browse the [API reference](api/gpc/index.md).
