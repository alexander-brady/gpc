"""World-model-based Generative Predictive Control.

- [`gpc.controllers`][gpc.controllers]: controllers that plan and act.
- [`gpc.models`][gpc.models]: policy and world model base classes and their configs.
- [`gpc.objectives`][gpc.objectives]: the protocol objectives implement to score plans.
- [`gpc.trajectory`][gpc.trajectory]: helpers for time-indexed tensors.
"""

from .controllers import GPCOpt, GPCRank, GPCRankOpt

__all__ = ["GPCOpt", "GPCRank", "GPCRankOpt"]
