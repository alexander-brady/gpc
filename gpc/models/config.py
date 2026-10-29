from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Self

from shape_extensions import Int, IntVar


@dataclass(kw_only=True)
class Config:
    """Base configuration class for models.

    Saved to the Hub as JSON with [`to_dict`](gpc.models.config.Config.to_dict) and restored with
    [`from_dict`](gpc.models.config.Config.from_dict). Override both for fields that aren't plain
    JSON values, such as nested configs.
    """

    def to_dict(self) -> dict[str, Any]:
        """The config as a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        """Build the config from a dict, ignoring keys that aren't fields."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


@dataclass(kw_only=True)
class WorldModelConfig[K: IntVar, T: IntVar](Config):
    """Configuration class for world models."""

    context_horizon: Int[K]
    """Number of past timesteps to condition on when predicting future states."""
    prediction_horizon: Int[T]
    """Number of future timesteps to predict."""


@dataclass(kw_only=True)
class PolicyConfig[K: IntVar, T: IntVar](Config):
    """Configuration class for policies."""

    observation_horizon: Int[K]
    """Number of past timesteps to condition on when planning actions."""
    prediction_horizon: Int[T]
    """Number of future timesteps to plan actions for."""
