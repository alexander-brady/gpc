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
    action_low: float | list[float] | None = None
    """Lower bound on actions, shared or per action dimension. ``None`` leaves them unbounded."""
    action_high: float | list[float] | None = None
    """Upper bound on actions, shared or per action dimension. ``None`` leaves them unbounded."""

    def __post_init__(self) -> None:
        low, high = self.action_low, self.action_high
        if low is None or high is None:
            return
        dims = next((len(b) for b in (low, high) if isinstance(b, list)), 1)
        lows = low if isinstance(low, list) else [low] * dims
        highs = high if isinstance(high, list) else [high] * dims
        if len(lows) != len(highs):
            raise ValueError("action_low and action_high must have the same length.")
        if any(lo > hi for lo, hi in zip(lows, highs, strict=True)):
            raise ValueError("action_low must not exceed action_high.")
