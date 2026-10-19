from __future__ import annotations

from dataclasses import dataclass

from shape_extensions import Int, IntVar


@dataclass(kw_only=True)
class Config:
    """Base configuration class for models."""

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls = dataclass(kw_only=True)(cls)


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
