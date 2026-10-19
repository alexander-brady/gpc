"""Fixtures providing fake GPC components, shared across the suite."""

import pytest

from gpc.models import PolicyConfig, WorldModelConfig
from tests.fakes import FakePolicy, FakeWorldModel, SumObjective


@pytest.fixture
def policy() -> FakePolicy:
    return FakePolicy(PolicyConfig(observation_horizon=2, prediction_horizon=3))


@pytest.fixture
def world_model() -> FakeWorldModel:
    return FakeWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))


@pytest.fixture
def objective() -> SumObjective:
    return SumObjective()
