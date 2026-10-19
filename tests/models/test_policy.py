import pytest

from gpc.models import Policy, PolicyConfig


def test_exposes_config_horizons(policy):
    assert (policy.observation_horizon, policy.prediction_horizon) == (2, 3)


def test_policies_are_abstract():
    with pytest.raises(TypeError):
        Policy(PolicyConfig(observation_horizon=1, prediction_horizon=1))  # pyrefly: ignore
