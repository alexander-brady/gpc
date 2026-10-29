from dataclasses import is_dataclass

import pytest

from gpc.models import PolicyConfig, WorldModelConfig

CONFIG_CLASSES = [PolicyConfig, WorldModelConfig]


@pytest.mark.parametrize("cls", CONFIG_CLASSES)
def test_config_classes_are_dataclasses(cls):
    assert is_dataclass(cls)


@pytest.mark.parametrize("cls", CONFIG_CLASSES)
def test_configs_are_keyword_only(cls):
    with pytest.raises(TypeError):
        cls(1, 2)


def test_policy_action_bounds_round_trip():
    config = PolicyConfig(
        observation_horizon=1, prediction_horizon=1, action_low=[-1.0, 0.0], action_high=1.0
    )
    assert PolicyConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("low", "high"),
    [(1.0, 0.0), ([0.0, 2.0], [1.0, 1.0]), (2.0, [3.0, 1.0]), ([0.0, 3.0], 2.0)],
)
def test_policy_rejects_low_above_high(low, high):
    with pytest.raises(ValueError, match="must not exceed"):
        PolicyConfig(observation_horizon=1, prediction_horizon=1, action_low=low, action_high=high)


def test_policy_rejects_mismatched_bound_lengths():
    with pytest.raises(ValueError, match="same length"):
        PolicyConfig(
            observation_horizon=1,
            prediction_horizon=1,
            action_low=[0.0, 0.0],
            action_high=[1.0, 1.0, 1.0],
        )


@pytest.mark.parametrize(
    ("low", "high"),
    [(None, None), (0.0, None), (None, [1.0]), (-1.0, 1.0), (0.0, 0.0), ([0.0, -1.0], 1.0)],
)
def test_policy_accepts_valid_bounds(low, high):
    PolicyConfig(observation_horizon=1, prediction_horizon=1, action_low=low, action_high=high)
