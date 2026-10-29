import pytest
import torch
from torch.testing import assert_close

from gpc.models import Policy, PolicyConfig
from tests.fakes import FakePolicy


def test_exposes_config_horizons(policy):
    assert (policy.observation_horizon, policy.prediction_horizon) == (2, 3)


def test_policies_are_abstract():
    with pytest.raises(TypeError):
        Policy(PolicyConfig(observation_horizon=1, prediction_horizon=1))  # pyrefly: ignore


def make_policy(**bounds) -> FakePolicy:
    return FakePolicy(PolicyConfig(observation_horizon=1, prediction_horizon=1, **bounds))


def test_clamp_actions_unbounded_returns_input():
    actions = torch.randn(2, 3, 2)
    assert make_policy().clamp_actions(actions) is actions


def test_clamp_actions_shared_bounds():
    actions = torch.tensor([[-2.0, 0.5], [3.0, -0.5]])
    clamped = make_policy(action_low=-1.0, action_high=1.0).clamp_actions(actions)
    assert_close(clamped, torch.tensor([[-1.0, 0.5], [1.0, -0.5]]))


def test_clamp_actions_per_dimension_bounds():
    actions = torch.full((2, 3, 2), 5.0)
    policy = make_policy(action_low=[0.0, -1.0], action_high=[1.0, 2.0])
    assert_close(policy.clamp_actions(actions), torch.tensor([1.0, 2.0]).expand(2, 3, 2))


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [({"action_low": 0.0}, [0.0, 3.0]), ({"action_high": 0.0}, [-3.0, 0.0])],
)
def test_clamp_actions_one_sided_bounds(bounds, expected):
    actions = torch.tensor([-3.0, 3.0])
    assert_close(make_policy(**bounds).clamp_actions(actions), torch.tensor(expected))
