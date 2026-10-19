import pytest
import torch
from torch.testing import assert_close

from gpc.models import WorldModel, WorldModelConfig
from tests.fakes import FakeWorldModel


def test_exposes_config_horizons(world_model):
    assert (world_model.context_horizon, world_model.prediction_horizon) == (3, 2)


def test_world_models_are_abstract():
    with pytest.raises(TypeError):
        WorldModel(WorldModelConfig(context_horizon=1, prediction_horizon=1))  # pyrefly: ignore


def test_decode_not_implemented_by_default(world_model):
    with pytest.raises(NotImplementedError):
        world_model.decode(torch.zeros(1, 1, 1))


@pytest.mark.parametrize("num_steps", [1, 2, 5, 6])
def test_rollout_returns_one_prediction_per_action(world_model, num_steps):
    history = torch.zeros(2, 3, 1)
    actions = torch.arange(num_steps, dtype=torch.float).view(1, -1, 1).expand(2, -1, 1)

    out = world_model.rollout(history, actions)

    assert out.shape == (2, num_steps, 1)
    assert_close(out, actions)  # fake predict echoes actions


def test_rollout_predicts_in_chunks_of_prediction_horizon(world_model):
    world_model.rollout(torch.zeros(1, 3, 1), torch.zeros(1, 5, 1))
    assert len(world_model.contexts) == 3  # chunks of 2, 2, 1


def test_rollout_conditions_on_latest_context_window():
    wm = FakeWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    history = torch.tensor([-3.0, -2.0, -1.0]).view(1, 3, 1)
    actions = torch.arange(1.0, 7.0).view(1, 6, 1)

    wm.rollout(history, actions)

    seen = [ctx.flatten().tolist() for ctx in wm.contexts]
    assert seen == [[-3, -2, -1], [-1, 1, 2], [2, 3, 4]]


def test_rollout_pads_short_history_to_context_horizon():
    wm = FakeWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    history = torch.tensor([-1.0]).view(1, 1, 1)

    wm.rollout(history, torch.tensor([1.0, 2.0]).view(1, 2, 1))

    assert wm.contexts[0].flatten().tolist() == [-1, -1, -1]
