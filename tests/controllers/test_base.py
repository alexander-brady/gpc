import pytest
import torch

from gpc.controllers.base import GPC
from tests.fakes import A, B, StubGPC, make_obs, step


@pytest.fixture
def gpc(policy, world_model, objective) -> StubGPC:
    return StubGPC(policy, world_model, objective, action_horizon=2)


def test_base_is_abstract(policy, world_model, objective):
    with pytest.raises(TypeError):
        GPC(policy, world_model, objective)  # pyrefly: ignore


@pytest.mark.parametrize("action_horizon", [0, -1])
def test_rejects_non_positive_action_horizon(policy, world_model, objective, action_horizon):
    with pytest.raises(ValueError, match="action_horizon"):
        StubGPC(policy, world_model, objective, action_horizon=action_horizon)


def test_history_size_is_max_of_model_horizons(gpc):
    # policy observation_horizon=2, world model context_horizon=3
    assert gpc.history_size == 3


def test_act_returns_one_action_per_batch(gpc):
    assert step(gpc).shape == (B, A)


def test_act_replans_every_action_horizon_steps(gpc):
    actions = [step(gpc)[0, 0].item() for _ in range(5)]

    # plan n yields steps [100n, 100n+1, ...]; only the first action_horizon=2 are executed
    assert actions == [100, 101, 200, 201, 300]
    assert len(gpc.plan_calls) == 3


def test_action_horizon_longer_than_plan_replans_when_plan_exhausted(
    policy, world_model, objective
):
    gpc = StubGPC(policy, world_model, objective, action_horizon=10)
    for _ in range(StubGPC.plan_len + 1):
        step(gpc)
    assert len(gpc.plan_calls) == 2


def test_plan_receives_accumulated_history_up_to_history_size(gpc):
    gpc.action_horizon = 1  # replan every step so each history is observed
    for t in range(5):
        step(gpc, fill=float(t))

    lengths = [f.shape[1] for f, _, _ in gpc.plan_calls]
    assert lengths == [1, 2, 3, 3, 3]
    last_frames, last_proprio, _ = gpc.plan_calls[-1]
    assert last_frames[0, :, 0, 0, 0].tolist() == [2, 3, 4]  # oldest dropped first
    assert last_proprio[0, :, 0].tolist() == [2, 3, 4]


def test_act_accepts_multiple_frames_per_call(gpc):
    frames, proprio = make_obs(k=2)
    gpc.act(frames, proprio=proprio)
    assert gpc.plan_calls[0][0].shape[1] == 2


def test_goal_is_forwarded_to_plan(gpc):
    goal = torch.ones(B, 7)
    step(gpc, goal=goal)
    assert gpc.plan_calls[0][2] is goal


def test_reset_clears_history_and_pending_plan(gpc):
    for _ in range(3):
        step(gpc)
    gpc.reset()
    step(gpc)

    assert len(gpc.plan_calls) == 3  # replanned immediately after reset
    assert gpc.plan_calls[-1][0].shape[1] == 1  # with fresh history
