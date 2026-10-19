import pytest
import torch
from torch.testing import assert_close

from gpc import GPCRank
from tests.fakes import A, B, make_obs, step

N = 4


@pytest.fixture
def rank(policy, world_model, objective) -> GPCRank:
    return GPCRank(policy, world_model, objective, num_samples=N, spread=1.0)


@pytest.mark.parametrize("num_samples", [0, -3])
def test_rejects_non_positive_num_samples(policy, world_model, objective, num_samples):
    with pytest.raises(ValueError, match="num_samples"):
        GPCRank(policy, world_model, objective, num_samples=num_samples)


def test_rejects_non_positive_action_horizon(policy, world_model, objective):
    with pytest.raises(ValueError, match="action_horizon"):
        GPCRank(policy, world_model, objective, action_horizon=0)


def test_propose_shape(rank, policy):
    plans = rank._propose(*make_obs(k=2))
    assert plans.shape == (B, N, policy.prediction_horizon, A)


@pytest.mark.parametrize("spread", [0.0, 1.0, 2.5])
def test_propose_scales_deviation_from_mean_by_spread(policy, world_model, objective, spread):
    frames, proprio = make_obs(k=2)
    base = GPCRank(policy, world_model, objective, num_samples=N, spread=1.0)
    spread_rank = GPCRank(policy, world_model, objective, num_samples=N, spread=spread)

    raw = base._propose(frames, proprio)
    mean = raw.mean(dim=1, keepdim=True)

    out = spread_rank._propose(frames, proprio)
    assert_close(out, mean + spread * (raw - mean))
    assert_close(out.mean(dim=1, keepdim=True), mean)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_policy_receives_its_observation_horizon(rank, policy, k):
    rank.plan(*make_obs(k=k))
    frames, proprio = policy.calls[-1]
    assert frames.shape[:2] == proprio.shape[:2] == (B * N, policy.observation_horizon)


def test_plan_picks_highest_scoring_proposal(rank, objective):
    frames, proprio = make_obs(k=2)
    goal = torch.zeros(B, 1)

    plan = rank.plan(frames, proprio, goal)

    _, plans, seen_goal = objective.calls[0]
    assert seen_goal is goal
    best = objective(frames, plans, goal).argmax(dim=1)
    assert_close(plan, plans[torch.arange(B), best])
    # FakePolicy assigns strictly increasing actions, so the last proposal scores highest
    assert_close(plan, plans[:, -1])


def test_plan_ties_resolve_to_first_maximum(policy, world_model):
    tie = lambda f, p, g: torch.zeros(p.shape[:2])  # noqa: E731
    rank = GPCRank(policy, world_model, tie, num_samples=N)  # pyrefly: ignore[bad-argument-type]
    frames, proprio = make_obs(k=2)
    assert_close(rank.plan(frames, proprio), rank._propose(frames, proprio)[:, 0])


def test_single_frame_skips_ranking_by_default(rank, objective):
    frames, proprio = make_obs(k=1)
    plan = rank.plan(frames, proprio)

    assert objective.calls == []
    assert_close(plan, rank._propose(frames, proprio)[:, 0])


def test_rank_first_step_ranks_single_frame(policy, world_model, objective):
    rank = GPCRank(policy, world_model, objective, num_samples=N, rank_first_step=True)
    rank.plan(*make_obs(k=1))
    assert len(objective.calls) == 1


def test_plan_does_not_track_gradients(rank):
    frames, proprio = make_obs(k=2)
    assert not rank.plan(frames.requires_grad_(), proprio).requires_grad


def test_act_end_to_end(policy, world_model, objective):
    rank = GPCRank(policy, world_model, objective, num_samples=N, action_horizon=2)
    actions = [step(rank) for _ in range(3)]
    assert all(a.shape == (B, A) for a in actions)
