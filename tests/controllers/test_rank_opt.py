import pytest
import torch
from torch import nn
from torch.testing import assert_close

from gpc import GPCRankOpt
from tests.fakes import A, B, make_obs, step

N, M, LR = 4, 3, 0.5


@pytest.fixture
def rank_opt(policy, world_model, objective) -> GPCRankOpt:
    return GPCRankOpt(
        policy, world_model, objective, num_samples=N, spread=1.0, num_steps=M, step_size=LR
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_samples": 0}, "num_samples"),
        ({"num_steps": -1}, "num_steps"),
        ({"step_size": 0.0}, "step_size"),
        ({"action_horizon": 0}, "action_horizon"),
    ],
)
def test_rejects_invalid_arguments(policy, world_model, objective, kwargs, match):
    with pytest.raises(ValueError, match=match):
        GPCRankOpt(policy, world_model, objective, **kwargs)


def test_propose_shape(rank_opt, policy):
    plans = rank_opt._propose(*make_obs(k=2))
    assert plans.shape == (B, N, policy.prediction_horizon, A)


def test_plan_optimizes_every_proposal_then_ranks(rank_opt, objective):
    frames, proprio = make_obs(k=2)
    goal = torch.zeros(B, 1)

    plan = rank_opt.plan(frames, proprio, goal)

    # M optimization calls over all N proposals, then one ranking call
    assert len(objective.calls) == M + 1
    assert all(call[2].shape[1] == N for call in objective.calls)
    _, seen_goal, ranked = objective.calls[-1]
    assert seen_goal is goal
    optimized = rank_opt._propose(frames, proprio) + M * LR
    assert_close(ranked, optimized)
    # FakePolicy assigns strictly increasing actions, so the last proposal scores highest
    assert_close(plan, optimized[:, -1])


def test_objective_gradient_includes_plans(policy, world_model):
    class Target(nn.Module):
        """Rewards closeness to zero, so optimization pulls every plan towards it."""

        def forward(self, futures, goal, *, plans):
            return -plans.square().sum(dim=(2, 3))

    rank_opt = GPCRankOpt(
        policy, world_model, Target(), num_samples=N, spread=1.0, num_steps=1, step_size=0.25
    )
    frames, proprio = make_obs(k=2)
    # One step on -|a|^2 with step 0.25 halves every action; the smallest proposal wins
    assert_close(rank_opt.plan(frames, proprio), rank_opt._propose(frames, proprio)[:, 0] / 2)


def test_zero_steps_matches_rank(policy, world_model, objective):
    rank_opt = GPCRankOpt(policy, world_model, objective, num_samples=N, num_steps=0)
    frames, proprio = make_obs(k=2)
    assert_close(rank_opt.plan(frames, proprio), rank_opt._propose(frames, proprio)[:, -1])
    assert len(objective.calls) == 1


def test_single_frame_skips_ranking_by_default(rank_opt, objective):
    frames, proprio = make_obs(k=1)
    plan = rank_opt.plan(frames, proprio)
    assert objective.calls == []
    assert_close(plan, rank_opt._propose(frames, proprio)[:, 0])


def test_rank_first_step_ranks_without_optimizing(policy, world_model, objective):
    rank_opt = GPCRankOpt(
        policy, world_model, objective, num_samples=N, num_steps=M, rank_first_step=True
    )
    rank_opt.plan(*make_obs(k=1))
    assert len(objective.calls) == 1


def test_optimize_first_step_optimizes_single_frame(policy, world_model, objective):
    rank_opt = GPCRankOpt(
        policy,
        world_model,
        objective,
        num_samples=N,
        num_steps=M,
        rank_first_step=True,
        optimize_first_step=True,
    )
    rank_opt.plan(*make_obs(k=1))
    assert len(objective.calls) == M + 1


def test_plan_does_not_track_gradients(rank_opt):
    frames, proprio = make_obs(k=2)
    assert not rank_opt.plan(frames.requires_grad_(), proprio).requires_grad


def test_act_end_to_end(policy, world_model, objective):
    rank_opt = GPCRankOpt(policy, world_model, objective, num_samples=N, action_horizon=2)
    actions = [step(rank_opt) for _ in range(3)]
    assert all(a.shape == (B, A) for a in actions)
