import functools

import pytest
import torch
from torch import nn, optim
from torch.testing import assert_close

from gpc import GPCOpt
from gpc.models import PolicyConfig, WorldModelConfig
from gpc.objectives import frame_objective
from tests.fakes import (
    A,
    B,
    C,
    DecodingWorldModel,
    FakePolicy,
    H,
    SumObjective,
    W,
    make_obs,
    step,
)

M, LR = 3, 0.5


@pytest.fixture
def opt(policy, world_model, objective) -> GPCOpt:
    return GPCOpt(policy, world_model, objective, num_steps=M, step_size=LR)


@pytest.mark.parametrize("num_steps", [-1, -5])
def test_rejects_negative_num_steps(policy, world_model, objective, num_steps):
    with pytest.raises(ValueError, match="num_steps"):
        GPCOpt(policy, world_model, objective, num_steps=num_steps)


@pytest.mark.parametrize("step_size", [0.0, -0.1])
def test_rejects_non_positive_step_size(policy, world_model, objective, step_size):
    with pytest.raises(ValueError, match="step_size"):
        GPCOpt(policy, world_model, objective, step_size=step_size)


def test_propose_shape(opt, policy):
    plan = opt._propose(*make_obs(k=2))
    assert plan.shape == (B, 1, policy.prediction_horizon, A)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_policy_receives_its_observation_horizon(opt, policy, k):
    opt.plan(*make_obs(k=k))
    frames, proprio = policy.calls[-1]
    assert frames.shape[:2] == proprio.shape[:2] == (B, policy.observation_horizon)


def test_plan_ascends_the_objective(opt):
    frames, proprio = make_obs(k=2)
    plan = opt.plan(frames, proprio)
    # SumObjective over FakeWorldModel sums the plan, so every action's gradient is 1
    assert_close(plan, opt._propose(frames, proprio)[:, 0] + M * LR)


def test_objective_called_once_per_step(opt, objective):
    goal = torch.zeros(B, 1)
    opt.plan(*make_obs(k=2), goal)
    assert len(objective.calls) == M
    futures, seen_goal, seen_plans = objective.calls[0]
    assert seen_goal is goal
    assert futures.shape == (B, 1, opt.policy.prediction_horizon, 1)
    assert seen_plans.shape == (B, 1, opt.policy.prediction_horizon, A)


def test_zero_steps_returns_proposal(policy, world_model, objective):
    opt = GPCOpt(policy, world_model, objective, num_steps=0)
    frames, proprio = make_obs(k=2)
    assert_close(opt.plan(frames, proprio), opt._propose(frames, proprio)[:, 0])
    assert objective.calls == []


def test_optimize_refines_each_candidate_independently(opt):
    frames, _ = make_obs(k=2)
    plans = torch.randn(B, 4, opt.policy.prediction_horizon, A)
    assert_close(opt.optimize(frames, plans), plans + M * LR)


def test_optimize_does_not_modify_input(opt):
    frames, _ = make_obs(k=2)
    plans = torch.zeros(B, 1, opt.policy.prediction_horizon, A)
    opt.optimize(frames, plans)
    assert_close(plans, torch.zeros_like(plans))


def test_plan_objective_gradient_includes_plans(policy, world_model):
    class Effort(nn.Module):
        def forward(self, futures, goal, *, plans):
            return -plans.square().sum(dim=(2, 3))

    opt = GPCOpt(policy, world_model, Effort(), num_steps=1, step_size=0.25)
    frames, proprio = make_obs(k=2)
    # One step on -|a|^2 with step 0.25 halves every action
    assert_close(opt.plan(frames, proprio), opt._propose(frames, proprio)[:, 0] / 2)


def test_frame_objective_differentiates_through_decode(policy):
    world_model = DecodingWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    objective = frame_objective(SumObjective())
    opt = GPCOpt(policy, world_model, objective, num_steps=1, step_size=LR)
    frames, proprio = make_obs(k=2)

    plan = opt.plan(frames, proprio)

    assert objective.calls[0][0].shape[3:] == (C, H, W)
    # Each latent is broadcast to C*H*W pixels
    assert_close(plan, opt._propose(frames, proprio)[:, 0] + LR * C * H * W)


def test_optimize_projects_onto_action_bounds(world_model, objective):
    config = PolicyConfig(observation_horizon=2, prediction_horizon=3, action_high=4.0)
    opt = GPCOpt(FakePolicy(config), world_model, objective, num_steps=M, step_size=LR)
    frames, proprio = make_obs(k=2)
    # SumObjective's gradient is 1 per action, so each step adds LR until the bound is hit
    expected = (opt._propose(frames, proprio)[:, 0] + M * LR).clamp(max=4.0)
    assert_close(opt.plan(frames, proprio), expected)


def test_custom_optimizer(policy, world_model, objective):
    adam = functools.partial(optim.Adam, betas=(0.0, 0.999))
    opt = GPCOpt(policy, world_model, objective, num_steps=1, step_size=LR, optimizer=adam)
    frames, proprio = make_obs(k=2)
    # Adam's first step moves each parameter by ~lr in the gradient's sign
    assert_close(opt.plan(frames, proprio), opt._propose(frames, proprio)[:, 0] + LR)


def test_world_model_parameters_receive_no_gradients(policy, world_model, objective):
    world_model.scale = nn.Parameter(torch.ones(()))
    predict = world_model.predict
    world_model.predict = lambda h, a: predict(h, a) * world_model.scale  # pyrefly: ignore
    GPCOpt(policy, world_model, objective, num_steps=M).plan(*make_obs(k=2))
    assert world_model.scale.grad is None


def test_parameters_frozen_during_optimization_and_restored(policy, world_model, objective):
    world_model.scale = nn.Parameter(torch.ones(()))
    world_model.fixed = nn.Parameter(torch.ones(()), requires_grad=False)
    seen = []
    predict = world_model.predict

    def recording_predict(history, actions):
        seen.append(world_model.scale.requires_grad)
        return predict(history, actions) * world_model.scale

    world_model.predict = recording_predict  # pyrefly: ignore
    GPCOpt(policy, world_model, objective, num_steps=M).plan(*make_obs(k=2))

    assert seen and not any(seen)
    assert world_model.scale.requires_grad
    assert not world_model.fixed.requires_grad
    assert all(p.requires_grad for p in policy.parameters())


def test_parameters_restored_when_optimization_raises(policy, world_model):
    def failing(_futures, _goal):
        raise RuntimeError("boom")

    opt = GPCOpt(policy, world_model, failing, num_steps=M)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(RuntimeError, match="boom"):
        opt.plan(*make_obs(k=2))
    assert all(p.requires_grad for p in opt.parameters())


def test_single_frame_skips_optimization_by_default(opt, objective):
    frames, proprio = make_obs(k=1)
    plan = opt.plan(frames, proprio)
    assert objective.calls == []
    assert_close(plan, opt._propose(frames, proprio)[:, 0])


def test_optimize_first_step_optimizes_single_frame(policy, world_model, objective):
    opt = GPCOpt(policy, world_model, objective, num_steps=M, optimize_first_step=True)
    opt.plan(*make_obs(k=1))
    assert len(objective.calls) == M


def test_plan_does_not_track_gradients(opt):
    frames, proprio = make_obs(k=2)
    assert not opt.plan(frames.requires_grad_(), proprio).requires_grad


def test_act_end_to_end(policy, world_model, objective):
    opt = GPCOpt(policy, world_model, objective, num_steps=M, action_horizon=2)
    actions = [step(opt) for _ in range(3)]
    assert all(a.shape == (B, A) for a in actions)
