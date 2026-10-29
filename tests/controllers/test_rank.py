import pytest
import torch
from torch import nn
from torch.testing import assert_close

from gpc import GPCRank
from gpc.models import WorldModelConfig
from gpc.objectives import frame_objective, latent_objective
from tests.fakes import A, B, C, DecodingWorldModel, H, SumObjective, W, make_obs, step

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

    futures, seen_goal, seen_plans = objective.calls[0]
    assert seen_goal is goal
    plans = rank._propose(frames, proprio)
    assert_close(seen_plans, plans)
    # FakeWorldModel predicts each step's action sum
    assert_close(futures, plans.sum(dim=-1, keepdim=True))
    # FakePolicy assigns strictly increasing actions, so the last proposal scores highest
    assert_close(plan, plans[:, -1])


def test_plan_ties_resolve_to_first_maximum(policy, world_model):
    tie = lambda f, g: torch.zeros(f.shape[:2])  # noqa: E731
    rank = GPCRank(policy, world_model, tie, num_samples=N)  # pyrefly: ignore[bad-argument-type]
    frames, proprio = make_obs(k=2)
    assert_close(rank.plan(frames, proprio), rank._propose(frames, proprio)[:, 0])


def test_plan_never_picks_nan_score(policy, world_model):
    def nan_last(futures, goal):
        scores = futures.flatten(2).sum(dim=-1)
        scores[:, -1] = torch.nan
        return scores

    rank = GPCRank(policy, world_model, nan_last, num_samples=N, spread=1.0)
    frames, proprio = make_obs(k=2)
    # The last proposal would score highest; with it NaN, the next-best is chosen
    assert_close(rank.plan(frames, proprio), rank._propose(frames, proprio)[:, -2])


def test_plan_all_nan_scores_returns_first_proposal(policy, world_model):
    all_nan = lambda f, g: torch.full(f.shape[:2], torch.nan)  # noqa: E731
    rank = GPCRank(policy, world_model, all_nan, num_samples=N)  # pyrefly: ignore[bad-argument-type]
    frames, proprio = make_obs(k=2)
    assert_close(rank.plan(frames, proprio), rank._propose(frames, proprio)[:, 0])


def test_objective_receives_latent_futures(rank, objective, policy):
    rank.plan(*make_obs(k=2))
    futures, _, _ = objective.calls[0]
    assert futures.shape == (B, N, policy.prediction_horizon, 1)


def test_frame_objective_receives_frames(policy):
    world_model = DecodingWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    objective = frame_objective(SumObjective())
    rank = GPCRank(policy, world_model, objective, num_samples=N)

    rank.plan(*make_obs(k=2))
    futures, _, _ = objective.calls[0]
    assert futures.shape == (B, N, policy.prediction_horizon, C, H, W)


def test_simulate_returns_latents_for_frame_objective(policy):
    world_model = DecodingWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    rank = GPCRank(policy, world_model, frame_objective(SumObjective()), num_samples=N)
    frames, proprio = make_obs(k=2)
    futures = rank.simulate(rank.encode_context(frames), rank._propose(frames, proprio))
    assert futures.shape == (B, N, policy.prediction_horizon, 1)


def test_frame_function_objective(policy):
    world_model = DecodingWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    seen = []

    @frame_objective
    def objective(futures, goal):
        seen.append(futures)
        return futures.flatten(2).sum(dim=-1)

    GPCRank(policy, world_model, objective, num_samples=N).plan(*make_obs(k=2))
    assert seen[0].shape[3:] == (C, H, W)


def test_frame_class_objective(policy):
    world_model = DecodingWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))

    @frame_objective
    class Objective(SumObjective):
        pass

    objective = Objective()
    GPCRank(policy, world_model, objective, num_samples=N).plan(*make_obs(k=2))
    assert objective.calls[0][0].shape[3:] == (C, H, W)


def test_frame_objective_requires_decoding_world_model(policy, world_model):
    with pytest.raises(ValueError, match="decode"):
        GPCRank(policy, world_model, frame_objective(SumObjective()))


def test_objective_without_plans_is_called_without_them(policy, world_model):
    calls = []

    def objective(futures, goal):
        calls.append((futures, goal))
        return futures.flatten(2).sum(dim=-1)

    GPCRank(policy, world_model, objective, num_samples=N).plan(*make_obs(k=2))
    assert len(calls) == 1


def test_module_objective_receives_plans_via_forward(policy, world_model):
    class Penalty(nn.Module):
        def forward(self, futures, goal, *, plans):
            self.plans = plans
            return -plans.square().sum(dim=(2, 3))

    objective = Penalty()
    rank = GPCRank(policy, world_model, objective, num_samples=N)
    frames, proprio = make_obs(k=2)
    plan = rank.plan(frames, proprio)

    assert_close(objective.plans, rank._propose(frames, proprio))
    # FakePolicy's actions grow with the proposal index, so the first has the least effort
    assert_close(plan, objective.plans[:, 0])


def test_latent_objective_receives_latents(policy, world_model):
    objective = latent_objective(SumObjective())
    GPCRank(policy, world_model, objective, num_samples=N).plan(*make_obs(k=2))
    assert objective.calls[0][0].shape[3:] == (1,)


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
