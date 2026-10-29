"""Saving components and controllers locally, and pushing to / pulling from a mocked Hub."""

import json
from functools import partial
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import optim
from torch.testing import assert_close

from gpc import GPCOpt, GPCRank, GPCRankOpt, hub
from gpc.controllers.base import GPC
from gpc.hub import decode_ref, encode_ref
from gpc.models import Policy, PolicyConfig, WorldModel, WorldModelConfig
from gpc.objectives import ObjectiveModule, frame_objective, is_frame_objective
from tests.fakes import (
    DecodingWorldModel,
    FakePolicy,
    FakeWorldModel,
    SumObjective,
    WeightedObjective,
    WeightedObjectiveConfig,
    scaled_objective,
    total_objective,
)


def read_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


# Components


def test_policy_round_trips_as_saved_subclass(tmp_path, policy):
    with torch.no_grad():
        policy.offset.fill_(3.0)
    policy.save_pretrained(tmp_path)

    loaded = Policy.from_pretrained(tmp_path)

    assert type(loaded) is FakePolicy
    assert loaded.config == policy.config
    assert_close(loaded.offset, policy.offset)
    assert not loaded.training
    assert read_config(tmp_path / "policy")["_class"] == "tests.fakes:FakePolicy"


def test_loading_through_unrelated_base_raises(tmp_path, policy):
    policy.save_pretrained(tmp_path, subfolder="")
    with pytest.raises(TypeError, match="not a WorldModel"):
        WorldModel.from_pretrained(tmp_path, subfolder="")


def test_components_share_a_directory(tmp_path, policy, world_model):
    policy.save_pretrained(tmp_path)
    world_model.save_pretrained(tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {"policy", "world_model", "README.md"}
    assert isinstance(Policy.from_pretrained(tmp_path), FakePolicy)
    assert isinstance(WorldModel.from_pretrained(tmp_path), FakeWorldModel)


@pytest.mark.parametrize("subfolder", ["", "policies/fast"])
def test_custom_subfolder(tmp_path, policy, subfolder):
    policy.save_pretrained(tmp_path, subfolder=subfolder)
    assert (tmp_path / subfolder / "model.safetensors").exists()
    assert Policy.from_pretrained(tmp_path, subfolder=subfolder).config == policy.config


def test_config_overrides(tmp_path, policy):
    policy.save_pretrained(tmp_path)
    loaded = Policy.from_pretrained(tmp_path, prediction_horizon=7)
    assert loaded.config == PolicyConfig(observation_horizon=2, prediction_horizon=7)


def test_existing_model_card_is_kept(tmp_path, policy):
    (tmp_path / "README.md").write_text("mine")
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "README.md").read_text() == "mine"


# Mocked Hub


class FakeApi:
    """Records uploads in place of ``HfApi``."""

    uploads: ClassVar[list[dict]] = []
    readme_exists = False

    def __init__(self, token=None):
        pass

    def create_repo(self, repo_id, **kwargs):
        return type("Repo", (), {"repo_id": repo_id})

    def file_exists(self, repo_id, filename, **kwargs):
        return self.readme_exists

    def upload_folder(self, *, folder_path, **kwargs):
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*.*"))
        FakeApi.uploads.append({**kwargs, "files": files})
        return "commit-url"


@pytest.fixture
def api(monkeypatch):
    FakeApi.uploads, FakeApi.readme_exists = [], False
    monkeypatch.setattr(hub, "HfApi", FakeApi)
    return FakeApi


def test_push_uploads_only_the_component_subfolder(api, policy):
    assert policy.push_to_hub("me/repo") == "commit-url"
    (upload,) = api.uploads
    assert upload["repo_id"] == "me/repo"
    assert upload["files"] == ["README.md", "policy/config.json", "policy/model.safetensors"]


def test_push_keeps_remote_model_card(api, policy):
    api.readme_exists = True
    policy.push_to_hub("me/repo", subfolder="alt")
    assert api.uploads[0]["ignore_patterns"] == ["README.md"]
    assert "alt/config.json" in api.uploads[0]["files"]


def test_save_pretrained_can_push(api, tmp_path, policy):
    policy.save_pretrained(tmp_path, push_to_hub=True, repo_id="me/repo")
    assert api.uploads[0]["repo_id"] == "me/repo"


def test_download_is_limited_to_subfolder(monkeypatch, tmp_path, policy):
    policy.save_pretrained(tmp_path)
    requested = {}

    def snapshot_download(repo_id, allow_patterns, **kwargs):
        requested.update(repo_id=repo_id, allow_patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    Policy.from_pretrained("me/repo")
    assert requested == {
        "repo_id": "me/repo",
        "allow_patterns": ["policy/config.json", "policy/model.safetensors"],
    }


# References


def test_refs_round_trip():
    ref = encode_ref(total_objective)
    assert ref is not None
    assert decode_ref(ref) is total_objective

    # JSON stores tuples as lists
    ref = encode_ref(partial(optim.Adam, betas=(0.0, 0.9)))
    assert ref is not None
    assert ref == {"ref": "torch.optim.adam:Adam", "kwargs": {"betas": [0.0, 0.9]}}
    assert decode_ref(ref).keywords == {"betas": [0.0, 0.9]}


@pytest.mark.parametrize(
    "obj",
    [
        lambda f, g: f,
        SumObjective(),
        partial(scaled_objective, 2.0),  # pyrefly: ignore[bad-argument-type]
        partial(scaled_objective, scale=object()),  # pyrefly: ignore[bad-argument-type]
    ],
)
def test_unreferenceable_objects(obj):
    assert encode_ref(obj) is None


@pytest.mark.parametrize(
    ("module", "qualname"),
    [("no_such_module", "orphan"), ("tests.fakes", "total_objective")],
)
def test_function_not_importable_at_its_path_is_unreferenceable(module, qualname):
    def orphan(futures, goal):
        return futures

    orphan.__module__, orphan.__qualname__ = module, qualname
    assert encode_ref(orphan) is None


# Controllers


@pytest.fixture
def saved_opt(tmp_path, policy, world_model) -> Path:
    adam = partial(optim.Adam, betas=(0.0, 0.9))
    controller = GPCOpt(
        policy, world_model, total_objective, action_horizon=2, num_steps=7, optimizer=adam
    )
    controller.save_pretrained(tmp_path)
    return tmp_path


def test_controller_layout(saved_opt):
    assert {p.name for p in saved_opt.iterdir()} == {
        "config.json",
        "README.md",
        "policy",
        "world_model",
    }
    config = read_config(saved_opt)
    assert config["_class"] == "gpc.controllers.opt:GPCOpt"
    assert config["action_horizon"] == 2
    assert config["objective"] == {"ref": "tests.fakes:total_objective"}
    assert config["controllers"]["gpc.controllers.opt:GPCOpt"]["num_steps"] == 7


def test_controller_round_trip(saved_opt):
    controller = GPCOpt.from_pretrained(saved_opt)

    assert controller.action_horizon == 2
    assert controller.num_steps == 7
    assert isinstance(controller.optimizer, partial)
    assert controller.optimizer.func is optim.Adam
    assert controller.optimizer.keywords == {"betas": [0.0, 0.9]}
    assert controller.objective is total_objective
    assert isinstance(controller.policy, FakePolicy)
    assert isinstance(controller.world_model, FakeWorldModel)


def test_gpc_loads_the_saved_class(saved_opt):
    assert type(GPC.from_pretrained(saved_opt)) is GPCOpt


def test_other_controller_uses_shared_params_and_own_defaults(saved_opt):
    controller = GPCRank.from_pretrained(saved_opt)
    assert type(controller) is GPCRank
    assert controller.action_horizon == 2
    assert controller.num_samples == 64


def test_subclass_inherits_parent_sections(tmp_path, policy, world_model, saved_opt):
    rank = GPCRank.from_pretrained(saved_opt, num_samples=5)
    rank.save_pretrained(saved_opt)

    sections = read_config(saved_opt)["controllers"]
    assert set(sections) == {"gpc.controllers.opt:GPCOpt", "gpc.controllers.rank:GPCRank"}

    rank_opt = GPCRankOpt.from_pretrained(saved_opt)
    assert rank_opt.num_samples == 5
    assert rank_opt.num_steps == 7
    assert isinstance(rank_opt.optimizer, partial)
    assert rank_opt.optimizer.func is optim.Adam


def test_own_section_wins_over_parents(tmp_path, policy, world_model):
    GPCRank(policy, world_model, total_objective, num_samples=5).save_pretrained(tmp_path)
    GPCRankOpt.from_pretrained(tmp_path, num_samples=9).save_pretrained(tmp_path)
    assert GPCRankOpt.from_pretrained(tmp_path).num_samples == 9
    assert GPCRank.from_pretrained(tmp_path).num_samples == 5


def test_hparam_overrides(saved_opt):
    assert GPCOpt.from_pretrained(saved_opt, num_steps=1, action_horizon=3).num_steps == 1


def test_passed_components_replace_saved_ones(saved_opt, policy, world_model):
    objective = SumObjective()
    controller = GPCOpt.from_pretrained(
        saved_opt, policy=policy, world_model=world_model, objective=objective
    )
    assert controller.policy is policy
    assert controller.world_model is world_model
    assert controller.objective is objective


def test_module_objective_round_trips(tmp_path, policy, world_model):
    objective = WeightedObjective(WeightedObjectiveConfig(scale=2.0))
    with torch.no_grad():
        objective.weight.fill_(0.5)
    GPCRank(policy, world_model, objective).save_pretrained(tmp_path)

    assert read_config(tmp_path)["objective"] == {"subfolder": "objective"}
    loaded = GPCRank.from_pretrained(tmp_path).objective
    assert isinstance(loaded, WeightedObjective)
    assert loaded.config.scale == 2.0
    assert_close(loaded.weight, objective.weight)


def test_stale_objective_folder_is_ignored(tmp_path, policy, world_model):
    GPCRank(policy, world_model, WeightedObjective()).save_pretrained(tmp_path)
    GPCRank(policy, world_model, total_objective).save_pretrained(tmp_path)
    assert GPCRank.from_pretrained(tmp_path).objective is total_objective


def test_partial_frame_objective_keeps_its_marker(tmp_path, policy):
    world_model = DecodingWorldModel(WorldModelConfig(context_horizon=3, prediction_horizon=2))
    objective = frame_objective(partial(scaled_objective, scale=2.0))
    GPCRank(policy, world_model, objective).save_pretrained(tmp_path)

    loaded = GPCRank.from_pretrained(tmp_path).objective
    assert isinstance(loaded, partial)
    assert loaded.func is scaled_objective
    assert loaded.keywords == {"scale": 2.0}
    assert is_frame_objective(loaded)


def test_unsaveable_objective_must_be_passed(tmp_path, policy, world_model):
    with pytest.warns(UserWarning, match="can't be saved"):
        GPCRank(policy, world_model, SumObjective()).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="pass `objective=`"):
        GPCRank.from_pretrained(tmp_path)
    assert isinstance(
        GPCRank.from_pretrained(tmp_path, objective=SumObjective()).objective, SumObjective
    )


def test_unsaveable_hparam_warns_and_uses_default(tmp_path, policy, world_model):
    controller = GPCOpt(
        policy, world_model, total_objective, optimizer=lambda p, lr: optim.SGD(p, lr)
    )
    with pytest.warns(UserWarning, match="optimizer"):
        controller.save_pretrained(tmp_path)
    assert GPCOpt.from_pretrained(tmp_path).optimizer is optim.SGD


def test_controller_must_store_its_arguments(tmp_path, policy, world_model):
    class Forgetful(GPCRank):
        def __init__(self, policy, world_model, objective, *, temperature: float = 1.0):
            super().__init__(policy, world_model, objective)

    with pytest.raises(AttributeError, match=r"self\.temperature"):
        Forgetful(policy, world_model, total_objective).save_pretrained(tmp_path)


def test_gpc_rejects_non_controller_class(tmp_path, saved_opt):
    config = read_config(saved_opt)
    config["_class"] = "tests.fakes:FakePolicy"
    (saved_opt / "config.json").write_text(json.dumps(config))
    with pytest.raises(TypeError, match="not a GPC"):
        GPC.from_pretrained(saved_opt)


def test_controller_save_can_push(api, tmp_path, policy, world_model):
    GPCRank(policy, world_model, total_objective).save_pretrained(
        tmp_path, push_to_hub=True, repo_id="me/repo"
    )
    assert api.uploads[0]["repo_id"] == "me/repo"


def test_controller_push_uploads_everything(api, policy, world_model):
    GPCRank(policy, world_model, WeightedObjective()).push_to_hub("me/repo")
    assert api.uploads[0]["files"] == [
        "README.md",
        "config.json",
        "objective/config.json",
        "objective/model.safetensors",
        "policy/config.json",
        "policy/model.safetensors",
        "world_model/config.json",
        "world_model/model.safetensors",
    ]


def test_objective_module_defaults_to_empty_config():
    class Plain(ObjectiveModule):
        def forward(self, futures, goal):
            return futures.flatten(2).sum(-1)

    assert Plain().config == ObjectiveModule.config_class()
