from __future__ import annotations

import inspect
import json
import warnings
from abc import ABC, abstractmethod
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn as nn
from huggingface_hub import constants
from huggingface_hub.utils import is_jsonable
from shape_extensions import Elements, IntTuple, IntVar
from torch import Tensor

from gpc.hub import (
    CLASS_KEY,
    GPC_VERSION,
    VERSION_KEY,
    HubBase,
    class_path,
    decode_ref,
    encode_ref,
    import_path,
    is_ref,
    resolve,
)
from gpc.models import Policy, WorldModel
from gpc.objectives import Objective, ObjectiveModule, accepts_plans, is_frame_objective
from gpc.trajectory import fit_horizon

COMPONENTS = ("policy", "world_model", "objective")
"""Constructor arguments of a controller that are saved as components, not hyperparameters."""

SECTIONS_KEY = "controllers"
"""Config key holding each controller class's own hyperparameters."""


class GPC(ABC, nn.Module, HubBase):
    """Abstract base class for a Generative Predictive Control (GPC) controller.

    ``save_pretrained`` and ``push_to_hub`` write the policy, world model and (if it is an
    [`ObjectiveModule`](gpc.objectives.ObjectiveModule)) objective to their own subfolders,
    and the hyperparameters to ``config.json``. Hyperparameters of ``GPC`` itself are shared
    by every controller; each subclass's own are kept in a section per class, so any
    controller can be loaded from a repo saved by another (see
    [`from_pretrained`](gpc.controllers.base.GPC.from_pretrained)). A subclass must store
    each constructor argument as an attribute of the same name.

    Attributes:
        policy (Policy): The generative policy model used for action selection.
        world_model (WorldModel): World model used for predicting future states.
        objective (Objective): The objective function used to evaluate action plans.
        action_horizon (int > 0): The number of action steps to execute before replanning.
    """

    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        objective: Objective,
        *,
        action_horizon: int = 1,
    ):
        super().__init__()

        if action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer.")
        if is_frame_objective(objective) and type(world_model).decode is WorldModel.decode:
            raise ValueError("A frame objective requires a world model that implements decode.")

        self.policy = policy
        self.world_model = world_model
        self.objective = objective
        self.action_horizon = action_horizon

        self.history_size = max(policy.observation_horizon, world_model.context_horizon)

        self.reset()

    def reset(self) -> None:
        """Reset the internal state of the GPC controller."""
        self._frames = deque(maxlen=self.history_size)
        self._proprio = deque(maxlen=self.history_size)

        self._pending: Tensor | None = None
        self._action_index: int = 0

    def act[B: IntVar](
        self,
        frames: Tensor[[B, 1, int, int, int]],
        *,
        proprio: Tensor[[B, 1, int]],
        goal: Tensor[[B, int]] | None = None,
    ) -> Tensor[[B, int]]:
        """Consume an observation and return the next action.

        Args:
            frames (Tensor[B, K, C, H, W]): Observed frames to condition
            proprio (Tensor[B, K, P]): Observed proprioception
            goal (Tensor[B, GoalDim]): Goal to plan towards.

        Returns:
            Tensor[B, A]: the next action to take.
        """
        self._frames.extend(frames.unbind(dim=1))
        self._proprio.extend(proprio.unbind(dim=1))

        # Replan when the action chunk is exhausted
        if self._pending is None or self._action_index >= self._pending.shape[1]:
            frames = torch.stack(list(self._frames), dim=1)
            proprio = torch.stack(list(self._proprio), dim=1)

            plan = self.plan(frames, proprio=proprio, goal=goal)

            self._pending = plan[:, : self.action_horizon]
            self._action_index = 0

        action = self._pending[:, self._action_index]
        self._action_index += 1

        return action

    def encode_context[B: IntVar](self, frames: Tensor[[B, int, int, int, int]]) -> Tensor:
        """Encode the latest observed frames into the world model's context latents.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition on

        Returns:
            Tensor[B, K', D...]: latents of the ``context_horizon`` latest frames.
        """
        return self.world_model.encode(fit_horizon(frames, self.world_model.context_horizon))

    def simulate[B: IntVar, N: IntVar, T: IntVar, D: IntTuple](
        self,
        context: Tensor[[B, int, *Elements[D]]],
        plans: Tensor[[B, N, T, int]],
    ) -> Tensor[[B, N, T, *Elements[D]]]:
        """Roll out each candidate plan with the world model.

        Args:
            context (Tensor[B, K, D...]): context latents from [`encode_context`](gpc.controllers.base.GPC.encode_context)
            plans (Tensor[B, N, T, A]): candidate action sequences

        Returns:
            Tensor[B, N, T, D...]: predicted latents.
        """
        batch, num_plans = plans.shape[:2]
        context = context.repeat_interleave(num_plans, dim=0)
        futures = self.world_model.rollout(context, plans.flatten(0, 1))
        return futures.unflatten(0, (batch, num_plans))

    def score[B: IntVar, N: IntVar](
        self,
        futures: Tensor,
        goal: Tensor[[B, int]] | None,
        plans: Tensor[[B, N, int, int]],
    ) -> Tensor[[B, N]]:
        """Score simulated futures with the objective, decoding them into frames for a frame
        objective and passing the plans if it accepts them.

        Args:
            futures (Tensor[B, N, T, D...]): latent futures from [`simulate`](gpc.controllers.base.GPC.simulate)
            goal (Tensor[B, GoalDim] | None): the optional goal to evaluate against
            plans (Tensor[B, N, T, A]): the candidate plans that produced the futures

        Returns:
            Tensor[B, N]: one score per batch and plan.
        """
        objective = self.objective
        if is_frame_objective(objective):
            futures = self.world_model.decode(futures.flatten(0, 1)).unflatten(0, plans.shape[:2])
        if accepts_plans(objective):
            return objective(futures, goal, plans=plans)
        return objective(futures, goal)

    @abstractmethod
    def plan[B: IntVar, K: IntVar](
        self,
        frames: Tensor[[B, K, int, int, int]],
        proprio: Tensor[[B, K, int]],
        goal: Tensor[[B, int]] | None = None,
    ) -> Tensor[[B, int, int]]:
        """Plan from an explicit observation history. Not stateful.

        Args:
            frames (Tensor[B, K, C, H, W]): observed frames to condition
            proprio (Tensor[B, K, P]): observed proprioception
            goal (Tensor[B, GoalDim]): the goal to plan towards.

        Returns:
            Tensor[B, T, A]: the next action(s) to take.
        """

    @classmethod
    def _hparam_names(cls) -> list[str]:
        """Names of the constructor arguments that are hyperparameters."""
        return [
            name
            for name, p in inspect.signature(cls.__init__).parameters.items()
            if name != "self"
            and name not in COMPONENTS
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]

    def _encode_hparams(self, names: list[str]) -> dict[str, Any]:
        """JSON values of the named hyperparameters, skipping (with a warning) any that are
        neither JSON nor referenceable."""
        encoded = {}
        for name in names:
            if not hasattr(self, name):
                raise AttributeError(
                    f"{type(self).__qualname__} must store its argument {name!r} as "
                    f"self.{name} to be saved."
                )
            value = getattr(self, name)
            if is_jsonable(value) and not is_ref(value):
                encoded[name] = value
            elif (ref := encode_ref(value)) is not None:
                encoded[name] = ref
            else:
                warnings.warn(
                    f"{type(self).__qualname__}.{name}={value!r} can't be saved; "
                    "its default is used when loading.",
                    stacklevel=3,
                )
        return encoded

    def _encode_objective(self, save_directory: Path) -> dict[str, Any] | None:
        objective = self.objective
        if isinstance(objective, ObjectiveModule):
            objective.save_pretrained(save_directory)
            return {"subfolder": objective.hub_subfolder}
        ref = encode_ref(objective)
        if ref is None:
            warnings.warn(
                f"Objective {objective!r} can't be saved; it must be an ObjectiveModule, an "
                "importable function, or a functools.partial of one with JSON keyword "
                "arguments. Pass it as `objective=` to `from_pretrained`.",
                stacklevel=3,
            )
            return None
        if isinstance(objective, partial) and hasattr(objective, "frame_objective"):
            ref["frame_objective"] = objective.frame_objective
        return ref

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config: dict | Any | None = None,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        model_card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs: Any,
    ) -> str | None:
        """Save the controller's components and hyperparameters to ``save_directory``.

        The policy and world model go to ``policy/`` and ``world_model/``. The objective goes
        to ``objective/`` if it is an [`ObjectiveModule`](gpc.objectives.ObjectiveModule), is
        referenced from ``config.json`` if it is an importable function (or a
        ``functools.partial`` of one), and is otherwise not saved, with a warning.

        Hyperparameter sections of other controllers, read when this one was loaded, are
        written back.

        Args:
            save_directory (str | Path): Directory to save to.
            config (dict | None): Extra entries for ``config.json``.
            repo_id (str | None): Repo to push to, if ``push_to_hub``. Defaults to the
                directory name.
            push_to_hub (bool): Whether to push the saved files to the Hub.
            model_card_kwargs (dict | None): Arguments for the model card template.

        Returns:
            str | None: The commit URL if pushed, ``None`` otherwise.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        shared = GPC._hparam_names()
        own = [name for name in self._hparam_names() if name not in shared]
        sections = dict(getattr(self, "_hub_sections", {}))
        sections[class_path(type(self))] = self._encode_hparams(own)
        payload: dict[str, Any] = {
            **(config or {}),
            CLASS_KEY: class_path(type(self)),
            VERSION_KEY: GPC_VERSION,
            **self._encode_hparams(shared),
            SECTIONS_KEY: sections,
        }

        self.policy.save_pretrained(save_directory)
        self.world_model.save_pretrained(save_directory)
        objective = self._encode_objective(save_directory)
        if objective is not None:
            payload["objective"] = objective

        config_path = save_directory / constants.CONFIG_NAME
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._save_model_card(save_directory, model_card_kwargs)

        if push_to_hub:
            return self.push_to_hub(
                repo_id or save_directory.name,
                model_card_kwargs=model_card_kwargs,
                **push_to_hub_kwargs,
            )
        return None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        policy: Policy | None = None,
        world_model: WorldModel | None = None,
        objective: Objective | None = None,
        map_location: str = "cpu",
        strict: bool = True,
        **hparams: Any,
    ) -> Self:
        """Build this controller class from a repo saved by any controller.

        Shared hyperparameters are always read. Then the section of each controller class in
        ``cls``'s MRO is applied, base classes first, so a controller inherits the saved
        values of its parents (``GPCRankOpt`` reads those saved by ``GPCRank`` and
        ``GPCOpt``) and its own win. Hyperparameters that weren't saved take their defaults.
        Called on ``GPC`` itself, it builds the controller class that saved the repo.

        Args:
            pretrained_model_name_or_path (str | Path): Repo id or local directory.
            policy (Policy | None): Policy to use instead of the saved one.
            world_model (WorldModel | None): World model to use instead of the saved one.
            objective (Objective | None): Objective to use instead of the saved one. Required
                if none was saved.
            map_location (str): Device to load the component weights onto.
            strict (bool): Whether every component weight must match exactly.
            hparams: Hyperparameters to use instead of the saved ones.

        Returns:
            The loaded controller.
        """
        hub_kwargs: dict[str, Any] = {
            "force_download": force_download,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        root = resolve(pretrained_model_name_or_path, **hub_kwargs)
        config = json.loads((root / constants.CONFIG_NAME).read_text())

        target: type[Self] = cls
        if cls is GPC:
            target = import_path(config[CLASS_KEY])
            if not (isinstance(target, type) and issubclass(target, GPC)):
                raise TypeError(f"{pretrained_model_name_or_path!r} holds a {target}, not a GPC.")

        sections: dict[str, dict[str, Any]] = config.get(SECTIONS_KEY, {})
        saved = {name: config[name] for name in GPC._hparam_names() if name in config}
        for base in reversed(target.__mro__):
            if isinstance(base, type) and issubclass(base, GPC):
                saved.update(sections.get(class_path(base), {}))
        values = {
            name: decode_ref(saved[name]) if is_ref(saved[name]) else saved[name]
            for name in target._hparam_names()
            if name in saved
        }

        load_kwargs = {"map_location": map_location, "strict": strict}
        if policy is None:
            policy = Policy.from_pretrained(root, **load_kwargs)
        if world_model is None:
            world_model = WorldModel.from_pretrained(root, **load_kwargs)
        if objective is None:
            objective = cls._load_objective(root, config.get("objective"), **load_kwargs)
            if objective is None:
                raise ValueError(
                    f"No objective was saved to {pretrained_model_name_or_path!r}; "
                    "pass `objective=` to `from_pretrained`."
                )

        controller = target(policy, world_model, objective, **{**values, **hparams})
        controller._hub_sections = sections
        return controller

    @staticmethod
    def _load_objective(
        root: Path, saved: dict[str, Any] | None, **load_kwargs: Any
    ) -> Objective | None:
        if saved is None:
            return None
        if "subfolder" in saved:
            return ObjectiveModule.from_pretrained(
                root, subfolder=saved["subfolder"], **load_kwargs
            )
        objective = decode_ref(saved)
        if "frame_objective" in saved and isinstance(objective, partial):
            objective.frame_objective = saved["frame_objective"]  # pyrefly: ignore
        return objective
