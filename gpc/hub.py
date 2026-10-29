"""Saving to and loading from the Hugging Face Hub.

A component (policy, world model or module objective) is saved to its own subfolder of a
repo, so several components can share one repo without overwriting each other. Its
``config.json`` records the concrete class, so loading through a base class returns the
saved subclass.

Loading imports the modules a repo names. Only installed code runs; no code is downloaded.
"""

from __future__ import annotations

import importlib
import json
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from huggingface_hub import HfApi, ModelHubMixin, constants, snapshot_download
from huggingface_hub.utils import SoftTemporaryDirectory, is_jsonable
from safetensors.torch import load_model, save_model

if TYPE_CHECKING:
    from torch import nn

    from gpc.models.config import Config

CLASS_KEY = "_class"
"""Config key holding the ``module:qualname`` of the saved class."""

VERSION_KEY = "_gpc_version"
"""Config key holding the gpclib version that saved the config."""

try:
    GPC_VERSION = version("gpclib")
except PackageNotFoundError:  # pragma: no cover
    GPC_VERSION = "unknown"


def class_path(cls: type) -> str:
    """The ``module:qualname`` path of a class or function."""
    return f"{cls.__module__}:{cls.__qualname__}"


def import_path(path: str) -> Any:
    """Import the object at a ``module:qualname`` path."""
    module, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def encode_ref(obj: Any) -> dict[str, Any] | None:
    """Reference to an importable function or class, or to a ``functools.partial`` of one with
    JSON keyword arguments, as ``{"ref": path, "kwargs": {...}}``. ``None`` if ``obj`` can't be
    referenced, e.g. a lambda, a local function or an instance."""
    if isinstance(obj, partial):
        inner = encode_ref(obj.func)
        if inner is None or "kwargs" in inner or obj.args or not is_jsonable(obj.keywords):
            return None
        # Through JSON, so the reference is the same before and after saving
        return {**inner, "kwargs": json.loads(json.dumps(obj.keywords))}

    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or "<" in qualname:
        return None
    path = f"{module}:{qualname}"
    try:
        if import_path(path) is not obj:
            return None
    except (ImportError, AttributeError):
        return None
    return {"ref": path}


def decode_ref(ref: dict[str, Any]) -> Any:
    """Inverse of [`encode_ref`](gpc.hub.encode_ref)."""
    obj = import_path(ref["ref"])
    return partial(obj, **ref["kwargs"]) if "kwargs" in ref else obj


def is_ref(value: Any) -> bool:
    """Whether a config value is a reference made by [`encode_ref`](gpc.hub.encode_ref)."""
    return isinstance(value, dict) and isinstance(value.get("ref"), str)


def resolve(
    model_id: str | Path, allow_patterns: list[str] | None = None, **hub_kwargs: Any
) -> Path:
    """Local directory holding ``model_id``: the directory itself, or a snapshot of the repo
    limited to ``allow_patterns``."""
    if Path(model_id).is_dir():
        return Path(model_id)
    return Path(
        snapshot_download(repo_id=str(model_id), allow_patterns=allow_patterns, **hub_kwargs)
    )


class HubBase(
    ModelHubMixin,
    library_name="gpclib",
    repo_url="https://github.com/alexander-brady/gpc",
    tags=["gpc"],
):
    """Shared push logic: one commit per push, never overwriting an existing model card."""

    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict[str, Any] | Any | None = None,
        commit_message: str = "Push model using gpclib.",
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        model_card_kwargs: dict[str, Any] | None = None,
        **save_kwargs: Any,
    ) -> str:
        """Save to a temporary directory and upload it to ``repo_id`` in a single commit.

        Files already in the repo are kept unless overwritten by this push or matched by
        ``delete_patterns``. Extra keyword arguments are passed to ``save_pretrained``.
        """
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        ignore = [ignore_patterns] if isinstance(ignore_patterns, str) else ignore_patterns or []
        if api.file_exists(repo_id, "README.md", revision=branch):
            ignore = [*ignore, "README.md"]

        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(
                saved_path, config=config, model_card_kwargs=model_card_kwargs, **save_kwargs
            )
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore or None,
                delete_patterns=delete_patterns,
            )

    def _save_model_card(self, save_directory: Path, model_card_kwargs: dict | None) -> None:
        """Write the model card to ``save_directory`` unless one is there already."""
        card_path = save_directory / "README.md"
        if not card_path.exists():
            self.generate_model_card(**(model_card_kwargs or {})).save(card_path)


class HubMixin(HubBase):
    """Hub support for an ``nn.Module`` component built from a [`Config`](gpc.models.Config).

    The component is saved to ``<repo>/<hub_subfolder>/`` as ``config.json`` and
    ``model.safetensors``. ``Base.from_pretrained`` returns the saved subclass, which must
    subclass ``Base``.

    Subclasses must be constructible as ``cls(config)`` with ``config`` an instance of their
    ``config_class``, like ``transformers`` models.

    Attributes:
        config_class (type[Config]): Config dataclass the component is built from.
        hub_subfolder (str): Default subfolder of the repo the component is saved to.
    """

    config_class: ClassVar[type[Config]]
    hub_subfolder: ClassVar[str]
    config: Any

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config: dict[str, Any] | Any | None = None,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        model_card_kwargs: dict[str, Any] | None = None,
        subfolder: str | None = None,
        **push_to_hub_kwargs: Any,
    ) -> str | None:
        """Save the component's config and weights to ``save_directory/subfolder``.

        Args:
            save_directory (str | Path): Root directory of the saved repo.
            config (dict | Config | None): Config to save instead of ``self.config``.
            repo_id (str | None): Repo to push to, if ``push_to_hub``. Defaults to the
                directory name.
            push_to_hub (bool): Whether to push the saved files to the Hub.
            model_card_kwargs (dict | None): Arguments for the root model card template.
            subfolder (str | None): Subfolder to save to. Defaults to ``hub_subfolder``;
                ``""`` saves to the root.

        Returns:
            str | None: The commit URL if pushed, ``None`` otherwise.
        """
        save_directory = Path(save_directory)
        subfolder = self.hub_subfolder if subfolder is None else subfolder
        folder = save_directory / subfolder
        folder.mkdir(parents=True, exist_ok=True)

        config = self.config if config is None else config
        values = dict(config) if isinstance(config, dict) else config.to_dict()
        payload = {CLASS_KEY: class_path(type(self)), VERSION_KEY: GPC_VERSION, **values}
        (folder / constants.CONFIG_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True))
        module: nn.Module = self  # pyrefly: ignore[bad-assignment]
        save_model(module, str(folder / constants.SAFETENSORS_SINGLE_FILE))
        self._save_model_card(save_directory, model_card_kwargs)

        if push_to_hub:
            return self.push_to_hub(
                repo_id or save_directory.name,
                model_card_kwargs=model_card_kwargs,
                subfolder=subfolder,
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
        subfolder: str | None = None,
        map_location: str = "cpu",
        strict: bool = True,
        **config_overrides: Any,
    ) -> Self:
        """Load a component saved with ``save_pretrained``, as the subclass that saved it.

        Only ``subfolder`` is downloaded from the repo.

        Args:
            pretrained_model_name_or_path (str | Path): Repo id or local directory.
            subfolder (str | None): Subfolder to load from. Defaults to ``hub_subfolder``.
            map_location (str): Device to load the weights onto.
            strict (bool): Whether every weight must match the model exactly.
            config_overrides: Config fields to change before building the component.

        Returns:
            The loaded component, in evaluation mode.

        Raises:
            TypeError: If the saved class is not a subclass of ``cls``.
        """
        subfolder = cls.hub_subfolder if subfolder is None else subfolder
        files = [constants.CONFIG_NAME, constants.SAFETENSORS_SINGLE_FILE]
        root = resolve(
            pretrained_model_name_or_path,
            [f"{subfolder}/{name}" for name in files] if subfolder else files,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        folder = root / subfolder
        values = json.loads((folder / constants.CONFIG_NAME).read_text())
        values.pop(VERSION_KEY, None)
        path = values.pop(CLASS_KEY, None)

        target = cls if path is None else import_path(path)
        if not (isinstance(target, type) and issubclass(target, cls)):
            raise TypeError(
                f"{pretrained_model_name_or_path!r} holds a {path}, not a {cls.__qualname__}."
            )

        config = target.config_class.from_dict({**values, **config_overrides})
        model = target(config)  # pyrefly: ignore[bad-instantiation]
        module: nn.Module = model  # pyrefly: ignore[bad-assignment]
        load_model(
            module,
            str(folder / constants.SAFETENSORS_SINGLE_FILE),
            strict=strict,
            device=map_location,
        )
        module.eval()
        return model
