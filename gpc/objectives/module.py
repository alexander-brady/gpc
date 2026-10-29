from __future__ import annotations

from torch import nn

from gpc.hub import HubMixin
from gpc.models.config import Config


class ObjectiveModule(nn.Module, HubMixin):
    """Base class for objectives with weights or settings to save with a controller, such
    as a learned reward model.

    Saved to the ``objective/`` subfolder of a controller's repo, and loaded back as the
    saved subclass. Implement ``forward`` with the signature of an
    [`Objective`](gpc.objectives.Objective). A subclass with settings defines a
    [`Config`](gpc.models.Config) for them and sets ``config_class`` to it.
    """

    config_class = Config
    hub_subfolder = "objective"
    frame_objective: bool = False

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config if config is not None else self.config_class()
