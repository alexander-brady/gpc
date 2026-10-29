from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from shape_extensions import Elements, IntTuple, IntVar
from torch import Tensor

from gpc.hub import HubMixin
from gpc.models.config import WorldModelConfig
from gpc.trajectory import fit_horizon


class WorldModel[K: IntVar, T: IntVar, D: IntTuple](ABC, nn.Module, HubMixin):
    """Abstract base class for action-conditioned world models.

    Frames are laid out as [B, K, D...] and actions as trajectory [B, T, int],
    predicting up to T future latent frames per pass. Long horizons can be handled
    with :meth:`rollout`, which predicts autoregressively in chunks of T frames,
    conditioned on the latest K frames.

    Saved to the ``world_model/`` subfolder of a Hub repo with ``save_pretrained`` or
    ``push_to_hub``; ``WorldModel.from_pretrained`` returns the saved subclass. A subclass
    with its own config sets ``config_class`` to it (see [`HubMixin`](gpc.hub.HubMixin)).
    """

    config_class = WorldModelConfig
    hub_subfolder = "world_model"

    def __init__(self, config: WorldModelConfig[K, T]):
        super().__init__()
        self.config = config

    @property
    def context_horizon(self) -> int:
        """Number of context frames K the model conditions on."""
        return self.config.context_horizon

    @property
    def prediction_horizon(self) -> int:
        """Number of frames T the model predicts per pass."""
        return self.config.prediction_horizon

    @abstractmethod
    def encode[B: IntVar, F: IntVar](
        self, frames: Tensor[[B, F, int, int, int]]
    ) -> Tensor[[B, F, *Elements[D]]]:
        """Encode environment frames into the world model's latent space.

        Args:
            frames (Tensor[[B, F, C, H, W]]): Environment frames

        Returns:
            Tensor[[B, F, D...]]: Latent frames
        """

    def decode[B: IntVar, F: IntVar](
        self, latents: Tensor[[B, F, *Elements[D]]]
    ) -> Tensor[[B, F, int, int, int]]:
        """Decode latent frames back into environment frames.

        Args:
            latents (Tensor[[B, F, D...]]): Latent frames

        Returns:
            Tensor[[B, F, C, H, W]]: Environment frames
        """
        raise NotImplementedError("Decoding is not implemented for this world model.")

    @abstractmethod
    def predict[B: IntVar](
        self, history: Tensor[[B, K, *Elements[D]]], actions: Tensor[[B, T, int]]
    ) -> Tensor[[B, T, *Elements[D]]]:
        """Predict the next T frames for a given action trajectory.

        Args:
            history (Tensor[[B, K, D...]]): Context latent frames
            actions (Tensor[[B, T, int]]): Action trajectory

        Returns:
            Tensor[[B, T, D...]]: Predicted latent trajectory
        """

    def rollout[B: IntVar, F: IntVar](
        self, history: Tensor[[B, K, *Elements[D]]], actions: Tensor[[B, F, int]]
    ) -> Tensor[[B, F, *Elements[D]]]:
        """Predict an arbitrary number of future frames autoregressively.
        Each pass predicts T frames, conditioned on the latest K frames.
        A history shorter than K is left-padded by repeating its earliest frame.

        Args:
            history (Tensor[[B, K, D...]]): Context latent frames
            actions (Tensor[[B, F, int]]): Action trajectory

        Returns:
            Tensor[[B, F, D...]]: Predicted latent trajectory
        """
        context: Tensor[[B, K, *Elements[D]]] = fit_horizon(history, self.context_horizon)
        predictions = []
        for chunk in actions.split(self.prediction_horizon, dim=1):
            predicted = self.predict(context, chunk)
            predictions.append(predicted)
            context = torch.cat([context, predicted], dim=1)[:, -self.context_horizon :]
        return torch.cat(predictions, dim=1)
