from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from shape_extensions import IntVar
from torch import Tensor

from gpc.models.config import PolicyConfig


class Policy[K: IntVar, T: IntVar](ABC, nn.Module):
    """Abstract base class for visuomotor policies that plan action chunks.

    Observations are frames [B, K, C, H, W] and optionally proprioception [B, K, P],
    the agent's own low-dimensional state (e.g. its position); actions are [B, T, A].
    All are in the policy's own normalized space.

    The policy conditions on the latest K observations and plans the next T actions.
    Sampling should be stochastic: controllers draw several plans per observation
    sequence and choose among them.
    """

    def __init__(self, config: PolicyConfig[K, T]):
        super().__init__()
        self.config = config

    @property
    def observation_horizon(self) -> int:
        """Number of context frames K the policy conditions on."""
        return self.config.observation_horizon

    @property
    def prediction_horizon(self) -> int:
        """Number of actions T the policy plans per pass."""
        return self.config.prediction_horizon

    @abstractmethod
    def sample[B: IntVar](
        self, frames: Tensor[[B, K, int, int, int]], proprio: Tensor[[B, K, int]]
    ) -> Tensor[[B, T, int]]:
        """Plan actions from the current step onwards.

        Args:
            frames (Tensor[[B, K, C, H, W]]): observed frames o_(t-K+1), ..., o_t
            proprio (Tensor[[B, K, P]] | None): observed proprioception p_(t-K+1), ..., p_t

        Returns:
            Tensor[[B, T, A]]: Planned actions.
        """
