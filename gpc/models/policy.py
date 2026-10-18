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

    The policy conditions on the latest K observations and is meant to
    execute T actions of each plan before replanning, creating
    N samples of planned actions for each input observation sequence.
    """

    def __init__(self, config: PolicyConfig[K, T]):
        super().__init__()
        self.config = config

    @property
    def observation_horizon(self) -> int:
        """Number of context frames K the policy conditions on."""
        return self.config.observation_horizon

    @property
    def action_horizon(self) -> int:
        """Number of frames T the policy predicts per pass."""
        return self.config.action_horizon

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
