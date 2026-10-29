from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from shape_extensions import Elements, Int, IntTuple, IntVar
from torch import Tensor

from gpc.hub import HubMixin
from gpc.models.config import PolicyConfig


class Policy[K: IntVar, T: IntVar](ABC, nn.Module, HubMixin):
    """Abstract base class for visuomotor policies that plan action chunks.

    Observations are frames [B, K, C, H, W] and optionally proprioception [B, K, P],
    the agent's own low-dimensional state (e.g. its position); actions are [B, T, A].
    All are in the policy's own normalized space.

    The policy conditions on the latest K observations and plans the next T actions.
    Sampling should be stochastic: controllers draw several plans per observation
    sequence with ``num_samples`` and choose among them.

    Saved to the ``policy/`` subfolder of a Hub repo with ``save_pretrained`` or
    ``push_to_hub``; ``Policy.from_pretrained`` returns the saved subclass. A subclass with
    its own config sets ``config_class`` to it (see [`HubMixin`](gpc.hub.HubMixin)).
    """

    config_class = PolicyConfig
    config: PolicyConfig[K, T]
    hub_subfolder = "policy"

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

    def clamp_actions[Ds: IntTuple](
        self, actions: Tensor[[*Elements[Ds], int]]
    ) -> Tensor[[*Elements[Ds], int]]:
        """Clamp actions [..., A] to the bounds set by ``action_low`` and ``action_high``.

        Args:
            actions (Tensor[..., A]): Actions in the policy's normalized space.

        Returns:
            Tensor[..., A]: The clamped actions, or ``actions`` itself if unbounded.
        """
        low, high = self.config.action_low, self.config.action_high
        if low is None and high is None:
            return actions
        return actions.clamp(
            None if low is None else actions.new_tensor(low),
            None if high is None else actions.new_tensor(high),
        )

    @abstractmethod
    def sample[B: IntVar, N: IntVar](
        self,
        frames: Tensor[[B, K, int, int, int]],
        proprio: Tensor[[B, K, int]],
        *,
        num_samples: Int[N] = 1,
    ) -> Tensor[[B, N, T, int]]:
        """Sample ``num_samples`` independent plans per batch element, from the current step
        onwards.

        Drawing them in one call lets the policy share work between samples of the same
        observation, such as encoding it once.

        Args:
            frames (Tensor[[B, K, C, H, W]]): observed frames o_(t-K+1), ..., o_t
            proprio (Tensor[[B, K, P]]): observed proprioception p_(t-K+1), ..., p_t
            num_samples (Int[N] > 0): number of plans to sample per batch element

        Returns:
            Tensor[[B, N, T, A]]: Planned actions.
        """
