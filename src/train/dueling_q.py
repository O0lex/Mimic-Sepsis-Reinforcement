from __future__ import annotations

import dataclasses
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from d3rlpy.models.q_functions import QFunctionFactory, register_q_func_factory
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.models.torch.q_functions.base import (
    ContinuousQFunction,
    ContinuousQFunctionForwarder,
    DiscreteQFunction,
    DiscreteQFunctionForwarder,
    QFunctionOutput,
)
from d3rlpy.models.torch.q_functions.utility import (
    compute_huber_loss,
    compute_reduce,
    pick_value_by_action,
)
from d3rlpy.types import TorchObservation


class DuelingDiscreteQFunction(DiscreteQFunction):
    _encoder: Encoder
    _value_fc: nn.Linear
    _advantage_fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._value_fc = nn.Linear(hidden_size, 1)
        self._advantage_fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        h = self._encoder(x)
        value = self._value_fc(h)
        advantage = self._advantage_fc(h)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return QFunctionOutput(q_value=q, quantiles=None, taus=None)

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class DuelingDiscreteQFunctionForwarder(DiscreteQFunctionForwarder):
    _q_func: DuelingDiscreteQFunction
    _action_size: int

    def __init__(self, q_func: DuelingDiscreteQFunction, action_size: int):
        self._q_func = q_func
        self._action_size = action_size

    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        return self._q_func(x).q_value

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        one_hot = F.one_hot(actions.view(-1), num_classes=self._action_size)
        value = (self._q_func(observations).q_value * one_hot.float()).sum(dim=1, keepdim=True)
        y = rewards + gamma * target * (1 - terminals)
        loss = compute_huber_loss(value, y)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: TorchObservation, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if action is None:
            return self._q_func(x).q_value
        return pick_value_by_action(self._q_func(x).q_value, action, keepdim=True)

    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        self._q_func = q_func


@dataclasses.dataclass()
class DuelingQFunctionFactory(QFunctionFactory):
    def create_discrete(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
    ) -> tuple[DuelingDiscreteQFunction, DuelingDiscreteQFunctionForwarder]:
        q_func = DuelingDiscreteQFunction(encoder=encoder, hidden_size=hidden_size, action_size=action_size)
        forwarder = DuelingDiscreteQFunctionForwarder(q_func=q_func, action_size=action_size)
        return q_func, forwarder

    def create_continuous(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
    ) -> tuple[ContinuousQFunction, ContinuousQFunctionForwarder]:
        raise NotImplementedError("DuelingQFunctionFactory currently supports only discrete action spaces.")

    @staticmethod
    def get_type() -> str:
        return "dueling"


# Ensure d3rlpy can deserialize this custom factory from saved .d3 artifacts.
register_q_func_factory(DuelingQFunctionFactory)
