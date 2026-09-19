from enum import Enum, auto
from random import Random
from typing import Protocol

from .dqn_state import GameState


class RelativeAction(Enum):
    LEFT = auto()
    FORWARD = auto()
    RIGHT = auto()


class Agent(Protocol):
    def choose_action(self, state: GameState) -> RelativeAction: ...


class RandomAgent:
    def __init__(self, rng: Random) -> None:
        self._rng = rng

    def choose_action(self, _state: GameState) -> RelativeAction:
        return self._rng.choice(tuple(RelativeAction))
