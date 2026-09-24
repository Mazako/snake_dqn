from enum import IntEnum
from random import Random
from typing import Protocol

from .state import GameState


class RelativeAction(IntEnum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2


class Agent(Protocol):
    def choose_action(self, state: GameState) -> RelativeAction: ...


class RandomAgent:
    def __init__(self, rng: Random) -> None:
        self._rng = rng

    def choose_action(self, _state: GameState) -> RelativeAction:
        return self._rng.choice(tuple(RelativeAction))
