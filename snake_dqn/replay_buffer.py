from collections import deque
from dataclasses import dataclass
from random import Random

from .agent import RelativeAction
from .dqn_state import GameState


@dataclass(frozen=True, slots=True)
class Transition:
    state: GameState
    action: RelativeAction
    reward: float
    next_state: GameState
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: Random) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._transitions: deque[Transition] = deque(maxlen=capacity)
        self._rng = rng

    def append(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("batch_size exceeds buffer size")

        return self._rng.sample(list(self._transitions), batch_size)

    def __len__(self) -> int:
        return len(self._transitions)
