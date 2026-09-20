from collections import deque
from dataclasses import dataclass
from random import Random

import torch

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

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("batch_size exceeds buffer size")

        batch = self._rng.sample(list(self._transitions), batch_size)

        return (
            torch.stack([transition.state.features() for transition in batch]),
            torch.tensor(
                [transition.action.value for transition in batch],
                dtype=torch.long,
            ),
            torch.tensor(
                [transition.reward for transition in batch],
                dtype=torch.float32,
            ),
            torch.stack([transition.next_state.features() for transition in batch]),
            torch.tensor(
                [transition.done for transition in batch],
                dtype=torch.float32,
            ),
        )

    def __len__(self) -> int:
        return len(self._transitions)
