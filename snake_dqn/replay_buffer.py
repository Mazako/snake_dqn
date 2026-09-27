from collections import deque
from dataclasses import dataclass
from random import Random

import torch

from .agent import RelativeAction


@dataclass(frozen=True, slots=True)
class Transition:
    state: torch.Tensor
    action: RelativeAction
    reward: float
    next_state: torch.Tensor
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
            torch.stack([transition.state for transition in batch]),
            torch.tensor(
                [transition.action.value for transition in batch],
                dtype=torch.long,
            ),
            torch.tensor(
                [transition.reward for transition in batch],
                dtype=torch.float32,
            ),
            torch.stack([transition.next_state for transition in batch]),
            torch.tensor(
                [transition.done for transition in batch],
                dtype=torch.float32,
            ),
        )

    def __len__(self) -> int:
        return len(self._transitions)


class NStepReplayBuffer(ReplayBuffer):
    def __init__(
        self, capacity: int, rng: Random, num_envs: int, n_steps: int, gamma: float
    ) -> None:
        super().__init__(capacity, rng)
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        self.n_steps = n_steps
        self.gamma = gamma
        self._pending = [deque[Transition]() for _ in range(num_envs)]

    def append(self, transition: Transition, env_index: int = 0) -> None:
        pending = self._pending[env_index]
        pending.append(transition)

        if not transition.done and len(pending) < self.n_steps:
            return

        count = len(pending) if transition.done else 1
        for _ in range(count):
            window = list(pending)[: self.n_steps]
            reward = sum(
                self.gamma**step * item.reward for step, item in enumerate(window)
            )
            first, last = window[0], window[-1]
            super().append(
                Transition(
                    first.state, first.action, reward, last.next_state, last.done
                )
            )
            pending.popleft()
