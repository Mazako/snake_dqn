from collections import deque
from dataclasses import dataclass

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
    def __init__(
        self,
        capacity: int,
        seed: int = 42,
        beta: float = 0.4,
        priority_eps: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._priorities = torch.zeros(capacity, dtype=torch.float32)

        self._states = torch.empty((capacity, 3, 11, 11), dtype=torch.float32)
        self._actions = torch.empty(capacity, dtype=torch.int64)
        self._rewards = torch.empty(capacity, dtype=torch.float32)
        self._next_states = torch.empty((capacity, 3, 11, 11), dtype=torch.float32)
        self._dones = torch.empty(capacity, dtype=torch.float32)

        self._i = 0
        self._size = 0
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self._capacity = capacity
        self.beta = beta
        self.priority_eps = priority_eps

    def append(self, transition: Transition) -> None:
        if self._size == 0:
            prior = 1.0
        else:
            prior = self._priorities.max()
        self._priorities[self._i] = prior

        self._states[self._i, :, :, :] = transition.state
        self._next_states[self._i, :, :, :] = transition.next_state
        self._actions[self._i] = transition.action
        self._rewards[self._i] = transition.reward
        self._dones[self._i] = transition.done

        self._i = (self._i + 1) % self._capacity
        self._size = min((self._size + 1), self._capacity)

    def sample(
        self, batch_size: int, alpha: float = 0.6
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("batch_size exceeds buffer size")

        probs = self._priorities[: self._size] ** alpha
        probs /= probs.sum()

        indices = torch.multinomial(
            probs, batch_size, replacement=True, generator=self._rng
        )
        weights = (len(self) * probs[indices]) ** -self.beta

        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            weights,
            indices,
        )

    def __len__(self) -> int:
        return self._size

    def update_priorities(self, td_errors: torch.Tensor, indices: torch.Tensor) -> None:
        priors = td_errors.detach().abs().cpu().float() + self.priority_eps
        self._priorities[indices] = priors


class NStepReplayBuffer(ReplayBuffer):
    def __init__(
        self, capacity: int, num_envs: int, n_steps: int, gamma: float, seed: int = 42
    ) -> None:
        super().__init__(capacity, seed=seed)
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
