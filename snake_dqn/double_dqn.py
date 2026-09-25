from random import Random

import torch
from torch import nn

from .agent import RelativeAction
from .replay_buffer import ReplayBuffer


class DoubleDqn11x11(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )


def select_actions(
    model: DoubleDqn11x11,
    states: torch.Tensor,
    epsilon: float,
    rng: Random,
    device: torch.device,
) -> list[RelativeAction]:
    if epsilon == 1.0:
        return [rng.choice(tuple(RelativeAction)) for _ in states]

    with torch.no_grad():
        action_indices = model(states.to(device)).argmax(dim=1).tolist()

    return [
        rng.choice(tuple(RelativeAction))
        if rng.random() < epsilon
        else RelativeAction(action_index)
        for action_index in action_indices
    ]


def train_step(
    model: DoubleDqn11x11,
    target_model: DoubleDqn11x11,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float | None:
    if len(buffer) < batch_size:
        return None

    states, actions_batch, rewards, next_states, dones = (
        tensor.to(device) for tensor in buffer.sample(batch_size)
    )

    chosen_q_values = model(states).gather(1, actions_batch.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + gamma * (1 - dones) * next_q_values

    loss = loss_fn(chosen_q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()
