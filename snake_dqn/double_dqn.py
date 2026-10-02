import torch
from torch import nn

from .agent import RelativeAction
from .replay_buffer import ReplayBuffer


def select_actions(
    model: nn.Module, states: torch.Tensor, device: torch.device
) -> list[RelativeAction]:
    with torch.no_grad():
        action_indices = model(states.to(device)).argmax(dim=1).tolist()

    return [RelativeAction(action_index) for action_index in action_indices]


def train_step(
    model: nn.Module,
    target_model: nn.Module,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_steps: int = 1,
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
        targets = rewards + gamma**n_steps * (1 - dones) * next_q_values

    loss = loss_fn(chosen_q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()


def estimate_support_range(
    gamma: float,
    apple_reward: float,
    step_reward: float,
    death_reward: float,
    apple_interval: int,
) -> tuple[float, float]:
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if apple_interval <= 0:
        raise ValueError("apple_interval must be positive")

    gamma_T = gamma**apple_interval
    v_min = step_reward + gamma * death_reward
    v_max = apple_reward * gamma_T / (1 - gamma_T) + step_reward / (1 - gamma)
    return (v_min, v_max)
