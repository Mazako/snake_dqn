import torch

from .agent import RelativeAction
from .dueling_dqn import DuelingDqn
from .replay_buffer import ReplayBuffer


def select_actions(
    model: DuelingDqn, states: torch.Tensor, device: torch.device
) -> list[RelativeAction]:
    with torch.no_grad():
        action_indices = model.q_values(states.to(device)).argmax(dim=1).tolist()

    return [RelativeAction(action_index) for action_index in action_indices]


def project_distribution(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    n_atoms = support.numel()
    v_min, v_max = support[0].item(), support[-1].item()
    delta_z = (v_max - v_min) / (n_atoms - 1)

    shifted = rewards.unsqueeze(1) + discount * (1 - dones).unsqueeze(1) * support
    positions = ((shifted.clamp(v_min, v_max) - v_min) / delta_z).clamp(0, n_atoms - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    lower[(upper > 0) & (lower == upper)] -= 1
    upper[(lower < n_atoms - 1) & (lower == upper)] += 1

    projected = torch.zeros_like(next_probs)
    projected.scatter_add_(1, lower, next_probs * (upper - positions))
    projected.scatter_add_(1, upper, next_probs * (positions - lower))
    return projected


def train_step(
    model: DuelingDqn,
    target_model: DuelingDqn,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_steps: int = 1,
) -> float | None:
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = (
        tensor.to(device) for tensor in buffer.sample(batch_size)
    )
    batch_indices = torch.arange(batch_size, device=device)

    log_probs = model(states)[batch_indices, actions]

    with torch.no_grad():
        next_actions = model.q_values(next_states).argmax(dim=1)
        next_probs = target_model(next_states)[batch_indices, next_actions].exp()
        target_probs = project_distribution(
            next_probs,
            rewards,
            dones,
            model.get_buffer("support"),
            gamma**n_steps,
        )

    loss = -(target_probs * log_probs).sum(dim=1).mean()
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
    v_min = death_reward
    v_max = apple_reward * gamma_T / (1 - gamma_T) + step_reward / (1 - gamma)
    return (v_min, v_max)
