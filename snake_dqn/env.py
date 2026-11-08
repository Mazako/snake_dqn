from random import Random

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, OneHot, Unbounded
from torchrl.envs import EnvBase

from snake_dqn.action import RelativeAction
from snake_dqn.end_reason import EndReason

from .game import Game


class SnakeEnv(EnvBase):
    def __init__(
        self, size: int, seed: int | None = None, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([]))
        self.size = size
        self.observation_spec = Composite(
            observation=Bounded(
                low=0.0,
                high=1.0,
                shape=torch.Size([3, size, size]),
                dtype=torch.float32,
                device=self.device,
            ),
            score=Unbounded(
                shape=torch.Size([1]), dtype=torch.int64, device=self.device
            ),
            device=self.device,
        )
        self.action_spec = OneHot(3, device=self.device)
        self.reward_spec = Unbounded(1, device=self.device)
        self.done_spec = Composite(
            done=Categorical(
                2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
            ),
            terminated=Categorical(
                2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
            ),
            truncated=Categorical(
                2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
            ),
            device=self.device,
        )
        self.game = Game(size, seed)
        self.set_seed(seed)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        idx = tensordict["action"].nonzero().item()
        action = RelativeAction(idx)
        observation, reward, end_reason = self.game.step(action)
        terminated = end_reason in (EndReason.COLLISION, EndReason.WIN)
        truncated = end_reason is EndReason.STARVATION
        done = terminated | truncated

        return TensorDict(
            {
                "observation": observation,
                "score": torch.tensor([self.game.score]),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done]),
                "terminated": torch.tensor([terminated]),
                "truncated": torch.tensor([truncated]),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.game = Game(self.size, self.rng.getrandbits(32))
        return TensorDict(
            {
                "observation": self.game.state_img(),
                "score": torch.tensor([self.game.score]),
                **self.full_done_spec.zero(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        self.rng = Random(seed)
