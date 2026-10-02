from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from .agent import RelativeAction
from .end_reason import EndReason
from .game import Game
from .replay_buffer import Transition


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    score: int
    end_reason: EndReason


class MultiGamesManager:
    def __init__(self, num_envs: int, board_size: int, seed: int | None = None) -> None:
        self.board_size = board_size
        self._rng = Random(seed)
        self.games = [self._new_game() for _ in range(num_envs)]
        self.states = [game.state_img() for game in self.games]

    def _new_game(self) -> Game:
        return Game(self.board_size, seed=self._rng.getrandbits(64))

    def step(
        self, actions: Sequence[RelativeAction], remaining_episodes: int
    ) -> tuple[list[Transition], list[EpisodeResult]]:
        transitions = []
        completed_episodes = []

        for index, (game, state, action) in enumerate(
            zip(self.games, self.states, actions, strict=True)
        ):
            _, reward, done = game.step(action)
            next_state = game.state_img()
            transitions.append(Transition(state, action, reward, next_state, done))

            if game.end_reason is not None:
                completed_episodes.append(EpisodeResult(game.score, game.end_reason))
                self.games[index] = self._new_game()
                self.states[index] = self.games[index].state_img()
            else:
                self.states[index] = next_state

            if len(completed_episodes) == remaining_episodes:
                break

        return transitions, completed_episodes
