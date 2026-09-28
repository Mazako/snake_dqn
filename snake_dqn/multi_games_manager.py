from collections.abc import Sequence
from random import Random

from .agent import RelativeAction
from .game import Game
from .replay_buffer import Transition


class MultiGamesManager:
    def __init__(
        self, num_envs: int, board_size: int, max_steps: int, seed: int | None = None
    ) -> None:
        self.board_size = board_size
        self.max_steps = max_steps
        self._rng = Random(seed)
        self.games = [self._new_game() for _ in range(num_envs)]
        self.states = [game.state_img() for game in self.games]

    def _new_game(self) -> Game:
        return Game(self.board_size, self.max_steps, seed=self._rng.getrandbits(64))

    def step(
        self, actions: Sequence[RelativeAction], remaining_episodes: int
    ) -> tuple[list[Transition], list[int]]:
        transitions = []
        completed_scores = []

        for index, (game, state, action) in enumerate(
            zip(self.games, self.states, actions, strict=True)
        ):
            _, reward, done = game.step(action)
            next_state = game.state_img()
            transitions.append(Transition(state, action, reward, next_state, done))

            if done:
                completed_scores.append(game.score)
                self.games[index] = self._new_game()
                self.states[index] = self.games[index].state_img()
            else:
                self.states[index] = next_state

            if len(completed_scores) == remaining_episodes:
                break

        return transitions, completed_scores
