from collections.abc import Sequence

from .agent import RelativeAction
from .game import Game
from .replay_buffer import Transition


class MultiGamesManager:
    def __init__(self, num_envs: int, board_size: int, max_steps: int) -> None:
        self.board_size = board_size
        self.max_steps = max_steps
        self.games = [Game(board_size, max_steps) for _ in range(num_envs)]
        self.states = [game.state_img() for game in self.games]

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
                self.games[index] = Game(self.board_size, self.max_steps)
                self.states[index] = self.games[index].state_img()
            else:
                self.states[index] = next_state

            if len(completed_scores) == remaining_episodes:
                break

        return transitions, completed_scores
