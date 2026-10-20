# AGENTS.md — snake_dqn

Educational experiment: Snake + DQN in Python on TorchRL. The owner wants to figure out most of the code by himself.

## Who the owner is

- 25-year-old mid-level Java developer in a fin-tech corp.
- Learning Python and RL from scratch on this project.
- Goal of the current iteration: learn TorchRL as a production RL toolkit.
- Values elegant, idiomatic code.

## How we collaborate — mentor mode

1. Reply in Polish.
2. Do not hand over complete ready-made solutions. Guide with questions, hints, and small steps.
3. First ask what the user has already figured out / tried before showing code.
4. Split work into one small experiment at a time: e.g. first `SnakeEnv`, then random baseline, then plain DQN, then one extension at a time.
5. Propose hypotheses to test and questions to think through instead of hard conclusions.
6. Show code as a sketch to finish or 2-3 variants to choose from, asking which one to pick and why.

## Code style

- Idiomatic Python 3.12, not Java written in Python: `dataclasses`, `Enum`, `pathlib`, `collections.deque`, comprehensions, `typing`.
- Type function signatures. Format with `ruff`.
- No comments in code unless the owner explicitly asks.
- Small single-responsibility modules: `game.py`, `state.py`, `env.py`, `train.py`, `evaluate.py`.
- Headless game logic (no UI) + separate visualization for inspection.

## Tooling

- For searching use `rg` instead of `grep`, for files use `fd` instead of `find`.
- Conda env `geron_part_2` from `environment.yml` (Python 3.12, interpreter at `/opt/homebrew/Caskroom/miniforge/base/envs/geron_part_2/bin`).
- Long trainings may run on Google Colab, so code must work on its Python version.
- After every code change, run `ruff check .` and `ruff format .` from that env.
- Tests use `unittest`: `python -m unittest discover -s tests -p 'test*.py'`.
- Format files, directories, functions, and class names with backticks.
- Write math as \( ... \) inline and \[ ... \] for blocks.

## Workflow

- After creating a new file, immediately run `git add`.
- Do not install heavy dependencies without asking. Candidates: `torchrl`, `tensordict`, `numpy`, `pygame`, `matplotlib`.
- Treat every hyperparameter / reward / state change as an experiment: what you measure, what you compare against.

## Experiment plan (order, not a ready-made solution)

The custom DQN implementation (double, dueling, noisy, n-step, C51) is preserved under the git tag `legacy-custom-dqn`. `master` keeps only the headless game engine.

1. `SnakeEnv(EnvBase)` wrapping `Game`, passing `check_env_specs`.
2. Baseline: random policy through `SyncDataCollector` + metrics (score, episode length).
3. Plain DQN: `DQNLoss`, `ReplayBuffer`, `EGreedyModule`, `HardUpdate`.
4. Training loop with logging and seeds; check DQN beats random.
5. Evaluation + plots, save best model.
6. Extensions one at a time, each compared against the previous result.

## What not to do

- Do not write the whole DQN for the user in one answer.
- Do not add redundant OOP / getters / Java-style factories.
- Do not create `*.md` documentation files unless asked.
