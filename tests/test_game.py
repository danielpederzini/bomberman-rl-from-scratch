"""Tests for the core game mechanics and the gym environment.

Run with:  pytest -q     (or)     python tests/test_game.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bomberman.actions import Action  # noqa: E402
from bomberman.entities import GameConfig, Player, Tile  # noqa: E402
from bomberman.env import BombermanEnv  # noqa: E402
from bomberman.game import Game  # noqa: E402
from bomberman.observation import ObservationBuilder  # noqa: E402


def _empty_game(width=7, height=7, fuse=4, rng_seed=0) -> Game:
    """A game with only border walls (no pillars/crates) for deterministic tests."""
    config = GameConfig(
        width=width,
        height=height,
        n_enemies=0,
        bomb_fuse=fuse,
        bomb_range=2,
        crate_density=0.0,
        seed=rng_seed,
    )
    game = Game(config)
    # Wipe the interior to pure empty floor (remove generated pillars).
    game.grid[1:-1, 1:-1] = Tile.EMPTY
    game.players = [Player(pos=(1, 1), pid=0, bomb_range=2, max_bombs=1)]
    return game


def test_bomb_fuse_timing_and_explosion():
    game = _empty_game(fuse=4)
    game.tick({0: Action.BOMB})  # place at (1,1)
    assert len(game.bombs) == 1, "bomb should exist right after placement"
    # Fuse=4 => explodes on the 4th tick (placement counts as tick 1).
    game.tick({0: Action.WAIT})
    game.tick({0: Action.WAIT})
    assert len(game.bombs) == 1, "bomb should still be ticking"
    game.tick({0: Action.WAIT})
    assert len(game.bombs) == 0, "bomb should have exploded"
    assert len(game.blasts) > 0, "explosion should create blast cells"


def test_blast_blocked_by_unbreakable_wall():
    game = _empty_game()
    # Put a wall two tiles to the right of the agent, then a clear tile beyond.
    game.players = [Player(pos=(3, 3), pid=0, bomb_range=3, max_bombs=1)]
    game.grid[3, 5] = Tile.WALL
    game.tick({0: Action.BOMB})
    for _ in range(game.config.bomb_fuse - 1):
        game.tick({0: Action.WAIT})
    cells = game.blast_cells()
    assert (3, 4) in cells, "blast should reach the tile before the wall"
    assert (3, 5) not in cells, "blast must not include the wall tile"
    assert (3, 6) not in cells, "blast must not pass through the wall"


def test_crate_destroyed_and_stops_blast():
    game = _empty_game()
    game.players = [Player(pos=(3, 3), pid=0, bomb_range=3, max_bombs=1)]
    game.grid[3, 4] = Tile.CRATE
    game.grid[3, 5] = Tile.CRATE
    game.tick({0: Action.BOMB})
    for _ in range(game.config.bomb_fuse - 1):
        game.tick({0: Action.WAIT})
    assert game.grid[3, 4] == Tile.EMPTY, "first crate should be destroyed"
    assert game.grid[3, 5] == Tile.CRATE, "blast should stop after first crate"


def test_player_dies_in_blast():
    game = _empty_game()
    game.players = [Player(pos=(1, 1), pid=0, bomb_range=2, max_bombs=1)]
    game.tick({0: Action.BOMB})  # agent stands on the bomb
    for _ in range(game.config.bomb_fuse - 1):
        game.tick({0: Action.WAIT})  # never moves away
    assert not game.agent.alive, "agent standing on its own bomb should die"
    assert game.done


def test_chain_reaction():
    game = _empty_game(width=9, height=9)
    game.players = [Player(pos=(1, 1), pid=0, bomb_range=2, max_bombs=2)]
    game.agent.max_bombs = 2
    # Place bomb A at (1,1).
    game.tick({0: Action.BOMB})
    # Move right and place bomb B at (1,2), within A's range.
    game.tick({0: Action.RIGHT})
    game.tick({0: Action.BOMB})
    # Run enough ticks for A to explode and chain B.
    for _ in range(game.config.bomb_fuse):
        game.tick({0: Action.WAIT})
    assert len(game.bombs) == 0, "both bombs should have detonated via chain reaction"


def test_env_random_episode_runs():
    config = GameConfig(seed=123, max_steps=200)
    env = BombermanEnv(config=config, render_mode=None)
    observation, info = env.reset(seed=123)
    rng = np.random.default_rng(0)
    steps = 0
    done = False
    while not done and steps < 1000:
        action = int(rng.integers(env.action_space.n))
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation.shape == (env.obs_builder.size,)
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))
        done = terminated or truncated
        steps += 1
    assert done, "episode should terminate within the step budget"


def test_observation_shape_stable():
    config = GameConfig(seed=1)
    game = Game(config)
    four_direction_observation_builder = ObservationBuilder(game, n_ray_dirs=4)
    eight_direction_observation_builder = ObservationBuilder(game, n_ray_dirs=8)
    assert four_direction_observation_builder.features().shape == (
        four_direction_observation_builder.size,
    )
    assert eight_direction_observation_builder.features().shape == (
        eight_direction_observation_builder.size,
    )
    assert eight_direction_observation_builder.size > four_direction_observation_builder.size
    # Values are within the declared Box bounds.
    features = four_direction_observation_builder.features()
    assert features.min() >= 0.0 and features.max() <= 1.0


def _enemy_bomb_game(enabled: bool) -> Game:
    config = GameConfig(
        width=7,
        height=7,
        n_enemies=1,
        crate_density=0.0,
        enemies_drop_bombs=enabled,
        enemy_bomb_prob=1.0,
        seed=0,
    )
    game = Game(config)
    game.grid[1:-1, 1:-1] = Tile.EMPTY
    # Agent far away; enemy next to a crate with walkable escape neighbours.
    game.players = [
        Player(pos=(5, 5), pid=0, bomb_range=2),
        Player(pos=(3, 3), pid=1, bomb_range=2),
    ]
    game.grid[3, 4] = Tile.CRATE
    return game


def test_enemy_drops_bomb_when_enabled():
    from bomberman.enemies import EnemyController

    game = _enemy_bomb_game(enabled=True)
    enemy_controller = EnemyController(np.random.default_rng(0))
    enemy_actions = enemy_controller.actions(game)
    assert enemy_actions[1] == Action.BOMB, "enemy should bomb the adjacent crate"


def test_enemy_never_bombs_when_disabled():
    from bomberman.enemies import EnemyController

    game = _enemy_bomb_game(enabled=False)
    enemy_controller = EnemyController(np.random.default_rng(0))
    # Sample many ticks; with the flag off, BOMB must never be chosen.
    for _ in range(50):
        enemy_actions = enemy_controller.actions(game)
        assert enemy_actions[1] != Action.BOMB


def _run_all():
    test_functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test_function in test_functions:
        test_function()
        print(f"PASS {test_function.__name__}")
    print(f"\nAll {len(test_functions)} tests passed.")


if __name__ == "__main__":
    _run_all()
