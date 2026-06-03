"""Gymnasium-style Bomberman environment for a single RL agent.

Example
-------
>>> from bomberman import BombermanEnv
>>> env = BombermanEnv(render_mode=None)
>>> observation, info = env.reset(seed=0)
>>> observation, reward, terminated, truncated, info = env.step(0)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except Exception:  # pragma: no cover - gymnasium should be installed
    gym = None
    spaces = None
    _GYM_BASE = object

from bomberman.actions import NUM_ACTIONS, Action
from bomberman.enemies import EnemyController
from bomberman.entities import GameConfig
from bomberman.game import Game
from bomberman.observation import ObservationBuilder


class BombermanEnv(_GYM_BASE):
    """Single-agent Bomberman with scripted enemies.

    Observation: flat float32 vector (ray-cast features + scalars).
    Action: Discrete(6) - UP, DOWN, LEFT, RIGHT, BOMB, WAIT.
    """

    metadata = {"render_modes": [None, "human", "ansi"], "render_fps": 8}

    def __init__(
        self,
        config: GameConfig | None = None,
        render_mode: str | None = None,
        n_ray_dirs: int = 4,
        show_state: bool = False,
    ):
        super().__init__()
        self.config = config or GameConfig()
        self.render_mode = render_mode
        self.n_ray_dirs = n_ray_dirs
        self.show_state = show_state

        self._rng = np.random.default_rng(self.config.seed)
        self.game = Game(self.config, rng=self._rng)
        self.obs_builder = ObservationBuilder(self.game, n_ray_dirs=n_ray_dirs)
        self.enemy_controller = EnemyController(self._rng)

        self._renderer = None  # lazily created for "human" mode

        if spaces is not None:
            self.action_space = spaces.Discrete(NUM_ACTIONS)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32
            )

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.game.rng = self._rng
            self.enemy_controller = EnemyController(self._rng)
        self.game.reset()
        self.obs_builder = ObservationBuilder(self.game, n_ray_dirs=self.n_ray_dirs)

        observation = self.obs_builder.features()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int):
        agent_action = Action(int(action))

        # Gather all players' actions for this tick.
        player_actions = {0: agent_action}
        player_actions.update(self.enemy_controller.actions(self.game))

        agent_alive_before = self.game.agent.alive
        tick_result = self.game.tick(player_actions)

        reward = self._compute_reward(tick_result, agent_alive_before)

        # A real terminal outcome is a win (no enemies left) or a loss (agent
        # dead). Hitting the step budget without a winner is a truncation.
        agent_alive = self.game.agent.alive
        enemies_alive = any(enemy.alive for enemy in self.game.enemies)
        terminated = (not agent_alive) or (self.config.n_enemies > 0 and not enemies_alive)
        truncated = self.game.done and not terminated

        observation = self.obs_builder.features()
        info = self._info()
        info["tick"] = {
            "crates_destroyed": tick_result.crates_destroyed,
            "kills": tick_result.kills,
            "deaths": tick_result.deaths,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _compute_reward(self, result, agent_alive_before: bool) -> float:
        config = self.config
        reward = config.reward_step

        reward += config.reward_crate * result.crates_destroyed.get(0, 0)
        reward += config.reward_kill_enemy * len(result.kills.get(0, []))

        agent = self.game.agent
        if agent_alive_before and not agent.alive:
            reward += config.reward_death
        elif agent.alive and not any(enemy.alive for enemy in self.game.enemies):
            # Agent is the last one standing -> win.
            reward += config.reward_win
        return float(reward)

    def _info(self) -> dict[str, Any]:
        return {
            "raw_state": self.obs_builder.raw_state(),
            "step": self.game.step_count,
            "agent_alive": self.game.agent.alive,
            "enemies_alive": sum(1 for enemy in self.game.enemies if enemy.alive),
        }

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _render_ansi(self) -> str:
        from bomberman.render import render_ascii

        text = render_ascii(self.game)
        return text

    def _render_human(self) -> None:
        from bomberman.render import PygameRenderer

        if self._renderer is None:
            self._renderer = PygameRenderer(self.config, show_state=self.show_state)
        self._renderer.draw(self.game, self.obs_builder)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
