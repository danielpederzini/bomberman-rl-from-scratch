"""Gymnasium-style Bomberman environment for a single RL agent.

Example
-------
>>> from bomberman import BombermanEnv
>>> env = BombermanEnv(render_mode=None)
>>> observation, info = env.reset(seed=0)
>>> observation, reward, terminated, truncated, info = env.step(0)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bomberman.actions import NUM_ACTIONS, Action
from bomberman.enemies import EnemyController
from bomberman.entities import GameConfig, Tile
from bomberman.game import Game
from bomberman.observation import ObservationBuilder

class BombermanEnv(gym.Env):
    """Single-agent Bomberman with scripted enemies.

    Observation: flat float32 vector (ray-cast features + scalars).
    Action: Discrete(6) - UP, DOWN, LEFT, RIGHT, BOMB, WAIT.
    """

    metadata = {"render_modes": [None, "human", "ansi"], "render_fps": 8}

    def __init__(
        self,
        config: GameConfig | None = None,
        render_mode: str | None = None,
        show_state: bool = False,
        render_fps: int = 8,
        network = None,
    ):
        super().__init__()
        self.config = config or GameConfig()
        self.render_mode = render_mode
        self.show_state = show_state
        self.render_fps = render_fps
        self.network = network

        self._rng = np.random.default_rng(self.config.seed)
        self.game = Game(self.config, rng=self._rng)
        self.obs_builder = ObservationBuilder(self.game)
        self.enemy_controller = EnemyController(self._rng)
        self._prev_potential = 0.0
        self._recent_bomb_cells: list[tuple[tuple[int, int], int]] = []

        self._renderer = None

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for reproducible episode generation.
            options: Additional reset options (unused).

        Returns:
            Tuple of (observation, info) for the initial state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.game.rng = self._rng
            self.enemy_controller = EnemyController(self._rng)
        self.game.reset()
        self.obs_builder = ObservationBuilder(self.game)
        self._prev_potential = self._potential()
        self._recent_bomb_cells = []

        observation = self.obs_builder.features()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int):
        """Execute one game tick with the given agent action.

        Args:
            action: Integer action index (UP, DOWN, LEFT, RIGHT, BOMB, WAIT).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        agent_action = Action(int(action))

        player_actions = {0: agent_action}
        player_actions.update(self.enemy_controller.actions(self.game))

        agent_alive_before = self.game.agent.alive
        agent_bombs_before = self.game.agent.active_bombs
        agent_pos_before = self.game.agent.pos
        tick_result = self.game.tick(player_actions)

        bombs_detonated = max(0, agent_bombs_before - self.game.agent.active_bombs)

        reward = self._compute_reward(
            tick_result, agent_alive_before, agent_action, agent_bombs_before,
            agent_pos_before, bombs_detonated,
        )

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

    def _compute_reward(
        self,
        result,
        agent_alive_before: bool,
        agent_action: Action | None = None,
        agent_bombs_before: int = 0,
        agent_pos_before: tuple[int, int] | None = None,
        bombs_detonated: int = 0,
    ) -> float:
        """Compute total reward as sum of individual shaping components."""
        reward = self.config.reward_step
        reward += self._compute_outcome_rewards(result)
        reward += self._compute_bomb_placement_bonus(agent_action, agent_bombs_before)
        reward += self._compute_useless_bomb_penalty(result, bombs_detonated)
        reward += self._compute_suicide_bomb_penalty(agent_action, agent_bombs_before)
        reward += self._compute_terminal_reward(agent_alive_before)
        reward += self._compute_potential_shaping()
        reward += self._compute_idle_penalty(agent_action, agent_pos_before)
        reward += self._compute_safety_reward(agent_alive_before, agent_pos_before)
        return float(reward)

    def _compute_useless_bomb_penalty(self, result, bombs_detonated: int) -> float:
        """Penalize agent bombs that detonate without destroying a crate or scoring a kill."""
        if bombs_detonated <= 0:
            return 0.0
        crates = result.crates_destroyed.get(0, 0)
        kills = len(result.kills.get(0, []))
        if crates == 0 and kills == 0:
            return self.config.reward_useless_bomb * bombs_detonated
        return 0.0

    def _compute_suicide_bomb_penalty(self, agent_action: Action | None, agent_bombs_before: int) -> float:
        """Penalize placing a bomb the agent cannot possibly escape from.

        A bomb is "inescapable" if, starting from the bomb cell, the agent
        cannot reach any tile outside the bomb's blast within the moves
        available before the fuse runs out.
        """
        placed_bomb = (
            agent_action == Action.BOMB
            and self.game.agent.active_bombs > agent_bombs_before
        )
        if not placed_bomb:
            return 0.0

        bomb_pos = self.game.agent.pos
        bomb = self.game.bomb_at(bomb_pos)
        if bomb is None:
            return 0.0
        if self._bomb_has_escape(bomb_pos, bomb.bomb_range):
            return 0.0
        return self.config.reward_suicide_bomb

    def _bomb_has_escape(self, bomb_pos: tuple[int, int], bomb_range: int) -> bool:
        """Whether a safe (out-of-blast) cell is reachable before the bomb detonates.

        BFS over walkable empty tiles from the bomb cell, bounded by the number
        of moves the agent gets before the fuse expires (bomb_fuse - 1, since the
        placement tick consumes the BOMB action).
        """
        game = self.game
        blast_cells = self._get_blast_cells(game, bomb_pos, bomb_range)
        max_moves = max(0, game.config.bomb_fuse - 1)

        visited = {bomb_pos}
        queue = deque([(bomb_pos, 0)])
        while queue:
            (row, column), distance = queue.popleft()
            if distance > 0 and (row, column) not in blast_cells:
                return True
            if distance >= max_moves:
                continue
            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + row_delta, column + column_delta)
                if neighbor in visited or not game.in_bounds(neighbor):
                    continue
                if game.tile_at(neighbor) != Tile.EMPTY or game.bomb_at(neighbor) is not None:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return False

    def _compute_outcome_rewards(self, result) -> float:
        """Compute rewards for crates destroyed and enemies killed."""
        reward = 0.0
        reward += self.config.reward_crate * result.crates_destroyed.get(0, 0)
        reward += self.config.reward_kill_enemy * len(result.kills.get(0, []))
        return reward

    def _compute_bomb_placement_bonus(self, agent_action: Action | None, agent_bombs_before: int) -> float:
        """Bonus for a bomb that threatens a target, penalty for a wasted bomb."""
        placed_bomb = (
            agent_action == Action.BOMB
            and self.game.agent.active_bombs > agent_bombs_before
        )
        if not placed_bomb:
            return 0.0

        bomb_pos = self.game.agent.pos
        now = self.game.step_count
        is_loop = self._is_bomb_loop(bomb_pos, now)
        self._record_bomb_placement(bomb_pos, now)

        if is_loop:
            return self.config.reward_bomb_spam
        if self._bomb_threatens_target():
            return self.config.reward_bomb_target
        return self.config.reward_bomb_spam

    def _is_bomb_loop(self, bomb_pos: tuple[int, int], now: int) -> bool:
        """True if a bomb was recently placed near this cell (bomb/return loop)."""
        radius = self.config.bomb_loop_radius
        window = self.config.bomb_loop_window
        for (prev_pos, prev_step) in self._recent_bomb_cells:
            if now - prev_step > window:
                continue
            manhattan = abs(prev_pos[0] - bomb_pos[0]) + abs(prev_pos[1] - bomb_pos[1])
            if manhattan <= radius:
                return True
        return False

    def _record_bomb_placement(self, bomb_pos: tuple[int, int], now: int) -> None:
        """Record a bomb placement and prune entries outside the loop window."""
        window = self.config.bomb_loop_window
        self._recent_bomb_cells.append((bomb_pos, now))
        self._recent_bomb_cells = [
            (pos, step) for (pos, step) in self._recent_bomb_cells
            if now - step <= window
        ]

    def _compute_terminal_reward(self, agent_alive_before: bool) -> float:
        """Compute reward for terminal outcomes (death or win)."""
        agent = self.game.agent
        if agent_alive_before and not agent.alive:
            return self.config.reward_death
        if agent.alive and self.config.n_enemies > 0 and not any(e.alive for e in self.game.enemies):
            return self.config.reward_win
        return 0.0

    def _compute_potential_shaping(self) -> float:
        """Compute potential-based shaping reward toward nearest target."""
        new_potential = self._potential()
        reward = self.config.shaping_gamma * new_potential - self._prev_potential
        self._prev_potential = new_potential
        return reward

    def _compute_idle_penalty(self, agent_action: Action | None, agent_pos_before: tuple[int, int] | None) -> float:
        """Compute penalty for waiting or blocked moves when safe."""
        agent = self.game.agent
        if not agent.alive or agent_action is None:
            return 0.0

        in_danger = agent.pos in self.game.predict_danger_cells()
        blocked_move = (
            agent_action.is_move
            and agent_pos_before is not None
            and agent.pos == agent_pos_before
        )
        if (agent_action == Action.WAIT or blocked_move) and not in_danger:
            return self.config.reward_idle
        return 0.0

    def _compute_safety_reward(self, agent_alive_before: bool, agent_pos_before: tuple[int, int] | None) -> float:
        """Compute reward for escaping danger and penalty for walking into it."""
        agent = self.game.agent
        if not agent_alive_before or not agent.alive or agent_pos_before is None:
            return 0.0

        was_in_danger = agent_pos_before in self.game.predict_danger_cells()
        now_in_danger = agent.pos in self.game.predict_danger_cells()
        if was_in_danger and not now_in_danger:
            return self.config.reward_escape_danger
        if not was_in_danger and now_in_danger:
            return self.config.reward_enter_danger
        return 0.0

    def _potential(self) -> float:
        """Potential = -progress_weight * BFS distance to the nearest target.

        Closer to 0 when the agent is near a crate/enemy, so moving closer
        yields positive potential-based shaping and retreating yields negative.
        """
        distance = self._nearest_target_distance()
        if distance is None:
            return 0.0
        return -self.config.progress_weight * float(distance)

    def _nearest_target_distance(self) -> int | None:
        """BFS distance (over empty, bomb-free tiles) from the agent to the
        nearest tile adjacent to a crate or alive enemy. None if unreachable."""
        game = self.game
        agent = game.agent
        if not agent.alive:
            return None

        targets = self._build_target_set(game)
        if not targets:
            return None

        start = agent.pos
        agent_row, agent_column = start

        if start in targets:
            return 0
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (agent_row + row_delta, agent_column + column_delta) in targets:
                return 1

        return self._breadth_first_search_distance(game, start, targets)

    def _build_target_set(self, game: Game) -> set[Pos]:
        """Build set of target positions (enemies + crates)."""
        targets = {enemy.pos for enemy in game.enemies if enemy.alive}
        for row in range(game.config.height):
            for column in range(game.config.width):
                if game.tile_at((row, column)) == Tile.CRATE:
                    targets.add((row, column))
        return targets

    def _breadth_first_search_distance(
        self,
        game: Game,
        start: Pos,
        targets: set[Pos],
    ) -> int | None:
        """BFS to find shortest distance to any target with early termination."""
        agent_row, agent_column = start
        visited = {start}
        queue = deque([(start, 0)])

        best_manhattan = min(
            abs(target_row - agent_row) + abs(target_column - agent_column)
            for target_row, target_column in targets
        )

        while queue:
            (row, column), distance = queue.popleft()
            if distance > best_manhattan + 2:
                continue

            next_distance = distance + 1
            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_row = row + row_delta
                neighbor_column = column + column_delta
                neighbor = (neighbor_row, neighbor_column)
                if neighbor in targets:
                    return distance
                if neighbor in visited or not game.in_bounds(neighbor):
                    continue
                if game.tile_at(neighbor) != Tile.EMPTY or game.bomb_at(neighbor) is not None:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, next_distance))
        return None



    def _bomb_threatens_target(self) -> bool:
        """Check if agent's just-placed bomb would hit a crate or enemy."""
        game = self.game
        agent = game.agent
        bomb = game.bomb_at(agent.pos)
        if bomb is None:
            return False

        enemy_positions = {enemy.pos for enemy in game.enemies if enemy.alive}
        blast_cells = self._get_blast_cells(game, agent.pos, bomb.bomb_range)

        for cell in blast_cells:
            tile = game.tile_at(cell)
            if tile == Tile.CRATE:
                return True
            if cell in enemy_positions:
                return True
        return False

    def _get_blast_cells(self, game: Game, origin: Pos, bomb_range: int) -> set[Pos]:
        """Calculate all cells affected by a bomb explosion at the given origin."""
        blast_area: set[Pos] = {origin}
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for distance in range(1, bomb_range + 1):
                cell = (
                    origin[0] + row_offset * distance,
                    origin[1] + column_offset * distance,
                )
                if not game.in_bounds(cell):
                    break
                tile = game.tile_at(cell)
                if tile == Tile.WALL:
                    break
                blast_area.add(cell)
                if tile == Tile.CRATE:
                    break
        return blast_area



    def _info(self) -> dict[str, Any]:
        """Return environment state info for debugging.

        Contains:
        - raw_state: Full observation vector
        - step: Current tick count
        - agent_alive: Whether agent is alive
        - agent_pos: Current agent coordinates (for stagnation detection)
        - bomb_count: Active bombs placed by agent (for bomb placement bonus)
        - enemies_alive: Count of surviving enemies
        """
        return {
            "raw_state": self.obs_builder.raw_state(),
            "step": self.game.step_count,
            "agent_alive": self.game.agent.alive,
            "agent_pos": self.game.agent.pos,
            "bomb_count": self.game.agent.active_bombs,
            "enemies_alive": sum(1 for enemy in self.game.enemies if enemy.alive),
        }

    def render(self):
        """Render the current game state in the configured mode.

        Returns:
            String representation for "ansi" mode, None otherwise.
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _render_ansi(self) -> str:
        """Generate ASCII text representation of the game board."""
        from bomberman.render import render_ascii
        return render_ascii(self.game)



    def get_valid_actions(self) -> list[int]:
        """Return list of valid action indices for the current state.

        BOMB is valid if the agent can place a bomb (has capacity).
        MOVE actions are valid if the target cell is walkable.
        WAIT is always valid.
        """
        from bomberman.actions import Action

        valid = [Action.WAIT.value]  # WAIT is always valid
        agent = self.game.agent

        if not agent.alive:
            return valid

        for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            row_delta, column_delta = action.delta
            target = (agent.pos[0] + row_delta, agent.pos[1] + column_delta)
            if self._is_move_valid(agent, target):
                valid.append(action.value)

        if agent.can_place_bomb():
            valid.append(Action.BOMB.value)

        return valid

    def _is_move_valid(self, agent, target: Pos) -> bool:
        """Check if a movement to the target position is valid."""
        return self.game.is_walkable(target, mover_pid=agent.pid)

    def _render_human(self) -> None:
        """Render the game using Pygame for visual display."""
        from bomberman.render import PygameRenderer
        if self._renderer is None:
            self._renderer = PygameRenderer(self.config, show_state=self.show_state, fps=self.render_fps)
        self._renderer.draw(self.game, self.obs_builder, self.network)

    def close(self):
        """Close the environment and cleanup resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

