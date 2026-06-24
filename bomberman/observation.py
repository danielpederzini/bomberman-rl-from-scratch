"""Observation extraction for the RL agent.

Two things are exposed:

* :meth:`ObservationBuilder.features` - a flat ``np.float32`` vector built from
  an agent-centered ``CROP_SIZE x CROP_SIZE`` grid (5 channels) plus scalar
  features. The agent sits at the center of the crop and out-of-bounds cells are
  encoded as walls.
  Grid channels:
      ch0 - tile:        0=empty, 0.5=crate, 1=wall (out-of-bounds=1)
      ch1 - danger:      1 if in predicted blast range, else 0
      ch2 - blast:       1 if active blast present, else 0
      ch3 - bomb_imminence: 0=no bomb, 1=detonating next tick
      ch4 - entity:      0=empty, 0.5=enemy, 1=agent
  Spatial size: CROP_SIZE^2 * 5 (CROP_SIZE=9 -> 9*9*5 = 405).  Scalars: 12
  (7 base + safe_up/down/left/right + is_trapped).
* :meth:`ObservationBuilder.raw_state` - a dict of ground-truth game state.
"""

from __future__ import annotations

import numpy as np

from bomberman.entities import Tile
from bomberman.game import Game

_N_CHANNELS = 5
CROP_SIZE = 9
CROP_RADIUS = CROP_SIZE // 2
SCALAR_NAMES = [
    "on_danger",
    "can_bomb",
    "near_enemy",
    "enemies_alive",
    "agent_row",
    "agent_col",
    "min_fuse",
    "safe_up",
    "safe_down",
    "safe_left",
    "safe_right",
    "is_trapped",
]
_CARDINAL_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class ObservationBuilder:
    """Build observation vectors from game state for the RL agent."""

    def __init__(self, game: Game, n_ray_dirs: int | None = None):
        """Initialize with game reference and optional ray casting configuration.

        Args:
            game: The Bomberman game instance to observe.
            n_ray_dirs: Number of ray directions for ray casting (4, 8, or None).
        """
        self.game = game
        self.max_dim = max(game.config.width, game.config.height)
        self.n_ray_dirs = n_ray_dirs

    @property
    def board_shape(self) -> tuple[int, int, int]:
        """Return the shape of the egocentric observation grid (size, size, channels)."""
        return (CROP_SIZE, CROP_SIZE, _N_CHANNELS)

    @property
    def spatial_size(self) -> int:
        """Return the total size of the flattened spatial grid."""
        height, width, channels = self.board_shape
        return height * width * channels

    @property
    def size(self) -> int:
        """Return the total observation vector size (spatial + scalars)."""
        return self.spatial_size + len(self.scalar_names())

    def features(self) -> np.ndarray:
        """Build and return the full observation vector for the current game state.

        The observation consists of:
        - Spatial features: Flattened height x width x 5 grid encoding tiles,
          danger zones, active blasts, bomb timers, and entity positions
        - Scalar features: 7 (or more with rays) normalized metrics like
          agent position, enemy proximity, and bomb readiness

        Returns:
            Flat float32 vector of length self.size.
        """
        game = self.game
        agent = game.agent

        if not agent.alive:
            return np.zeros(self.size, dtype=np.float32)

        spatial = self._egocentric_grid()
        scalars = np.asarray(self._scalar_features(), dtype=np.float32)
        return np.concatenate([spatial, scalars])

    def _egocentric_grid(self) -> np.ndarray:
        """Build a flattened agent-centered (size x size x 5) crop of the board.

        The agent is always at the center of the crop. Cells outside the board
        are encoded as walls (channel 0 = 1.0, all other channels 0.0).

        Channels:
        - 0: Tile type (0=empty, 0.5=crate, 1=wall; out-of-bounds=1)
        - 1: Danger flag (1 if in blast range)
        - 2: Active blast flag (1 if explosion present)
        - 3: Bomb imminence (0=no bomb, rising to 1=detonating next tick)
        - 4: Entity (0=empty, 0.5=enemy, 1=agent)
        """
        game = self.game
        size = CROP_SIZE

        danger_cells = game.predict_danger_cells()
        blast_cells = game.blast_cells()
        fuse = max(1, game.config.bomb_fuse)
        bomb_map = {
            bomb.pos: (fuse - bomb.timer + 1) / fuse
            for bomb in game.bombs
        }
        enemy_positions = {enemy.pos for enemy in game.enemies if enemy.alive}
        agent_row, agent_column = game.agent.pos

        grid = np.zeros((size, size, _N_CHANNELS), dtype=np.float32)

        for row_offset in range(-CROP_RADIUS, size - CROP_RADIUS):
            for column_offset in range(-CROP_RADIUS, size - CROP_RADIUS):
                grid_row = row_offset + CROP_RADIUS
                grid_column = column_offset + CROP_RADIUS
                position = (agent_row + row_offset, agent_column + column_offset)

                if not game.in_bounds(position):
                    grid[grid_row, grid_column, 0] = 1.0  # out-of-bounds = wall
                    continue

                tile = game.tile_at(position)
                if tile == Tile.WALL:
                    grid[grid_row, grid_column, 0] = 1.0
                elif tile == Tile.CRATE:
                    grid[grid_row, grid_column, 0] = 0.5

                grid[grid_row, grid_column, 1] = 1.0 if position in danger_cells else 0.0
                grid[grid_row, grid_column, 2] = 1.0 if position in blast_cells else 0.0
                grid[grid_row, grid_column, 3] = bomb_map.get(position, 0.0)
                if position == (agent_row, agent_column):
                    grid[grid_row, grid_column, 4] = 1.0
                elif position in enemy_positions:
                    grid[grid_row, grid_column, 4] = 0.5

        return grid.flatten()

    def scalar_names(self) -> list[str]:
        """Return list of scalar feature names.

        Includes base 7 features plus ray casting features if enabled.

        Returns:
            List of feature name strings.
        """
        names = list(SCALAR_NAMES)
        if self.n_ray_dirs is None or self.n_ray_dirs <= 0:
            return names
        for ray_idx in range(self.n_ray_dirs):
            names.extend([f"ray_{ray_idx}_dist", f"ray_{ray_idx}_type"])
        return names

    def _ray_directions(self) -> list[tuple[int, int]]:
        """Calculate direction vectors for ray casting.

        Returns:
            List of (row_delta, column_delta) tuples for each ray direction.
        """
        if self.n_ray_dirs is None or self.n_ray_dirs <= 0:
            return []
        if self.n_ray_dirs == 4:
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.n_ray_dirs == 8:
            return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        directions = []
        for ray_idx in range(self.n_ray_dirs):
            angle = (2 * np.pi * ray_idx) / self.n_ray_dirs
            directions.append((int(round(np.sin(angle))), int(round(np.cos(angle)))))
        return directions

    def ray_casts(self) -> list[dict[str, object]]:
        """Cast rays in all directions to detect walls, crates, bombs, and enemies.

        Returns:
            List of ray hit dictionaries containing position, normalized distance,
            and hit type for each direction.
        """
        if self.n_ray_dirs is None or self.n_ray_dirs <= 0:
            return []
        if not self.game.agent.alive:
            return []

        agent_pos = self.game.agent.pos
        rays: list[dict[str, object]] = []
        for row_delta, column_delta in self._ray_directions():
            hit_pos, hit_type = self._cast_single_ray(agent_pos, row_delta, column_delta)
            distance = abs(hit_pos[0] - agent_pos[0]) + abs(hit_pos[1] - agent_pos[1])
            distance_norm = distance / max(1, self.max_dim)
            hit_value = {
                "empty": 0.0,
                "crate": 0.25,
                "bomb": 0.5,
                "enemy": 0.75,
                "wall": 1.0,
            }[hit_type]
            rays.append(
                {
                    "hit_pos": hit_pos,
                    "distance_norm": distance_norm,
                    "hit_type": hit_value,
                    "hit_type_name": hit_type,
                }
            )
        return rays

    def _cast_single_ray(
        self,
        origin: tuple[int, int],
        row_delta: int,
        column_delta: int,
    ) -> tuple[tuple[int, int], str]:
        """Cast a single ray in a direction and return what it hits.

        Args:
            origin: Starting position of the ray.
            row_delta: Row direction offset (-1, 0, or 1).
            column_delta: Column direction offset (-1, 0, or 1).

        Returns:
            Tuple of (hit_position, hit_type) where hit_type is one of:
            "empty", "wall", "crate", "bomb", "enemy".
        """
        hit_pos: tuple[int, int] | None = None
        hit_type = "empty"

        for step in range(1, self.max_dim + 1):
            candidate = (
                origin[0] + row_delta * step,
                origin[1] + column_delta * step,
            )
            if not self.game.in_bounds(candidate):
                break

            tile = self.game.tile_at(candidate)
            if tile == Tile.WALL:
                hit_pos = candidate
                hit_type = "wall"
                break
            if tile == Tile.CRATE:
                hit_pos = candidate
                hit_type = "crate"
                break
            if self.game.bomb_at(candidate) is not None:
                hit_pos = candidate
                hit_type = "bomb"
                break
            if self.game.alive_player_at(candidate, exclude_pid=self.game.agent.pid) is not None:
                hit_pos = candidate
                hit_type = "enemy"
                break
            hit_pos = candidate

        if hit_pos is None:
            hit_pos = origin
        return hit_pos, hit_type

    def _scalar_features(self) -> list[float]:
        """Compute 7 base scalar features for the agent.

        Features include danger status, bomb capacity, enemy proximity,
        enemy count, normalized position, and minimum bomb timer.

        Returns:
            List of normalized float features.
        """
        game = self.game
        agent = game.agent
        danger_cells = game.predict_danger_cells()
        blast_cells = game.blast_cells()

        on_danger = 1.0 if agent.pos in danger_cells else 0.0
        can_bomb = 1.0 if agent.can_place_bomb() else 0.0

        # Directional safety: a neighbor is safe if walkable and not threatened.
        safe_flags = []
        for row_delta, column_delta in _CARDINAL_DELTAS:
            neighbor = (agent.pos[0] + row_delta, agent.pos[1] + column_delta)
            safe = (
                game.is_walkable(neighbor, mover_pid=agent.pid)
                and neighbor not in danger_cells
                and neighbor not in blast_cells
            )
            safe_flags.append(1.0 if safe else 0.0)
        is_trapped = 0.0 if any(safe_flags) else 1.0

        alive_enemies = [enemy for enemy in game.enemies if enemy.alive]
        if alive_enemies:
            nearest_enemy_distance = min(
                abs(enemy.pos[0] - agent.pos[0]) + abs(enemy.pos[1] - agent.pos[1])
                for enemy in alive_enemies
            )
            nearest_enemy_distance_norm = nearest_enemy_distance / (2 * self.max_dim)
        else:
            nearest_enemy_distance_norm = 1.0
        enemy_alive_fraction = len(alive_enemies) / max(1, game.config.n_enemies)

        agent_row_norm = agent.pos[0] / game.config.height
        agent_column_norm = agent.pos[1] / game.config.width

        min_threatening_bomb_timer_norm = 1.0
        for bomb in game.bombs:
            if agent.pos in danger_cells:
                min_threatening_bomb_timer_norm = min(
                    min_threatening_bomb_timer_norm,
                    bomb.timer / max(1, game.config.bomb_fuse),
                )

        features = [
            on_danger,
            can_bomb,
            nearest_enemy_distance_norm,
            enemy_alive_fraction,
            agent_row_norm,
            agent_column_norm,
            min_threatening_bomb_timer_norm,
            *safe_flags,
            is_trapped,
        ]
        for ray in self.ray_casts():
            features.append(float(ray["distance_norm"]))
            features.append(float(ray["hit_type"]))
        return features

    def raw_state(self) -> dict:
        """Return ground-truth game state for debugging and custom representations.

        Contains:
        - grid: Full board state as numpy array
        - agent_pos: Current agent coordinates
        - agent_alive: Whether agent is alive
        - enemy_positions: List of (position, alive) tuples for all enemies
        - bombs: List of (position, timer, owner_id, range) tuples
        - blasts: List of (position, timer) tuples
        - danger_cells: Sorted list of positions in blast range
        - step: Current game tick count
        """
        game = self.game
        return {
            "grid": game.grid.copy(),
            "agent_pos": game.agent.pos,
            "agent_alive": game.agent.alive,
            "enemy_positions": [(enemy.pos, enemy.alive) for enemy in game.enemies],
            "bombs": [
                (bomb.pos, bomb.timer, bomb.owner_pid, bomb.bomb_range)
                for bomb in game.bombs
            ],
            "blasts": [(blast.pos, blast.timer) for blast in game.blasts],
            "danger_cells": sorted(game.predict_danger_cells()),
            "step": game.step_count,
        }
