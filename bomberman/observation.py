"""Observation extraction for the RL agent.

Two things are exposed:

* :meth:`ObservationBuilder.features` - a flat ``np.float32`` vector built from
  ray-casts in N directions plus a handful of scalar features. This is the
  default observation and is ready for an MLP DQN.
* :meth:`ObservationBuilder.raw_state` - a dict of ground-truth game state
  (grid, positions, bombs, blasts) so you can build your own representation.

Ray-cast semantics: from the agent, walk outward one cell at a time. The ray
records the first crate / enemy / bomb it meets and is blocked by unbreakable
walls and crates (you cannot "see" past them). Distances are normalised to
``[0, 1]`` where ``1.0`` means "nothing of that kind is visible in that
direction".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bomberman.entities import Tile
from bomberman.game import Game

# 4-connected directions (up, down, left, right).
DIRS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
# Diagonals, appended when 8 directions are requested.
DIRS_8 = DIRS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

# Human-readable names for each direction (used by visualisation).
DIR_NAMES = {
    (-1, 0): "UP",
    (1, 0): "DOWN",
    (0, -1): "LEFT",
    (0, 1): "RIGHT",
    (-1, -1): "UL",
    (-1, 1): "UR",
    (1, -1): "DL",
    (1, 1): "DR",
}

# Labels for the scalar features, in order (see _scalar_features).
SCALAR_NAMES = [
    "on_danger",
    "can_bomb",
    "near_enemy",
    "enemies_alive",
    "agent_row",
    "agent_col",
    "min_fuse",
]

# Number of distance features recorded per ray: wall, crate, enemy, bomb.
_FEATURES_PER_RAY = 4
# Number of global scalar features (see _scalar_features).
_SCALAR_FEATURES = len(SCALAR_NAMES)


@dataclass
class RayHit:
    wall: float = 1.0
    crate: float = 1.0
    enemy: float = 1.0
    bomb: float = 1.0

    def as_list(self) -> list[float]:
        return [self.wall, self.crate, self.enemy, self.bomb]


class ObservationBuilder:
    def __init__(self, game: Game, n_ray_dirs: int = 4):
        if n_ray_dirs not in (4, 8):
            raise ValueError("n_ray_dirs must be 4 or 8")
        self.game = game
        self.dirs = DIRS_8 if n_ray_dirs == 8 else DIRS_4
        self.max_dim = max(game.config.width, game.config.height)

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self.dirs) * _FEATURES_PER_RAY + _SCALAR_FEATURES

    def features(self) -> np.ndarray:
        game = self.game
        agent = game.agent
        features: list[float] = []

        # If the agent is dead, return a zero vector (terminal anyway).
        if not agent.alive:
            return np.zeros(self.size, dtype=np.float32)

        enemy_positions = {enemy.pos for enemy in game.enemies if enemy.alive}
        bomb_positions = {bomb.pos for bomb in game.bombs}

        for row_delta, col_delta in self.dirs:
            features.extend(
                self._cast_ray(
                    agent.pos,
                    row_delta,
                    col_delta,
                    enemy_positions,
                    bomb_positions,
                ).as_list()
            )

        features.extend(self._scalar_features())
        return np.asarray(features, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _cast_ray(
        self,
        start_position,
        row_delta,
        col_delta,
        enemy_positions,
        bomb_positions,
    ) -> RayHit:
        game = self.game
        ray_hit = RayHit()
        row, col = start_position
        for distance in range(1, self.max_dim):
            row += row_delta
            col += col_delta
            ray_position = (row, col)
            if not game.in_bounds(ray_position):
                ray_hit.wall = self._norm(distance)
                break
            normalized_distance = self._norm(distance)
            # Record dynamic objects (only the first, closest one).
            if ray_position in bomb_positions and ray_hit.bomb == 1.0:
                ray_hit.bomb = normalized_distance
            if ray_position in enemy_positions and ray_hit.enemy == 1.0:
                ray_hit.enemy = normalized_distance

            tile = game.tile_at(ray_position)
            if tile == Tile.WALL:
                ray_hit.wall = normalized_distance
                break
            if tile == Tile.CRATE:
                ray_hit.crate = normalized_distance
                break  # crate blocks line of sight
        return ray_hit

    def _scalar_features(self) -> list[float]:
        game = self.game
        agent = game.agent
        danger_cells = game.predict_danger_cells()

        on_danger = 1.0 if agent.pos in danger_cells else 0.0
        can_bomb = 1.0 if agent.can_place_bomb() else 0.0

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
        agent_col_norm = agent.pos[1] / game.config.width

        # Smallest fuse among bombs whose blast reaches the agent (1.0 if none).
        min_threatening_bomb_timer_norm = 1.0
        for bomb in game.bombs:
            if agent.pos in danger_cells:
                min_threatening_bomb_timer_norm = min(
                    min_threatening_bomb_timer_norm,
                    bomb.timer / max(1, game.config.bomb_fuse),
                )

        return [
            on_danger,
            can_bomb,
            nearest_enemy_distance_norm,
            enemy_alive_fraction,
            agent_row_norm,
            agent_col_norm,
            min_threatening_bomb_timer_norm,
        ]

    def _norm(self, distance: int) -> float:
        return min(1.0, distance / self.max_dim)

    # ------------------------------------------------------------------ #
    def ray_visualization(self) -> list[dict]:
        """Structured ray data for rendering overlays.

        Returns one dict per direction with:
            ``dir``   - the (drow, dcol) step,
            ``name``  - human-readable direction name,
            ``cells`` - list of grid cells the ray passes through (in order),
            ``hits``  - mapping of {"wall"/"crate"/"enemy"/"bomb": cell} for the
                        first object of each type the ray meets.
        """
        game = self.game
        agent = game.agent
        if not agent.alive:
            return []

        enemy_positions = {enemy.pos for enemy in game.enemies if enemy.alive}
        bomb_positions = {bomb.pos for bomb in game.bombs}

        rays: list[dict] = []
        for row_delta, col_delta in self.dirs:
            ray_cells: list = []
            hits: dict = {}
            row, col = agent.pos
            for _ in range(1, self.max_dim):
                row += row_delta
                col += col_delta
                ray_position = (row, col)
                if not game.in_bounds(ray_position):
                    break
                ray_cells.append(ray_position)
                if "bomb" not in hits and ray_position in bomb_positions:
                    hits["bomb"] = ray_position
                if "enemy" not in hits and ray_position in enemy_positions:
                    hits["enemy"] = ray_position
                tile = game.tile_at(ray_position)
                if tile == Tile.WALL:
                    hits["wall"] = ray_position
                    break
                if tile == Tile.CRATE:
                    hits["crate"] = ray_position
                    break
            rays.append(
                {
                    "dir": (row_delta, col_delta),
                    "name": DIR_NAMES[(row_delta, col_delta)],
                    "cells": ray_cells,
                    "hits": hits,
                }
            )
        return rays

    # ------------------------------------------------------------------ #
    def raw_state(self) -> dict:
        """Ground-truth state for building custom representations / debugging."""
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
