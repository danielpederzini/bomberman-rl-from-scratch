"""Scripted enemy AI.

Enemies follow a simple but escape-aware policy:
    1. If the current tile is dangerous (about to be hit by a blast), path to
       the nearest safe tile (BFS) instead of moving randomly.
    2. If bombing is enabled (``GameConfig.enemies_drop_bombs``) and dropping a
       bomb now would hit a crate or the agent *and* a tile outside the blast is
       reachable within the fuse window, drop a bomb (with probability
       ``GameConfig.enemy_bomb_prob``).
    3. Otherwise wander to a random walkable neighbour (or wait).

Bombing is gated behind a feature flag so you can train without it first and
switch it on later. The escape checks keep enemies from blowing themselves up
most of the time, while still being beatable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from bomberman.actions import Action
from bomberman.entities import Player, Pos, Tile
from bomberman.game import Game


# Maps an action to the delta it applies, restricted to moves.
_MOVE_BY_DELTA = {
    (-1, 0): Action.UP,
    (1, 0): Action.DOWN,
    (0, -1): Action.LEFT,
    (0, 1): Action.RIGHT,
}

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class ScriptedEnemy:
    """Decides an action for a single enemy player each tick."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, game: Game, enemy: Player) -> Action:
        if not enemy.alive:
            return Action.WAIT

        danger_cells = game.predict_danger_cells()

        # 1) Flee danger by pathing to the nearest safe tile.
        if enemy.pos in danger_cells:
            return self._flee(game, enemy, danger_cells)

        # 2) Drop a bomb when it's useful and survivable (feature-flagged).
        if game.config.enemies_drop_bombs and enemy.can_place_bomb():
            if self._should_bomb(game, enemy) and self.rng.random() < game.config.enemy_bomb_prob:
                return Action.BOMB

        # 3) Wander: prefer safe neighbours, sometimes wait.
        neighbours = self._walkable_neighbours(game, enemy)
        safe_neighbours = [neighbour for neighbour in neighbours if neighbour not in danger_cells]
        if safe_neighbours and self.rng.random() < 0.8:
            return self._move_towards(enemy.pos, self._choice(safe_neighbours))
        return Action.WAIT

    # ------------------------------------------------------------------ #
    # Bombing decision
    # ------------------------------------------------------------------ #
    def _should_bomb(self, game: Game, enemy: Player) -> bool:
        """Bomb only if it hits a crate/agent AND the enemy can escape its blast."""
        hypothetical_blast_cells = self._hypothetical_blast(game, enemy.pos, enemy.bomb_range)

        hits_crate = any(
            game.tile_at(blast_position) == Tile.CRATE
            for blast_position in hypothetical_blast_cells
        )
        agent = game.agent
        hits_agent = agent.alive and agent.pos in hypothetical_blast_cells
        if not (hits_crate or hits_agent):
            return False

        return self._can_escape(game, enemy, hypothetical_blast_cells)

    def _can_escape(self, game: Game, enemy: Player, blast_cells: set[Pos]) -> bool:
        """Is a tile outside `blast_cells` reachable within the bomb's fuse window?

        The enemy gets ``bomb_fuse - 1`` moves before the bomb detonates, so we
        BFS up to that depth over walkable tiles (which we may pass through even
        if they are inside the blast, since the bomb has not gone off yet).
        """
        max_depth = max(0, game.config.bomb_fuse - 1)
        start = enemy.pos
        queue: deque[tuple[Pos, int]] = deque([(start, 0)])
        seen_positions = {start}
        while queue:
            current_position, depth = queue.popleft()
            if current_position not in blast_cells:
                return True
            if depth >= max_depth:
                continue
            for row_delta, col_delta in _DIRS:
                next_position = (
                    current_position[0] + row_delta,
                    current_position[1] + col_delta,
                )
                if next_position in seen_positions:
                    continue
                if not game.is_walkable(next_position, mover_pid=enemy.pid):
                    continue
                seen_positions.add(next_position)
                queue.append((next_position, depth + 1))
        return False

    def _hypothetical_blast(self, game: Game, bomb_position: Pos, bomb_range: int) -> set[Pos]:
        """Cells a bomb placed at `bomb_position` would set on fire (without mutating)."""
        blast_cells: set[Pos] = {bomb_position}
        for row_delta, col_delta in _DIRS:
            for distance in range(1, bomb_range + 1):
                blast_position = (
                    bomb_position[0] + row_delta * distance,
                    bomb_position[1] + col_delta * distance,
                )
                if not game.in_bounds(blast_position):
                    break
                tile = game.tile_at(blast_position)
                if tile == Tile.WALL:
                    break
                blast_cells.add(blast_position)
                if tile == Tile.CRATE:
                    break
        return blast_cells

    # ------------------------------------------------------------------ #
    # Fleeing
    # ------------------------------------------------------------------ #
    def _flee(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action:
        """Return the first step along the shortest path to the nearest safe tile."""
        start = enemy.pos
        queue: deque[Pos] = deque([start])
        previous_position_by_position: dict[Pos, Pos | None] = {start: None}
        target: Pos | None = None

        while queue:
            current_position = queue.popleft()
            if current_position != start and current_position not in danger_cells:
                target = current_position
                break
            for row_delta, col_delta in _DIRS:
                next_position = (
                    current_position[0] + row_delta,
                    current_position[1] + col_delta,
                )
                if next_position in previous_position_by_position:
                    continue
                if not game.is_walkable(next_position, mover_pid=enemy.pid):
                    continue
                previous_position_by_position[next_position] = current_position
                queue.append(next_position)

        if target is None:
            # No safe tile reachable; move anywhere walkable, else wait.
            neighbours = self._walkable_neighbours(game, enemy)
            if neighbours:
                return self._move_towards(start, self._choice(neighbours))
            return Action.WAIT

        # Backtrack to the first step from the start tile.
        step = target
        while previous_position_by_position[step] != start:
            step = previous_position_by_position[step]  # type: ignore[assignment]
        return self._move_towards(start, step)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _walkable_neighbours(self, game: Game, enemy: Player) -> list[Pos]:
        walkable_neighbours = []
        for row_delta, col_delta in _MOVE_BY_DELTA:
            target_position = (enemy.pos[0] + row_delta, enemy.pos[1] + col_delta)
            if game.is_walkable(target_position, mover_pid=enemy.pid):
                walkable_neighbours.append(target_position)
        return walkable_neighbours

    def _move_towards(self, source_position: Pos, destination_position: Pos) -> Action:
        delta = (
            destination_position[0] - source_position[0],
            destination_position[1] - source_position[1],
        )
        return _MOVE_BY_DELTA.get(delta, Action.WAIT)

    def _choice(self, items: list[Pos]) -> Pos:
        return items[int(self.rng.integers(len(items)))]


class EnemyController:
    """Owns one ScriptedEnemy policy and produces actions for all enemies."""

    def __init__(self, rng: np.random.Generator):
        self.policy = ScriptedEnemy(rng)

    def actions(self, game: Game) -> dict[int, Action]:
        return {enemy.pid: self.policy.act(game, enemy) for enemy in game.enemies}
