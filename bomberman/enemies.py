"""Scripted enemy AI for Bomberman.

This module provides rule-based enemy behavior including:
- Fleeing from bomb explosions
- Chasing the agent
- Destroying crates to reach the agent
- Avoiding friendly fire
"""

from __future__ import annotations

from collections import deque
from typing import TypeAlias

import numpy as np

from bomberman.actions import Action
from bomberman.entities import Player, Pos, Tile
from bomberman.game import Game


DELTA_TO_ACTION: dict[tuple[int, int], Action] = {
    (-1, 0): Action.UP,
    (1, 0): Action.DOWN,
    (0, -1): Action.LEFT,
    (0, 1): Action.RIGHT,
}

DIRECTION_OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
Path: TypeAlias = list[Pos]
ENEMY_DISENGAGE_DURATION: int = 8
ENEMY_STUCK_THRESHOLD: int = 6


class ScriptedEnemy:
    """Rule-based AI that decides actions for a single enemy each game tick.

    The enemy follows a priority-based behavior system:
    1. Flee from danger (explosions)
    2. Drop bombs when advantageous (hits target, can escape)
    3. Chase the agent when visible
    4. Destroy crates blocking the path to the agent
    5. Patrol when no immediate objective
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._disengage_ticks: dict[int, int] = {}
        self._stuck_counter: dict[int, int] = {}
        self._last_position: dict[int, Pos] = {}
        self._position_history: dict[int, list[Pos]] = {}

    def act(self, game: Game, enemy: Player) -> Action:
        """Select the next action for this enemy based on game state.

        Priority order:
        1. Survival (flee from explosions)
        2. Combat (drop bomb if safe and beneficial)
        3. Movement (chase, seek crates, patrol)
        4. Fallback (wait)
        """
        if not enemy.alive:
            self._clear_enemy_state(enemy.pid)
            return Action.WAIT

        danger_cells = game.predict_danger_cells()
        self._update_stuck_state(enemy)

        if self._try_break_stalemate_with_bomb(game, enemy, danger_cells):
            return Action.BOMB

        if enemy.pos in danger_cells:
            return self._flee(game, enemy, danger_cells)

        if self._should_drop_bomb(game, enemy):
            return Action.BOMB

        if self._disengage_ticks.get(enemy.pid, 0) > 0:
            self._disengage_ticks[enemy.pid] -= 1
            return self._step_away(game, enemy, game.agent.pos, danger_cells) or Action.WAIT

        if danger_cells:
            return self._wander(game, enemy, danger_cells) or Action.WAIT

        return self._decide_combat_or_exploration_action(game, enemy, danger_cells)

    def _clear_enemy_state(self, enemy_pid: int) -> None:
        """Remove all tracked state for a dead enemy."""
        self._disengage_ticks.pop(enemy_pid, None)
        self._stuck_counter.pop(enemy_pid, None)
        self._last_position.pop(enemy_pid, None)
        self._position_history.pop(enemy_pid, None)

    def _update_stuck_state(self, enemy: Player) -> None:
        """Track position history and detect if enemy is stuck or oscillating."""
        position_history = self._position_history.get(enemy.pid, [])
        position_history.append(enemy.pos)
        if len(position_history) > 4:
            position_history = position_history[-4:]
        self._position_history[enemy.pid] = position_history

        last_position = self._last_position.get(enemy.pid)
        is_moving_back_and_forth = (
            len(position_history) >= 4 and
            position_history[0] == position_history[2] and
            position_history[1] == position_history[3]
        )

        if last_position == enemy.pos or is_moving_back_and_forth:
            self._stuck_counter[enemy.pid] = self._stuck_counter.get(enemy.pid, 0) + 1
        else:
            self._stuck_counter[enemy.pid] = 0
        self._last_position[enemy.pid] = enemy.pos

    def _try_break_stalemate_with_bomb(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> bool:
        """Attempt to drop a bomb to break out of being stuck.

        Returns True if bomb was dropped successfully.
        """
        stuck_count = self._stuck_counter.get(enemy.pid, 0)
        if stuck_count < ENEMY_STUCK_THRESHOLD:
            return False
        if not game.config.enemies_drop_bombs:
            return False
        if not enemy.can_place_bomb():
            return False
        if danger_cells:
            return False
        if not self._can_escape(game, enemy, {enemy.pos}):
            return False

        blast_area = self._hypothetical_blast(game, enemy.pos, enemy.bomb_range)
        blast_hits_crate = any(game.tile_at(position) == Tile.CRATE for position in blast_area)
        blast_reaches_agent = game.agent.alive and game.agent.pos in blast_area

        if blast_hits_crate or blast_reaches_agent:
            self._stuck_counter[enemy.pid] = 0
            return True
        return False

    def _decide_combat_or_exploration_action(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action:
        """Choose between chasing, seeking crates, or patrolling."""
        if self.rng.random() < game.config.enemy_chase_prob:
            chase_action = self._chase(game, enemy, danger_cells)
            if chase_action is not None:
                return chase_action
            if self._should_drop_bomb(game, enemy):
                return Action.BOMB

        if self.rng.random() < 0.70:
            crate_action = self._seek_crate(game, enemy, danger_cells)
            if crate_action is not None:
                return crate_action

        patrol_action = self._patrol(game, enemy, danger_cells)
        if patrol_action is not None:
            return patrol_action

        if self._try_bomb_blocked_path(game, enemy, danger_cells):
            return Action.BOMB

        if self._try_bomb_to_escape(game, enemy, danger_cells):
            return Action.BOMB

        return Action.WAIT

    def _try_bomb_blocked_path(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> bool:
        """Try bombing when very close to agent but path is blocked."""
        if not game.config.enemies_drop_bombs:
            return False
        if not enemy.can_place_bomb():
            return False
        if danger_cells:
            return False
        if not game.agent.alive:
            return False

        distance_to_agent = abs(enemy.pos[0] - game.agent.pos[0]) + abs(enemy.pos[1] - game.agent.pos[1])
        if distance_to_agent > 2:
            return False

        blast_area = self._hypothetical_blast(game, enemy.pos, enemy.bomb_range)
        if game.agent.pos not in blast_area:
            return False
        if not self._can_escape(game, enemy, blast_area):
            return False

        return True

    def _try_bomb_to_escape(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> bool:
        """Try bombing when completely stuck, if it would hit something useful."""
        if not game.config.enemies_drop_bombs:
            return False
        if not enemy.can_place_bomb():
            return False
        if danger_cells:
            return False
        if not self._can_escape(game, enemy, {enemy.pos}):
            return False
        if self.rng.random() >= game.config.enemy_bomb_prob:
            return False

        blast_area = self._hypothetical_blast(game, enemy.pos, enemy.bomb_range)
        hits_something = any(
            game.tile_at(position) == Tile.CRATE or
            (game.agent.alive and game.agent.pos == position)
            for position in blast_area
        )
        return hits_something

    def _should_drop_bomb(self, game: Game, enemy: Player) -> bool:
        if not game.config.enemies_drop_bombs or not enemy.can_place_bomb():
            return False
        if not self._bomb_hits_target(game, enemy):
            return False
        if self.rng.random() >= game.config.enemy_bomb_prob:
            return False
        return True

    def _bomb_hits_target(self, game: Game, enemy: Player) -> bool:
        """Check if placing a bomb would hit a target and be safe for allies.

        Validates:
        - Enemy can escape the blast area
        - Blast hits a crate or the agent
        - All allies in blast area can also escape
        """
        blast_area = self._hypothetical_blast(game, enemy.pos, enemy.bomb_range)
        if not self._can_escape(game, enemy, blast_area):
            return False

        blast_hits_crate = any(game.tile_at(position) == Tile.CRATE for position in blast_area)
        agent = game.agent
        blast_reaches_agent = agent.alive and agent.pos in blast_area
        if not (blast_hits_crate or blast_reaches_agent):
            return False

        for ally in game.enemies:
            if ally.alive and ally.pid != enemy.pid and ally.pos in blast_area:
                if not self._can_escape(game, ally, blast_area):
                    return False
        return True

    def _can_escape(self, game: Game, enemy: Player, blast_cells: set[Pos]) -> bool:
        """Check whether an enemy can reach a safe cell before the bomb explodes."""
        max_depth = max(0, game.config.bomb_fuse - 2)
        start = enemy.pos
        queue: deque[tuple[Pos, int]] = deque([(start, 0)])
        visited_positions = {start}

        while queue:
            current_position, depth = queue.popleft()
            if current_position not in blast_cells:
                return True
            if depth >= max_depth:
                continue
            for neighbor in self._neighboring_positions(current_position):
                if neighbor in visited_positions:
                    continue
                if neighbor != start and not game.is_walkable(neighbor, mover_pid=enemy.pid):
                    continue
                visited_positions.add(neighbor)
                queue.append((neighbor, depth + 1))
        return False

    def _hypothetical_blast(self, game: Game, bomb_position: Pos, bomb_range: int) -> set[Pos]:
        """Return all cells that a bomb at the given position would hit on explosion."""
        blast_area: set[Pos] = {bomb_position}
        for row_offset, column_offset in DIRECTION_OFFSETS:
            for distance in range(1, bomb_range + 1):
                blast_cell = (
                    bomb_position[0] + row_offset * distance,
                    bomb_position[1] + column_offset * distance,
                )
                if not game.in_bounds(blast_cell):
                    break
                tile = game.tile_at(blast_cell)
                if tile == Tile.WALL:
                    break
                blast_area.add(blast_cell)
                if tile == Tile.CRATE:
                    break
        return blast_area

    def _flee(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action:
        """Find and take the first step toward the nearest safe cell."""
        safe_cell = self._find_safe_cell_bfs(game, enemy, danger_cells)

        if safe_cell is None:
            walkable_neighbors = self._walkable_neighbors(game, enemy)
            if walkable_neighbors:
                return self._move_towards(enemy.pos, self._random_choice(walkable_neighbors))
            return Action.WAIT

        first_step = self._get_first_step(safe_cell, enemy.pos, self._came_from_flee)
        if first_step:
            return self._move_towards(enemy.pos, first_step)
        return Action.WAIT

    def _find_safe_cell_bfs(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Pos | None:
        """BFS to find the nearest cell not in danger, avoiding ally positions."""
        start = enemy.pos
        ally_positions = {other.pos for other in game.enemies if other.alive and other.pid != enemy.pid}
        queue: deque[Pos] = deque([start])
        self._came_from_flee: dict[Pos, Pos | None] = {start: None}

        while queue:
            current = queue.popleft()
            if current != start and current not in danger_cells:
                return current
            for neighbor in self._neighboring_positions(current):
                if neighbor in self._came_from_flee:
                    continue
                if neighbor not in ally_positions and not game.is_walkable(neighbor, mover_pid=enemy.pid):
                    continue
                self._came_from_flee[neighbor] = current
                queue.append(neighbor)
        return None

    def _chase(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action | None:
        """Take the first step toward the agent via the shortest safe path."""
        agent = game.agent
        if not agent.alive:
            return self._wander(game, enemy, danger_cells)

        target = agent.pos
        first_step = self._find_path_to_target(game, enemy, target, danger_cells)

        if first_step:
            return self._move_towards(enemy.pos, first_step)
        return self._wander(game, enemy, danger_cells)

    def _find_path_to_target(self, game: Game, enemy: Player, target: Pos, danger_cells: set[Pos]) -> Pos | None:
        """BFS to find first step toward target, optionally avoiding danger.

        Tries safe path first, then allows dangerous path if necessary.
        """
        start = enemy.pos

        for avoid_danger in (True, False):
            queue: deque[Pos] = deque([start])
            came_from: dict[Pos, Pos | None] = {start: None}
            found = False

            while queue:
                current = queue.popleft()
                if current == target:
                    found = True
                    break
                for neighbor in self._neighboring_positions(current):
                    if neighbor in came_from:
                        continue
                    if not game.is_walkable(neighbor, mover_pid=enemy.pid) and neighbor != target:
                        continue
                    if avoid_danger and neighbor in danger_cells:
                        continue
                    came_from[neighbor] = current
                    queue.append(neighbor)

            if found:
                first_step = self._get_first_step(target, start, came_from)
                if first_step and game.is_walkable(first_step, mover_pid=enemy.pid):
                    return first_step
        return None

    def _get_first_step(self, target: Pos, start: Pos, came_from: dict[Pos, Pos | None]) -> Pos | None:
        """Backtrack from target to find the first step from start."""
        step = target
        while came_from.get(step) is not None and came_from[step] != start:
            step = came_from[step]  # type: ignore
        return step if step != start else None

    def _step_away(
        self,
        game: Game,
        enemy: Player,
        reference_position: Pos,
        danger_cells: set[Pos] | None = None,
    ) -> Action | None:
        """Move to the walkable neighbor farthest from a reference position."""
        valid_neighbors = self._get_valid_neighbors(game, enemy, danger_cells)
        if not valid_neighbors:
            return None

        best_neighbor = max(
            valid_neighbors,
            key=lambda position: abs(position[0] - reference_position[0]) + abs(position[1] - reference_position[1]),
        )
        return self._move_towards(enemy.pos, best_neighbor)

    def _get_valid_neighbors(self, game: Game, enemy: Player, danger_cells: set[Pos] | None) -> list[Pos]:
        """Get all walkable neighbors, avoiding allies and optionally danger."""
        ally_positions = {other.pos for other in game.enemies if other.alive and other.pid != enemy.pid}
        valid_neighbors = []

        for row_offset, column_offset in DIRECTION_OFFSETS:
            neighbor_position = (enemy.pos[0] + row_offset, enemy.pos[1] + column_offset)
            if neighbor_position in ally_positions or game.is_walkable(neighbor_position, mover_pid=enemy.pid):
                valid_neighbors.append(neighbor_position)

        if danger_cells:
            safe_neighbors = [n for n in valid_neighbors if n not in danger_cells]
            if safe_neighbors:
                return safe_neighbors
        return valid_neighbors

    def _wander(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action | None:
        """Move to a random safe neighbor, or wait if none is available."""
        neighbors = self._walkable_neighbors(game, enemy)
        safe_neighbors = [n for n in neighbors if n not in danger_cells]
        if safe_neighbors:
            return self._move_towards(enemy.pos, self._random_choice(safe_neighbors))
        return None

    def _walkable_neighbors(self, game: Game, enemy: Player) -> list[Pos]:
        """Return list of positions the enemy can move to."""
        walkable = []
        for row_offset, column_offset in DIRECTION_OFFSETS:
            neighbor = (enemy.pos[0] + row_offset, enemy.pos[1] + column_offset)
            if game.is_walkable(neighbor, mover_pid=enemy.pid):
                walkable.append(neighbor)
        return walkable

    def _neighboring_positions(self, position: Pos) -> list[Pos]:
        """Return the four adjacent positions (up, down, left, right)."""
        return [
            (position[0] + row_offset, position[1] + column_offset)
            for row_offset, column_offset in DIRECTION_OFFSETS
        ]

    def _move_towards(self, source: Pos, destination: Pos) -> Action:
        """Convert a position delta to the corresponding movement action."""
        delta = (destination[0] - source[0], destination[1] - source[1])
        return DELTA_TO_ACTION.get(delta, Action.WAIT)

    def _random_choice(self, items: list[Pos]) -> Pos:
        """Select a random item from the list."""
        return items[int(self.rng.integers(len(items)))]

    def _seek_crate(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action | None:
        """Move toward a crate, preferring those closer to the agent."""
        start = enemy.pos
        agent = game.agent

        crate_positions = []
        for row in range(game.config.height):
            for col in range(game.config.width):
                position = (row, col)
                if game.tile_at(position) == Tile.CRATE:
                    crate_positions.append(position)

        if not crate_positions:
            return None

        if agent.alive:
            def crate_score(crate_position):
                distance_to_enemy = abs(crate_position[0] - start[0]) + abs(crate_position[1] - start[1])
                distance_to_agent = abs(crate_position[0] - agent.pos[0]) + abs(crate_position[1] - agent.pos[1])
                return distance_to_enemy + distance_to_agent * 0.5
            nearest_crate = min(crate_positions, key=crate_score)
        else:
            nearest_crate = min(crate_positions, key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1]))

        for avoid_danger in (True, False):
            queue: deque[Pos] = deque([start])
            came_from: dict[Pos, Pos | None] = {start: None}
            found = False

            while queue:
                current = queue.popleft()
                if current == nearest_crate:
                    found = True
                    break
                for neighbor in self._neighboring_positions(current):
                    if neighbor in came_from:
                        continue
                    if not game.is_walkable(neighbor, mover_pid=enemy.pid) and neighbor != nearest_crate:
                        continue
                    if avoid_danger and neighbor in danger_cells:
                        continue
                    came_from[neighbor] = current
                    queue.append(neighbor)

            if found:
                first_step = self._get_first_step(nearest_crate, start, came_from)
                if first_step and game.is_walkable(first_step, mover_pid=enemy.pid):
                    return self._move_towards(start, first_step)

        return None

    def _patrol(self, game: Game, enemy: Player, danger_cells: set[Pos]) -> Action | None:
        """Move in a safe direction, preferring movement toward the agent."""
        agent = game.agent
        neighbors = self._walkable_neighbors(game, enemy)
        safe_neighbors = [n for n in neighbors if n not in danger_cells]

        if not safe_neighbors:
            return None

        if agent.alive:
            current_distance = abs(enemy.pos[0] - agent.pos[0]) + abs(enemy.pos[1] - agent.pos[1])
            better_neighbors = [
                n for n in safe_neighbors
                if abs(n[0] - agent.pos[0]) + abs(n[1] - agent.pos[1]) < current_distance
            ]
            if better_neighbors:
                return self._move_towards(enemy.pos, self._random_choice(better_neighbors))

        return self._move_towards(enemy.pos, self._random_choice(safe_neighbors))


class EnemyController:
    """Coordinates AI actions for all enemies in the game.

    Maintains a single ScriptedEnemy policy instance and generates
    actions for each enemy at every game tick.
    """

    def __init__(self, rng: np.random.Generator):
        self.policy = ScriptedEnemy(rng)

    def actions(self, game: Game) -> dict[int, Action]:
        """Generate actions for all alive enemies.

        Returns a mapping from enemy player ID to their chosen action.
        """
        return {enemy.pid: self.policy.act(game, enemy) for enemy in game.enemies}
