"""Core Bomberman game logic: board, bombs, explosions and the tick model.

The game is fully turn-based / tick-based (no real-time physics) so it is
deterministic and easy to drive from a reinforcement-learning loop.

One call to :meth:`Game.tick` consumes one action per alive player and advances
the world by a single discrete step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bomberman.actions import Action
from bomberman.entities import Blast, Bomb, GameConfig, Player, Pos, Tile


@dataclass
class TickResult:
    """Per-tick bookkeeping used for reward shaping and diagnostics."""

    crates_destroyed: dict[int, int] = field(default_factory=dict)
    kills: dict[int, list[int]] = field(default_factory=dict)
    deaths: list[int] = field(default_factory=list)


class Game:
    """Stateful Bomberman game.

    Coordinates are ``(row, col)`` with the origin at the top-left.
    Player id 0 is the RL agent; ids >= 1 are scripted enemies.
    """

    def __init__(self, config: GameConfig | None = None, rng: np.random.Generator | None = None):
        """Initialize a new Bomberman game with empty board and no players.

        Args:
            config: Game configuration parameters. Uses defaults if not provided.
            rng: Random number generator for reproducible randomness.
        """
        self.config = config or GameConfig()
        self.rng = rng or np.random.default_rng(self.config.seed)

        self.grid: np.ndarray = np.zeros(
            (self.config.height, self.config.width), dtype=np.int8
        )
        self.players: list[Player] = []
        self.bombs: list[Bomb] = []
        self.blasts: list[Blast] = []
        self.step_count: int = 0
        self.done: bool = False
        self._cached_danger_cells: set[Pos] | None = None
        self._cached_bomb_count: int = 0
        self._cached_blast_count: int = 0

        self.reset()

    def reset(self) -> None:
        """Generate a fresh board and place players at the corners."""
        config = self.config
        self.grid = np.full((config.height, config.width), Tile.EMPTY, dtype=np.int8)
        self.bombs = []
        self.blasts = []
        self.step_count = 0
        self.done = False

        self.grid[0, :] = Tile.WALL
        self.grid[-1, :] = Tile.WALL
        self.grid[:, 0] = Tile.WALL
        self.grid[:, -1] = Tile.WALL

        for row in range(2, config.height - 1, 2):
            for col in range(2, config.width - 1, 2):
                self.grid[row, col] = Tile.WALL

        corners: list[Pos] = [
            (1, 1),
            (config.height - 2, config.width - 2),
            (1, config.width - 2),
            (config.height - 2, 1),
        ]

        safe_spawn_cells: set[Pos] = set()
        for spawn_row, spawn_col in corners:
            safe_spawn_cells.add((spawn_row, spawn_col))
            for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                safe_spawn_cells.add((spawn_row + row_delta, spawn_col + col_delta))

        for row in range(1, config.height - 1):
            for col in range(1, config.width - 1):
                if self.grid[row, col] != Tile.EMPTY or (row, col) in safe_spawn_cells:
                    continue
                if self.rng.random() < config.crate_density:
                    self.grid[row, col] = Tile.CRATE

        self.players = []
        player_count = 1 + config.n_enemies
        spawn_positions = self._choose_spawn_positions(player_count, corners)
        if len(spawn_positions) > player_count:
            spawn_positions = spawn_positions[:player_count]

        for player_id, spawn_position in enumerate(spawn_positions):
            self.players.append(
                Player(
                    pos=spawn_position,
                    alive=True,
                    bomb_range=config.bomb_range,
                    max_bombs=1,
                    active_bombs=0,
                    pid=player_id,
                )
            )

    @property
    def agent(self) -> Player:
        """Return the RL agent (player with ID 0)."""
        return self.players[0]

    @property
    def enemies(self) -> list[Player]:
        """Return list of enemy players (all players except agent)."""
        return self.players[1:]

    def in_bounds(self, position: Pos) -> bool:
        """Check if a position is within the game board boundaries.

        Args:
            position: Grid coordinates as (row, column).

        Returns:
            True if position is inside the board, False otherwise.
        """
        row, col = position
        return 0 <= row < self.config.height and 0 <= col < self.config.width

    def tile_at(self, position: Pos) -> Tile:
        """Get the tile type at a specific board position.

        Args:
            position: Grid coordinates as (row, column).

        Returns:
            The Tile enum value at the given position.
        """
        return Tile(int(self.grid[position[0], position[1]]))

    def bomb_at(self, position: Pos) -> Bomb | None:
        """Find a bomb at a specific board position if one exists.

        Args:
            position: Grid coordinates as (row, column).

        Returns:
            The Bomb at the position, or None if no bomb is present.
        """
        for bomb in self.bombs:
            if bomb.pos == position:
                return bomb
        return None

    def alive_player_at(self, position: Pos, exclude_pid: int | None = None) -> Player | None:
        """Find an alive player at a specific board position.

        Args:
            position: Grid coordinates as (row, column).
            exclude_pid: Optional player ID to exclude from search.

        Returns:
            The alive Player at the position, or None if no player is present.
        """
        for player in self.players:
            if player.alive and player.pos == position and player.pid != exclude_pid:
                return player
        return None

    def is_walkable(self, position: Pos, mover_pid: int | None = None) -> bool:
        """Whether `position` can be moved into: empty tile, no bomb, and no other player."""
        if not self.in_bounds(position):
            return False
        if self.tile_at(position) != Tile.EMPTY:
            return False
        if self.bomb_at(position) is not None:
            return False
        if mover_pid is not None and self.alive_player_at(position, exclude_pid=mover_pid) is not None:
            return False
        if mover_pid is None and self.alive_player_at(position) is not None:
            return False
        return True

    def blast_cells(self) -> set[Pos]:
        """Return the set of all positions currently covered by blast effects.

        Returns:
            Set of (row, column) positions with active explosions.
        """
        return {blast.pos for blast in self.blasts}

    def _choose_spawn_positions(self, player_count: int, corners: list[Pos]) -> list[Pos]:
        """Select spawn positions for all players starting with corners.

        Args:
            player_count: Total number of players (agent + enemies).
            corners: List of corner positions to try first.

        Returns:
            List of spawn positions, one per player.
        """
        positions: list[Pos] = []
        used_positions: set[Pos] = set()

        agent_spawn = tuple(self.rng.choice(corners))
        positions.append(agent_spawn)
        used_positions.add(agent_spawn)

        candidate_positions = [c for c in corners if c != agent_spawn]
        candidate_positions.extend(self._valid_spawn_positions())

        for candidate_position in candidate_positions:
            if candidate_position in used_positions:
                continue
            if not self.is_walkable(candidate_position):
                continue
            positions.append(candidate_position)
            used_positions.add(candidate_position)
            if len(positions) >= player_count:
                break

        return positions

    def _valid_spawn_positions(self) -> list[Pos]:
        """Find all valid non-corner spawn positions on the board.

        Returns:
            List of empty positions excluding the four corners.
        """
        positions: list[Pos] = []
        corner_positions = {
            (1, 1),
            (self.config.height - 2, self.config.width - 2),
            (1, self.config.width - 2),
            (self.config.height - 2, 1),
        }
        for row in range(1, self.config.height - 1):
            for col in range(1, self.config.width - 1):
                if self.grid[row, col] != Tile.EMPTY:
                    continue
                if (row, col) in corner_positions:
                    continue
                positions.append((row, col))
        return positions

    def tick(self, actions: dict[int, Action]) -> TickResult:
        """Advance the world by one tick.

        `actions` maps player id -> Action. Missing/ dead players default to WAIT.
        """
        result = TickResult()
        if self.done:
            return result

        self._age_blasts()

        for player in self.players:
            if not player.alive:
                continue
            action = actions.get(player.pid, Action.WAIT)
            self._apply_action(player, action)

        self._tick_bombs(result)
        self._resolve_blast_damage(result)
        self.step_count += 1
        self._check_done()

        return result

    def _apply_action(self, player: Player, action: Action) -> None:
        """Apply a single player's action (movement or bomb placement).

        Args:
            player: The player performing the action.
            action: The action to apply (BOMB, WAIT, or movement).
        """
        if action == Action.BOMB:
            self._place_bomb(player)
            return
        if action == Action.WAIT or not action.is_move:
            return
        row_delta, column_delta = action.delta
        target_position = (player.pos[0] + row_delta, player.pos[1] + column_delta)
        if self.is_walkable(target_position, mover_pid=player.pid):
            player.pos = target_position

    def _place_bomb(self, player: Player) -> None:
        """Place a bomb at the player's position if possible.

        Args:
            player: The player attempting to place a bomb.
        """
        if not player.can_place_bomb():
            return
        if self.bomb_at(player.pos) is not None:
            return
        self.bombs.append(
            Bomb(
                pos=player.pos,
                timer=self.config.bomb_fuse,
                owner_pid=player.pid,
                bomb_range=player.bomb_range,
            )
        )
        player.active_bombs += 1

    def _tick_bombs(self, result: TickResult) -> None:
        """Process bomb timers and trigger detonations with chain reactions.

        Args:
            result: TickResult to record destroyed crates and kills.
        """
        for bomb in self.bombs:
            bomb.timer -= 1

        bombs_to_detonate = [bomb for bomb in self.bombs if bomb.timer <= 0]
        processed_bomb_ids: set[int] = set()

        while bombs_to_detonate:
            bomb = bombs_to_detonate.pop()
            if id(bomb) in processed_bomb_ids:
                continue
            processed_bomb_ids.add(id(bomb))

            if bomb in self.bombs:
                self.bombs.remove(bomb)
            owner = self._player_by_pid(bomb.owner_pid)
            if owner is not None:
                owner.active_bombs = max(0, owner.active_bombs - 1)

            chained_bombs = self._explode(bomb, result)
            bombs_to_detonate.extend(chained_bombs)

    def _explode(self, bomb: Bomb, result: TickResult) -> list[Bomb]:
        """Create blast cells for a bomb; destroy crates; return chained bombs.

        Args:
            bomb: The bomb that is exploding.
            result: TickResult to record destroyed crates.

        Returns:
            List of bombs caught in the blast (for chain reactions).
        """
        chained: list[Bomb] = []
        blast_positions: list[Pos] = [bomb.pos]

        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for row_delta, column_delta in directions:
            for distance in range(1, bomb.bomb_range + 1):
                blast_position = (
                    bomb.pos[0] + row_delta * distance,
                    bomb.pos[1] + column_delta * distance,
                )
                if not self.in_bounds(blast_position):
                    break
                tile = self.tile_at(blast_position)
                if tile == Tile.WALL:
                    break
                blast_positions.append(blast_position)
                if tile == Tile.CRATE:
                    self.grid[blast_position[0], blast_position[1]] = Tile.EMPTY
                    result.crates_destroyed[bomb.owner_pid] = (
                        result.crates_destroyed.get(bomb.owner_pid, 0) + 1
                    )
                    break

        for blast_position in blast_positions:
            self.blasts.append(
                Blast(
                    pos=blast_position,
                    timer=self.config.blast_duration,
                    owner_pid=bomb.owner_pid,
                )
            )
            chained_bomb = self.bomb_at(blast_position)
            if chained_bomb is not None:
                chained.append(chained_bomb)

        return chained

    def _resolve_blast_damage(self, result: TickResult) -> None:
        """Kill any players standing on active blast cells.

        Args:
            result: TickResult to record deaths and attribute kills.
        """
        lethal_cells = self.blast_cells()
        if not lethal_cells:
            return
        for player in self.players:
            if player.alive and player.pos in lethal_cells:
                player.alive = False
                result.deaths.append(player.pid)
                killer = self._blast_owner_at(player.pos)
                if killer is not None and killer != player.pid:
                    result.kills.setdefault(killer, []).append(player.pid)

    def _blast_owner_at(self, position: Pos) -> int | None:
        """Find which player owns a blast at the given position.

        Args:
            position: Grid coordinates as (row, column).

        Returns:
            Player ID of the blast owner, or None if no blast at position.
        """
        for blast in self.blasts:
            if blast.pos == position:
                return blast.owner_pid
        return None

    def _age_blasts(self) -> None:
        """Remove blast cells that have expired."""
        for blast in self.blasts:
            blast.timer -= 1
        self.blasts = [blast for blast in self.blasts if blast.timer > 0]

    def _player_by_pid(self, player_id: int) -> Player | None:
        """Look up a player by their player ID.

        Args:
            player_id: The player ID to search for.

        Returns:
            The Player with matching ID, or None if not found.
        """
        for player in self.players:
            if player.pid == player_id:
                return player
        return None

    def _check_done(self) -> None:
        """Check if the game should end and set done flag if so.

        Game ends when:
        - Agent dies
        - All enemies are dead (if there were enemies)
        - Step count reaches maximum
        """
        agent_alive = self.agent.alive
        enemies_alive = any(enemy.alive for enemy in self.enemies)
        if not agent_alive:
            self.done = True
        elif self.config.n_enemies > 0 and not enemies_alive:
            self.done = True
        elif self.step_count >= self.config.max_steps:
            self.done = True

    def predict_danger_cells(self) -> set[Pos]:
        """Cells that are currently on fire or will be hit by a soon-to-explode bomb.

        This is an approximation that treats every active bomb as if it explodes
        now (ignoring fuse length), which is a conservative, useful danger map.
        """
        if (self._cached_danger_cells is not None and
            self._cached_bomb_count == len(self.bombs) and
            self._cached_blast_count == len(self.blasts)):
            return self._cached_danger_cells

        danger_cells: set[Pos] = set(self.blast_cells())
        for bomb in self.bombs:
            danger_cells.add(bomb.pos)
            directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
            br, bc = bomb.pos
            for row_delta, col_delta in directions:
                for distance in range(1, bomb.bomb_range + 1):
                    danger_position = (br + row_delta * distance, bc + col_delta * distance)
                    if not self.in_bounds(danger_position):
                        break
                    tile = self.tile_at(danger_position)
                    if tile == Tile.WALL:
                        break
                    danger_cells.add(danger_position)
                    if tile == Tile.CRATE:
                        break

        self._cached_danger_cells = danger_cells
        self._cached_bomb_count = len(self.bombs)
        self._cached_blast_count = len(self.blasts)
        return danger_cells
