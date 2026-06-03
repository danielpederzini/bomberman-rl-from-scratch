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

    crates_destroyed: dict[int, int] = field(default_factory=dict)  # pid -> count
    kills: dict[int, list[int]] = field(default_factory=dict)        # killer_pid -> [victim_pid]
    deaths: list[int] = field(default_factory=list)                  # pids that died this tick


class Game:
    """Stateful Bomberman game.

    Coordinates are ``(row, col)`` with the origin at the top-left.
    Player id 0 is the RL agent; ids >= 1 are scripted enemies.
    """

    def __init__(self, config: GameConfig | None = None, rng: np.random.Generator | None = None):
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

        self.reset()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Generate a fresh board and place players at the corners."""
        config = self.config
        self.grid = np.full((config.height, config.width), Tile.EMPTY, dtype=np.int8)
        self.bombs = []
        self.blasts = []
        self.step_count = 0
        self.done = False

        # Border walls.
        self.grid[0, :] = Tile.WALL
        self.grid[-1, :] = Tile.WALL
        self.grid[:, 0] = Tile.WALL
        self.grid[:, -1] = Tile.WALL

        # Interior pillars on even/even coordinates.
        for row in range(2, config.height - 1, 2):
            for col in range(2, config.width - 1, 2):
                self.grid[row, col] = Tile.WALL

        # Spawn corners (interior).
        corners: list[Pos] = [
            (1, 1),
            (config.height - 2, config.width - 2),
            (1, config.width - 2),
            (config.height - 2, 1),
        ]

        # Keep a small safe zone around each spawn free of crates.
        safe_spawn_cells: set[Pos] = set()
        for spawn_row, spawn_col in corners:
            safe_spawn_cells.add((spawn_row, spawn_col))
            for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                safe_spawn_cells.add((spawn_row + row_delta, spawn_col + col_delta))

        # Seed breakable crates on empty, non-safe tiles.
        for row in range(1, config.height - 1):
            for col in range(1, config.width - 1):
                if self.grid[row, col] != Tile.EMPTY or (row, col) in safe_spawn_cells:
                    continue
                if self.rng.random() < config.crate_density:
                    self.grid[row, col] = Tile.CRATE

        # Create players: agent first, then enemies at the remaining corners.
        self.players = []
        player_count = 1 + config.n_enemies
        for player_id in range(player_count):
            spawn_position = corners[player_id % len(corners)]
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

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def agent(self) -> Player:
        return self.players[0]

    @property
    def enemies(self) -> list[Player]:
        return self.players[1:]

    def in_bounds(self, position: Pos) -> bool:
        row, col = position
        return 0 <= row < self.config.height and 0 <= col < self.config.width

    def tile_at(self, position: Pos) -> Tile:
        return Tile(int(self.grid[position[0], position[1]]))

    def bomb_at(self, position: Pos) -> Bomb | None:
        for bomb in self.bombs:
            if bomb.pos == position:
                return bomb
        return None

    def alive_player_at(self, position: Pos, exclude_pid: int | None = None) -> Player | None:
        for player in self.players:
            if player.alive and player.pos == position and player.pid != exclude_pid:
                return player
        return None

    def is_walkable(self, position: Pos, mover_pid: int | None = None) -> bool:
        """Whether `position` can be moved into: empty tile, no bomb, no other player."""
        if not self.in_bounds(position):
            return False
        if self.tile_at(position) != Tile.EMPTY:
            return False
        if self.bomb_at(position) is not None:
            return False
        if self.alive_player_at(position, exclude_pid=mover_pid) is not None:
            return False
        return True

    def blast_cells(self) -> set[Pos]:
        return {blast.pos for blast in self.blasts}

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def tick(self, actions: dict[int, Action]) -> TickResult:
        """Advance the world by one tick.

        `actions` maps player id -> Action. Missing/ dead players default to WAIT.
        """
        result = TickResult()
        if self.done:
            return result

        # 1) Age out blasts created on previous ticks. Doing this first means a
        #    freshly created blast stays observable for the whole tick it is
        #    created on (and for `blast_duration` ticks total).
        self._age_blasts()

        # 2) Apply player actions (movement + bomb placement).
        #    Process in pid order; movement is blocked by walls/bombs/players.
        for player in self.players:
            if not player.alive:
                continue
            action = actions.get(player.pid, Action.WAIT)
            self._apply_action(player, action)

        # 3) Tick bombs and resolve detonations (with chain reactions).
        self._tick_bombs(result)

        # 4) Kill anyone standing on an active blast cell.
        self._resolve_blast_damage(result)

        # 5) Termination checks.
        self.step_count += 1
        self._check_done()

        return result

    def _apply_action(self, player: Player, action: Action) -> None:
        if action == Action.BOMB:
            self._place_bomb(player)
            return
        if action == Action.WAIT or not action.is_move:
            return
        row_delta, col_delta = action.delta
        target_position = (player.pos[0] + row_delta, player.pos[1] + col_delta)
        if self.is_walkable(target_position, mover_pid=player.pid):
            player.pos = target_position

    def _place_bomb(self, player: Player) -> None:
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
        for bomb in self.bombs:
            bomb.timer -= 1

        # Collect bombs that should detonate now and process chain reactions.
        bombs_to_detonate = [bomb for bomb in self.bombs if bomb.timer <= 0]
        processed_bomb_ids: set[int] = set()

        while bombs_to_detonate:
            bomb = bombs_to_detonate.pop()
            if id(bomb) in processed_bomb_ids:
                continue
            processed_bomb_ids.add(id(bomb))

            # Remove from active bombs & free up owner capacity.
            if bomb in self.bombs:
                self.bombs.remove(bomb)
            owner = self._player_by_pid(bomb.owner_pid)
            if owner is not None:
                owner.active_bombs = max(0, owner.active_bombs - 1)

            chained_bombs = self._explode(bomb, result)
            bombs_to_detonate.extend(chained_bombs)

    def _explode(self, bomb: Bomb, result: TickResult) -> list[Bomb]:
        """Create blast cells for a bomb; destroy crates; return chained bombs."""
        chained: list[Bomb] = []
        blast_positions: list[Pos] = [bomb.pos]

        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for row_delta, col_delta in directions:
            for distance in range(1, bomb.bomb_range + 1):
                blast_position = (
                    bomb.pos[0] + row_delta * distance,
                    bomb.pos[1] + col_delta * distance,
                )
                if not self.in_bounds(blast_position):
                    break
                tile = self.tile_at(blast_position)
                if tile == Tile.WALL:
                    break  # unbreakable: stops the blast, not included
                blast_positions.append(blast_position)
                if tile == Tile.CRATE:
                    # Destroy the crate and stop propagating further this way.
                    self.grid[blast_position[0], blast_position[1]] = Tile.EMPTY
                    result.crates_destroyed[bomb.owner_pid] = (
                        result.crates_destroyed.get(bomb.owner_pid, 0) + 1
                    )
                    break

        # Register blast cells and trigger any bombs caught in the blast.
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
        lethal_cells = self.blast_cells()
        if not lethal_cells:
            return
        for player in self.players:
            if player.alive and player.pos in lethal_cells:
                player.alive = False
                result.deaths.append(player.pid)
                # Attribute the kill to the owner of a blast on this tile.
                killer = self._blast_owner_at(player.pos)
                if killer is not None and killer != player.pid:
                    result.kills.setdefault(killer, []).append(player.pid)

    def _blast_owner_at(self, position: Pos) -> int | None:
        for blast in self.blasts:
            if blast.pos == position:
                return blast.owner_pid
        return None

    def _age_blasts(self) -> None:
        for blast in self.blasts:
            blast.timer -= 1
        self.blasts = [blast for blast in self.blasts if blast.timer > 0]

    def _player_by_pid(self, player_id: int) -> Player | None:
        for player in self.players:
            if player.pid == player_id:
                return player
        return None

    def _check_done(self) -> None:
        agent_alive = self.agent.alive
        enemies_alive = any(enemy.alive for enemy in self.enemies)
        if not agent_alive:
            self.done = True
        elif self.config.n_enemies > 0 and not enemies_alive:
            self.done = True
        elif self.step_count >= self.config.max_steps:
            self.done = True

    # ------------------------------------------------------------------ #
    # Danger helper (used by enemies and the observation extractor)
    # ------------------------------------------------------------------ #
    def predict_danger_cells(self) -> set[Pos]:
        """Cells that are currently on fire or will be hit by a soon-to-explode bomb.

        This is an approximation that treats every active bomb as if it explodes
        now (ignoring fuse length), which is a conservative, useful danger map.
        """
        danger_cells: set[Pos] = set(self.blast_cells())
        for bomb in self.bombs:
            danger_cells.add(bomb.pos)
            directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
            for row_delta, col_delta in directions:
                for distance in range(1, bomb.bomb_range + 1):
                    danger_position = (
                        bomb.pos[0] + row_delta * distance,
                        bomb.pos[1] + col_delta * distance,
                    )
                    if not self.in_bounds(danger_position):
                        break
                    tile = self.tile_at(danger_position)
                    if tile == Tile.WALL:
                        break
                    danger_cells.add(danger_position)
                    if tile == Tile.CRATE:
                        break
        return danger_cells
