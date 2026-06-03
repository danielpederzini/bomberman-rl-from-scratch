"""Tile definitions and game entities (player, bomb, enemy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tile(IntEnum):
    """Static tile types stored in the grid."""

    EMPTY = 0
    WALL = 1   # unbreakable
    CRATE = 2  # breakable


# Type alias for a grid coordinate (row, col).
Pos = tuple[int, int]


@dataclass
class Player:
    """An agent or scripted character on the board."""

    pos: Pos
    alive: bool = True
    bomb_range: int = 2
    max_bombs: int = 1
    # Number of bombs this player currently has active on the board.
    active_bombs: int = 0
    # Identifier; 0 is the RL agent, >=1 are enemies.
    pid: int = 0

    def can_place_bomb(self) -> bool:
        return self.alive and self.active_bombs < self.max_bombs


@dataclass
class Bomb:
    """A bomb ticking down to detonation."""

    pos: Pos
    timer: int            # ticks remaining until explosion
    owner_pid: int
    bomb_range: int


@dataclass
class Blast:
    """A blast cell currently on fire. Lives for `timer` ticks."""

    pos: Pos
    timer: int
    owner_pid: int


@dataclass
class GameConfig:
    """Tunable parameters for a game instance."""

    width: int = 11
    height: int = 11
    n_enemies: int = 2
    bomb_fuse: int = 4          # ticks from placement to explosion
    bomb_range: int = 2         # blast reach in each direction
    blast_duration: int = 1     # ticks a blast cell stays lethal/visible
    crate_density: float = 0.45  # fraction of eligible empty tiles seeded as crates
    max_steps: int = 200
    enemies_drop_bombs: bool = False  # master feature flag for enemy bombing
    enemy_bomb_prob: float = 0.4       # chance to bomb on a tick when it's a good idea
    # Reward shaping
    reward_step: float = -0.01
    reward_crate: float = 0.1
    reward_kill_enemy: float = 1.0
    reward_win: float = 5.0
    reward_death: float = -5.0
    seed: int | None = None
