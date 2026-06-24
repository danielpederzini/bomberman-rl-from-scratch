"""Tile definitions and game entities (player, bomb, enemy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tile(IntEnum):
    """Static tile types stored in the grid."""
    EMPTY = 0
    WALL = 1
    CRATE = 2

Pos = tuple[int, int]

@dataclass
class Player:
    """An agent or scripted character on the board."""
    pos: Pos
    alive: bool = True
    bomb_range: int = 2
    max_bombs: int = 1
    active_bombs: int = 0
    pid: int = 0

    def can_place_bomb(self) -> bool:
        return self.alive and self.active_bombs < self.max_bombs

@dataclass
class Bomb:
    """A bomb ticking down to detonation."""
    pos: Pos
    timer: int
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
    bomb_fuse: int = 4
    bomb_range: int = 2
    blast_duration: int = 1
    crate_density: float = 0.45
    max_steps: int = 200
    enemies_drop_bombs: bool = False
    enemy_bomb_prob: float = 0.4
    enemy_chase_prob: float = 0.60
    enemy_skill: float = 1.0  # 0=clumsy, 1=optimal scripted AI
    reward_step: float = -0.02
    reward_crate: float = 2.5
    reward_kill_enemy: float = 3.0
    reward_win: float = 10.0
    reward_death: float = -20.0
    reward_bomb_target: float = 0.2
    reward_idle: float = -0.05
    reward_escape_danger: float = 0.1
    reward_enter_danger: float = -0.5
    reward_bomb_spam: float = -0.5
    reward_useless_bomb: float = -0.5
    reward_suicide_bomb: float = -3.0
    bomb_loop_radius: int = 2
    bomb_loop_window: int = 10
    progress_weight: float = 0.0
    shaping_gamma: float = 0.99
    seed: int | None = None
