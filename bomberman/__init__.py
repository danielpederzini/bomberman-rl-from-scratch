"""A small, RL-friendly Bomberman game.

Public API:
    from bomberman import BombermanEnv
    from bomberman.actions import Action
"""

from bomberman.actions import Action
from bomberman.env import BombermanEnv

__all__ = ["BombermanEnv", "Action"]
