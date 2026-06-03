"""Discrete action space for the Bomberman environment."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """The six discrete actions an agent can take on a single tick."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    BOMB = 4
    WAIT = 5

    @property
    def is_move(self) -> bool:
        return self in _MOVE_ACTIONS

    @property
    def delta(self) -> tuple[int, int]:
        """Return the (drow, dcol) grid delta for this action.

        Non-move actions (BOMB, WAIT) return (0, 0).
        """
        return _DELTAS[self]


# (drow, dcol) deltas. Rows increase downward, cols increase rightward.
_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.BOMB: (0, 0),
    Action.WAIT: (0, 0),
}

_MOVE_ACTIONS = frozenset({Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT})

NUM_ACTIONS = len(Action)
