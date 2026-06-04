"""Manually play the Bomberman game to sanity-check the mechanics.

Controls:
    Arrow keys / WASD : move
    Space             : drop bomb
    Period (.)        : wait
    R                 : reset
    Esc / window close: quit

The world only advances one tick per key press (turn-based), so you can inspect
exactly what each action does.

Run:
    python play_human.py                  # passive enemies (default)
    python play_human.py --enemy-bombs     # enemies also drop bombs
    python play_human.py --enemy-bombs --enemy-bomb-prob 0.6 --enemies 3
    python play_human.py --show-state          # overlay rays + feature panel
"""

from __future__ import annotations

import argparse
import sys

from bomberman.actions import Action
from bomberman.entities import GameConfig
from bomberman.env import BombermanEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually play Bomberman.")
    parser.add_argument("--width", type=int, default=11,
                        help="width of the game grid")
    parser.add_argument("--height", type=int, default=11,
                        help="height of the game grid")
    parser.add_argument("--enemy-bombs", action="store_true",
                        help="let scripted enemies drop bombs")
    parser.add_argument("--enemy-bomb-prob", type=float, default=0.4,
                        help="per-tick bomb probability when an enemy decides to bomb")
    parser.add_argument("--enemies", type=int, default=2,
                        help="number of scripted enemies")
    parser.add_argument("--show-state", action="store_true",
                        help="overlay the agent's state (rays, danger, feature panel)")
    return parser.parse_args()


def main() -> int:
    import pygame

    args = parse_args()
    config = GameConfig(
        width=args.width,
        height=args.height,
        n_enemies=args.enemies,
        enemies_drop_bombs=args.enemy_bombs,
        enemy_bomb_prob=args.enemy_bomb_prob,
    )
    print(
        f"Starting Bomberman with config: {config.width}x{config.height} "
        f"enemy_bombs={'ON' if args.enemy_bombs else 'OFF'} "
        f"(prob={args.enemy_bomb_prob}) enemies={args.enemies} "
        f"show_state={'ON' if args.show_state else 'OFF'}"
    )
    env = BombermanEnv(config=config, render_mode="human", show_state=args.show_state)
    observation, info = env.reset()
    env.render()

    key_to_action = {
        pygame.K_UP: Action.UP,
        pygame.K_w: Action.UP,
        pygame.K_DOWN: Action.DOWN,
        pygame.K_s: Action.DOWN,
        pygame.K_LEFT: Action.LEFT,
        pygame.K_a: Action.LEFT,
        pygame.K_RIGHT: Action.RIGHT,
        pygame.K_d: Action.RIGHT,
        pygame.K_SPACE: Action.BOMB,
        pygame.K_PERIOD: Action.WAIT,
    }

    running = True
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    env.render()
                elif event.key in key_to_action:
                    action = key_to_action[event.key]

        if action is not None:
            observation, reward, terminated, truncated, info = env.step(int(action))
            print(
                f"action={Action(int(action)).name:5s} reward={reward:+.2f} "
                f"terminated={terminated} truncated={truncated} "
                f"enemies={info['enemies_alive']}"
            )
            if terminated or truncated:
                outcome = "WIN" if info["agent_alive"] else "LOSE/END"
                print(f"--- episode finished: {outcome} (press R to restart) ---")

        env.render()
        pygame.time.wait(20)

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
