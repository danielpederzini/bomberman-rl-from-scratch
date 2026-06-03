"""Rendering: an ASCII renderer (zero deps) and an optional pygame window.

The pygame renderer can optionally draw a "state visualisation" overlay
(``show_state=True``): danger cells are tinted, the agent's ray-casts are drawn
with markers on what they hit, and a side panel prints the live observation
vector. This is handy for recording explanatory videos.
"""

from __future__ import annotations

from bomberman.entities import Tile
from bomberman.game import Game

# ASCII glyphs.
_GLYPH_EMPTY = "."
_GLYPH_WALL = "#"
_GLYPH_CRATE = "+"
_GLYPH_BOMB = "o"
_GLYPH_BLAST = "*"
_GLYPH_AGENT = "A"
_GLYPH_ENEMY = "E"


def render_ascii(game: Game) -> str:
    """Return a human-readable text view of the board.

    Precedence (highest first): agent/enemy, blast, bomb, tile.
    """
    blast_cells = game.blast_cells()
    bomb_cells = {bomb.pos for bomb in game.bombs}
    agent = game.agent
    enemy_cells = {enemy.pos: enemy for enemy in game.enemies if enemy.alive}

    rows = []
    for row in range(game.config.height):
        line = []
        for col in range(game.config.width):
            position = (row, col)
            if agent.alive and agent.pos == position:
                line.append(_GLYPH_AGENT)
            elif position in enemy_cells:
                line.append(_GLYPH_ENEMY)
            elif position in blast_cells:
                line.append(_GLYPH_BLAST)
            elif position in bomb_cells:
                line.append(_GLYPH_BOMB)
            else:
                tile = Tile(int(game.grid[row, col]))
                if tile == Tile.WALL:
                    line.append(_GLYPH_WALL)
                elif tile == Tile.CRATE:
                    line.append(_GLYPH_CRATE)
                else:
                    line.append(_GLYPH_EMPTY)
        rows.append(" ".join(line))
    status = (
        f"step={game.step_count} agent_alive={game.agent.alive} "
        f"enemies={sum(1 for enemy in game.enemies if enemy.alive)}"
    )
    return "\n".join(rows) + "\n" + status


# Colours (R, G, B).
_COLORS = {
    "bg": (30, 30, 40),
    "empty": (50, 50, 60),
    "wall": (90, 90, 100),
    "crate": (160, 110, 60),
    "bomb": (20, 20, 20),
    "blast": (240, 140, 40),
    "agent": (70, 170, 240),
    "enemy": (230, 70, 90),
    "dead": (110, 110, 110),
    "panel": (18, 18, 26),
    "text": (220, 220, 230),
    "text_dim": (140, 140, 160),
    "ray": (90, 200, 120),
}

# Marker colours for what a ray hits.
_HIT_COLORS = {
    "wall": (200, 200, 210),
    "crate": (210, 150, 80),
    "enemy": (255, 80, 100),
    "bomb": (250, 220, 60),
}

_PANEL_WIDTH = 360


class PygameRenderer:
    """Draws the game in a pygame window. Created lazily on first use."""

    def __init__(self, config, cell_size: int = 40, show_state: bool = False):
        import pygame

        self.pygame = pygame
        self.config = config
        self.cell = cell_size
        self.show_state = show_state
        self._closed = False

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Bomberman RL")

        self.board_w = config.width * cell_size
        self.board_h = config.height * cell_size
        panel = _PANEL_WIDTH if show_state else 0
        self.screen = pygame.display.set_mode((self.board_w + panel, self.board_h))
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_small = pygame.font.SysFont("consolas", 12)
        self.clock = pygame.time.Clock()
        self.fps = 8

    def _rect(self, row: int, col: int):
        return (col * self.cell, row * self.cell, self.cell, self.cell)

    def _center(self, row: int, col: int):
        return (int((col + 0.5) * self.cell), int((row + 0.5) * self.cell))

    def draw(self, game: Game, obs_builder=None) -> None:
        if self._closed:
            return
        pygame = self.pygame
        # Pump events so the window stays responsive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(_COLORS["bg"])
        overlay = self.show_state and obs_builder is not None

        blast_cells = game.blast_cells()
        bomb_cells = {bomb.pos for bomb in game.bombs}

        # Tiles.
        for row in range(game.config.height):
            for col in range(game.config.width):
                tile = int(game.grid[row, col])
                if tile == Tile.WALL:
                    color = _COLORS["wall"]
                elif tile == Tile.CRATE:
                    color = _COLORS["crate"]
                else:
                    color = _COLORS["empty"]
                pygame.draw.rect(self.screen, color, self._rect(row, col))
                pygame.draw.rect(self.screen, _COLORS["bg"], self._rect(row, col), 1)

        # Danger overlay (semi-transparent red) under entities.
        if overlay:
            self._draw_danger(game)

        # Bombs, blasts, players.
        for bomb_row, bomb_col in bomb_cells:
            self._draw_circle(bomb_row, bomb_col, _COLORS["bomb"], 0.32)
        for blast_row, blast_col in blast_cells:
            pygame.draw.rect(self.screen, _COLORS["blast"], self._rect(blast_row, blast_col))
        agent = game.agent
        if agent.alive:
            self._draw_circle(agent.pos[0], agent.pos[1], _COLORS["agent"], 0.4)
        for enemy in game.enemies:
            if enemy.alive:
                self._draw_circle(enemy.pos[0], enemy.pos[1], _COLORS["enemy"], 0.4)

        # State overlay: rays + side panel.
        if overlay:
            self._draw_rays(game, obs_builder)
            self._draw_panel(game, obs_builder)

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------ #
    def _draw_danger(self, game: Game) -> None:
        pygame = self.pygame
        surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        surf.fill((255, 50, 50, 70))
        for danger_row, danger_col in game.predict_danger_cells():
            self.screen.blit(surf, (danger_col * self.cell, danger_row * self.cell))

    def _draw_rays(self, game: Game, obs_builder) -> None:
        pygame = self.pygame
        agent = game.agent
        if not agent.alive:
            return
        origin = self._center(*agent.pos)
        for ray in obs_builder.ray_visualization():
            ray_cells = ray["cells"]
            ray_end = ray_cells[-1] if ray_cells else agent.pos
            pygame.draw.line(
                self.screen,
                _COLORS["ray"],
                origin,
                self._center(*ray_end),
                2,
            )
            for hit_type, hit_position in ray["hits"].items():
                pygame.draw.circle(
                    self.screen,
                    _HIT_COLORS[hit_type],
                    self._center(*hit_position),
                    max(4, int(self.cell * 0.14)),
                    0,
                )

    def _draw_panel(self, game: Game, obs_builder) -> None:
        pygame = self.pygame
        from bomberman.observation import SCALAR_NAMES

        panel_left = self.board_w
        pygame.draw.rect(
            self.screen,
            _COLORS["panel"],
            (panel_left, 0, _PANEL_WIDTH, self.board_h),
        )

        features = obs_builder.features()
        ray_count = len(obs_builder.dirs)
        ray_features = features[: ray_count * 4]
        scalar_features = features[ray_count * 4:]

        text_x = panel_left + 12
        text_y = 10
        text_y = self._text(
            text_x,
            text_y,
            f"step {game.step_count}   observation dim {obs_builder.size}",
            _COLORS["text"],
        )
        text_y += 6
        text_y = self._text(
            text_x,
            text_y,
            "RAYS  [wall crate enemy bomb]",
            _COLORS["text_dim"],
            small=True,
        )
        rays = obs_builder.ray_visualization()
        for ray_index, ray in enumerate(rays):
            ray_values = ray_features[ray_index * 4:ray_index * 4 + 4]
            ray_text = f"{ray['name']:<5} " + " ".join(
                f"{value:4.2f}" for value in ray_values
            )
            text_y = self._text(text_x, text_y, ray_text, _COLORS["text"], small=True)

        text_y += 8
        text_y = self._text(text_x, text_y, "SCALARS", _COLORS["text_dim"], small=True)
        for name, value in zip(SCALAR_NAMES, scalar_features):
            text_y = self._text(
                text_x,
                text_y,
                f"{name:<14} {value:5.2f}",
                _COLORS["text"],
                small=True,
            )

        text_y += 10
        text_y = self._text(text_x, text_y, "LEGEND", _COLORS["text_dim"], small=True)
        for label, color in _HIT_COLORS.items():
            pygame.draw.circle(self.screen, color, (text_x + 6, text_y + 7), 5)
            self._text(text_x + 18, text_y, label, _COLORS["text"], small=True)
            text_y += 18
        # Danger swatch.
        danger_swatch = pygame.Surface((10, 10), pygame.SRCALPHA)
        danger_swatch.fill((255, 50, 50, 160))
        self.screen.blit(danger_swatch, (text_x + 1, text_y + 2))
        self._text(text_x + 18, text_y, "danger cell", _COLORS["text"], small=True)

    def _text(self, text_x: int, text_y: int, text: str, color, small: bool = False) -> int:
        font = self.font_small if small else self.font
        surf = font.render(text, True, color)
        self.screen.blit(surf, (text_x, text_y))
        return text_y + font.get_height() + 2

    def _draw_circle(self, row: int, col: int, color, radius_fraction: float) -> None:
        self.pygame.draw.circle(
            self.screen,
            color,
            self._center(row, col),
            int(self.cell * radius_fraction),
        )

    def close(self) -> None:
        self._closed = True
        try:
            self.pygame.display.quit()
            self.pygame.quit()
        except Exception:
            pass
