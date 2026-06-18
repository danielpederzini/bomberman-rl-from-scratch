"""Rendering: an ASCII renderer (zero deps) and an optional pygame window.
The pygame renderer can optionally draw a "state visualisation" overlay
(``show_state=True``): danger cells are tinted, the agent's ray-casts are drawn
with markers on what they hit, and a side panel prints the live observation
vector. This is handy for recording explanatory videos.
"""

from __future__ import annotations

from bomberman.entities import Tile
from bomberman.game import Game
import cupy as cp

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
        for column in range(game.config.width):
            position = (row, column)
            if agent.alive and agent.pos == position:
                line.append(_GLYPH_AGENT)
            elif position in enemy_cells:
                line.append(_GLYPH_ENEMY)
            elif position in blast_cells:
                line.append(_GLYPH_BLAST)
            elif position in bomb_cells:
                line.append(_GLYPH_BOMB)
            else:
                tile = Tile(int(game.grid[row, column]))
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

_HIT_COLORS = {
    "wall": (200, 200, 210),
    "crate": (210, 150, 80),
    "enemy": (255, 80, 100),
    "bomb": (250, 220, 60),
}

_PANEL_WIDTH = 1180

class PygameRenderer:
    """Draws the game in a pygame window. Created lazily on first use."""

    def __init__(self, config, cell_size: int = 40, show_state: bool = False, fps: int = 8):
        """Initialize pygame window with optional state visualization panel.

        Args:
            config: Game configuration for board dimensions.
            cell_size: Pixel size of each grid cell.
            show_state: Whether to show observation panel.
            fps: Target frames per second.
        """
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
        self.fps = fps

    def _rect(self, row: int, column: int):
        """Calculate pygame rectangle for a grid cell."""
        return (column * self.cell, row * self.cell, self.cell, self.cell)

    def _center(self, row: int, column: int):
        """Calculate center pixel position for a grid cell."""
        return (int((column + 0.5) * self.cell), int((row + 0.5) * self.cell))

    def draw(self, game: Game, obs_builder=None, network=None) -> None:
        """Render the game board and optionally the state panel.

        Args:
            game: Current game state to render.
            obs_builder: Optional observation builder for state panel.
            network: Optional neural network for visualization.
        """
        if self._closed:
            return
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(_COLORS["bg"])
        overlay = self.show_state and obs_builder is not None

        blast_cells = game.blast_cells()
        bomb_cells = {bomb.pos for bomb in game.bombs}

        self._draw_tiles(game)

        if overlay:
            self._draw_danger(game)

        for bomb_row, bomb_column in bomb_cells:
            self._draw_circle(bomb_row, bomb_column, _COLORS["bomb"], 0.32)
        for blast_row, blast_column in blast_cells:
            pygame.draw.rect(self.screen, _COLORS["blast"], self._rect(blast_row, blast_column))
        agent = game.agent
        if agent.alive:
            self._draw_circle(agent.pos[0], agent.pos[1], _COLORS["agent"], 0.4)
        for enemy in game.enemies:
            if enemy.alive:
                self._draw_circle(enemy.pos[0], enemy.pos[1], _COLORS["enemy"], 0.4)

        if overlay:
            self._draw_panel(game, obs_builder, network)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_tiles(self, game: Game) -> None:
        """Draw the game board tiles (walls, crates, empty spaces)."""
        for row in range(game.config.height):
            for column in range(game.config.width):
                tile = int(game.grid[row, column])
                if tile == Tile.WALL:
                    color = _COLORS["wall"]
                elif tile == Tile.CRATE:
                    color = _COLORS["crate"]
                else:
                    color = _COLORS["empty"]
                self.pygame.draw.rect(self.screen, color, self._rect(row, column))
                self.pygame.draw.rect(self.screen, _COLORS["bg"], self._rect(row, column), 1)

    def _draw_danger(self, game: Game) -> None:
        """Draw semi-transparent danger overlay on threatened cells."""
        pygame = self.pygame
        surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        surf.fill((255, 50, 50, 70))
        blast_cells = game.blast_cells()
        for danger_row, danger_column in game.predict_danger_cells():
            if (danger_row, danger_column) not in blast_cells:
                self.screen.blit(surf, (danger_column * self.cell, danger_row * self.cell))

    def _draw_panel(self, game: Game, obs_builder, network=None) -> None:
        """Draw the observation panel with channel grids and network visualization."""
        from bomberman.observation import SCALAR_NAMES, _N_CHANNELS
        pygame = self.pygame
        panel_left = self.board_w
        obs_width = 380

        pygame.draw.rect(
            self.screen,
            _COLORS["panel"],
            (panel_left, 0, _PANEL_WIDTH, self.board_h),
        )
        pygame.draw.line(
            self.screen,
            (60, 60, 70),
            (panel_left + obs_width, 0),
            (panel_left + obs_width, self.board_h),
            2,
        )
        if network is not None:
            self._draw_network_diagram(obs_builder, network, panel_left + obs_width + 10, 10)

        features = obs_builder.features()
        height, width, _ = obs_builder.board_shape
        spatial_size = height * width * _N_CHANNELS
        spatial = features[:spatial_size].reshape(height, width, _N_CHANNELS)
        scalar_features = features[spatial_size:]

        grids_x = panel_left + 12
        text_y = 10
        text_y = self._text(
            grids_x, text_y,
            f"step {game.step_count}   obs dim {obs_builder.size}",
            _COLORS["text"],
        )
        text_y += 2

        channel_names = ["tile", "danger", "blast", "bomb_t", "entity"]
        mini = 10
        grid_gap_x = 8
        grid_gap_y = 6

        def draw_single_grid(channel_index, start_x, start_y):
            """Draw a single channel grid at the given position."""
            self._text(start_x, start_y, channel_names[channel_index], _COLORS["text_dim"], small=True)
            grid_y = start_y + 12

            for grid_row in range(height):
                for grid_column in range(width):
                    value = float(spatial[grid_row, grid_column, channel_index])

                    if channel_index == 0:
                        if value >= 0.9:
                            color = _COLORS["wall"]
                        elif value >= 0.4:
                            color = _COLORS["crate"]
                        else:
                            color = _COLORS["empty"]
                    elif channel_index == 1:
                        intensity = int(value * 200)
                        color = (intensity, 0, 0)
                    elif channel_index == 2:
                        intensity = int(value * 240)
                        color = (intensity, int(intensity * 0.58), 0)
                    elif channel_index == 3:
                        heat = int(value * 220)
                        color = (heat, heat // 2, 0)
                    else:
                        if value >= 0.9:
                            color = _COLORS["agent"]
                        elif value >= 0.4:
                            color = _COLORS["enemy"]
                        else:
                            color = _COLORS["empty"]

                    pygame.draw.rect(
                        self.screen, color,
                        (start_x + grid_column * mini, grid_y + grid_row * mini, mini - 1, mini - 1),
                    )
            return grid_y + height * mini

        grid_width = width * mini
        grid_height = height * mini
        column_positions = [grids_x, grids_x + grid_width + grid_gap_x, grids_x + 2 * (grid_width + grid_gap_x)]
        row_0_y = text_y
        row_1_y = row_0_y + 12 + grid_height + grid_gap_y

        draw_single_grid(0, column_positions[0], row_0_y)
        draw_single_grid(1, column_positions[1], row_0_y)
        draw_single_grid(2, column_positions[2], row_0_y)
        draw_single_grid(3, column_positions[0], row_1_y)
        draw_single_grid(4, column_positions[1], row_1_y)

        scalar_y = row_1_y + 12 + grid_height + 8
        self._text(grids_x, scalar_y, "SCALARS", _COLORS["text_dim"], small=True)
        scalar_y += 14

        font = self.font_small
        line_height = font.get_height() + 1

        for name, value in zip(SCALAR_NAMES, scalar_features):
            surf = font.render(f"{name:<14} {float(value):5.2f}", True, _COLORS["text"])
            self.screen.blit(surf, (grids_x, scalar_y))
            scalar_y += line_height



    def _activation_color(self, value: float) -> tuple[int, int, int]:
        """Map activation value to RGB color.

        Negative values: blue (intensity based on magnitude)
        Positive values: orange (intensity based on magnitude)
        """
        if value < 0:
            intensity = min(255, int(abs(value) * 100))
            return (0, 100, 200 + min(55, intensity))
        else:
            intensity = min(255, int(value * 100))
            return (200 + min(55, intensity), 120 + min(80, intensity // 2), 40)

    def _draw_network_diagram(self, obs_builder, network, start_x: int, start_y: int) -> None:
        """Draw neural network activation flow on the right side panel."""
        pygame = self.pygame

        state = obs_builder.features()
        state_cp = cp.array(state.reshape(1, -1), dtype=cp.float32)
        q_values, activations = network.forward_with_activations(state_cp)
        q_values_np = q_values.get().flatten()
        selected_action = int(cp.argmax(q_values).get())

        layer_names = ["H1", "H2", "H3", "OUTPUT"]
        layer_sizes = [512, 256, 128, 6]

        all_activations = [activation.get().flatten() for activation in activations]

        cell_size = 10
        gap = 1
        max_width = 780

        y = start_y
        total_neurons = sum(layer_sizes)
        y = self._text(start_x, y, f"NETWORK ({total_neurons} neurons)", _COLORS["text_dim"], small=True)
        y += 8

        for name, layer_activations, total_neurons in zip(layer_names, all_activations, layer_sizes):
            y = self._text(start_x, y, f"{name} ({total_neurons})", _COLORS["text_dim"], small=True)
            y += 2

            columns = min(total_neurons, max_width // (cell_size + gap))
            rows = (total_neurons + columns - 1) // columns

            for i in range(total_neurons):
                row = i // columns
                column = i % columns
                value = float(layer_activations[i])
                color = self._activation_color(value)
                x = start_x + column * (cell_size + gap)
                cell_y = y + row * (cell_size + gap)
                pygame.draw.rect(self.screen, color, (x, cell_y, cell_size, cell_size))

            y += rows * (cell_size + gap) + 8

        y += 4
        y = self._text(start_x, y, "Q-VALUES", _COLORS["text_dim"], small=True)
        y += 4

        action_names = ["UP", "DN", "LT", "RT", "BOM", "WT"]
        for i, (name, q_value) in enumerate(zip(action_names, q_values_np)):
            is_selected = (i == selected_action)
            color = _COLORS["ray"] if is_selected else _COLORS["text"]
            marker = " <<" if is_selected else ""
            q_text = f"{name}:{q_value:5.2f}{marker}"
            y = self._text(start_x, y, q_text, color, small=True)

    def _text(self, text_x: int, text_y: int, text: str, color, small: bool = False) -> int:
        """Render text at the given position and return the next Y coordinate."""
        font = self.font_small if small else self.font
        surf = font.render(text, True, color)
        self.screen.blit(surf, (text_x, text_y))
        return text_y + font.get_height() + 2

    def _draw_circle(self, row: int, column: int, color, radius_fraction: float) -> None:
        """Draw a colored circle at the given grid position."""
        self.pygame.draw.circle(
            self.screen,
            color,
            self._center(row, column),
            int(self.cell * radius_fraction),
        )

    def close(self) -> None:
        """Close the pygame window and cleanup resources."""
        self._closed = True
        try:
            self.pygame.display.quit()
            self.pygame.quit()
        except Exception:
            pass

