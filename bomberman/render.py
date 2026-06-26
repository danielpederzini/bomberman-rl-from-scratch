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
    "reward": (100, 255, 150),  # Green for positive rewards
    "penalty": (255, 80, 80),   # Red for negative rewards
}

_MAX_CELL = 40
_MIN_CELL = 10
_SCREEN_MARGIN = 0.9
_STATS_PANEL_WIDTH = 240
_MIN_FEATURE_PANEL_WIDTH = 280
_MIN_STATS_PANEL_WIDTH = 210
_MAX_UI_SCALE = 1.0
_MIN_UI_SCALE = 0.6

class PygameRenderer:
    """Draws the game in a pygame window. Created lazily on first use."""

    def __init__(self, config, cell_size: int | None = None, show_state: bool = False,
                 fps: int = 8, max_window_size: tuple[int, int] | None = None):
        """Initialize pygame window with optional state visualization panel.

        Args:
            config: Game configuration for board dimensions.
            cell_size: Fixed pixel size per cell. If None, the cell size is
                auto-fit to the monitor (or ``max_window_size``) so the whole
                window stays on-screen for any board dimensions.
            show_state: Whether to show observation panel.
            fps: Target frames per second.
            max_window_size: Optional (width, height) budget overriding the
                detected monitor size. Useful for tests and manual caps.
        """
        import pygame
        self.pygame = pygame
        self.config = config
        self.show_state = show_state
        self._closed = False

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Bomberman RL")

        screen_budget_w, screen_budget_h = self._screen_budget(max_window_size)

        self.panel_w = self._dashboard_width(config, screen_budget_w) if show_state else 0
        self.feature_panel_w, self.stats_panel_w = (
            self._split_dashboard_width(self.panel_w) if show_state else (0, 0)
        )

        self.cell = self._fit_cell_size(config, cell_size, self.panel_w, screen_budget_w, screen_budget_h)

        self.board_w = config.width * self.cell
        self.board_h = config.height * self.cell

        self.ui_scale = 1.0
        self.panel_h = 0
        self.top_region_h = self.board_h

        if show_state:
            natural_panel_h = self._calculate_panel_height(self.feature_panel_w, 1.0)
            self.ui_scale = max(_MIN_UI_SCALE, min(_MAX_UI_SCALE, self.board_h / max(1, natural_panel_h)))
            self.panel_h = self.board_h

        self.left_region_w = self.board_w

        if show_state:
            window_w = self.left_region_w + self.panel_w
            window_h = self.board_h
        else:
            window_w = self.left_region_w
            window_h = self.board_h

        self.screen = pygame.display.set_mode((window_w, window_h))
        self.font = pygame.font.SysFont("consolas", max(10, int(14 * self.ui_scale)))
        self.font_small = pygame.font.SysFont("consolas", max(8, int(12 * self.ui_scale)))
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.event_history: list[tuple[int, str, float]] = []
        self.episode_metrics: dict[str, float] = {
            "reward": 0.0,
            "steps": 0,
            "crates": 0,
            "kills": 0,
        }
        self.current_qvalues: list[float] = [0.0] * 6
        self.valid_actions: set[int] = set(range(6))
        self.selected_action: int = 0

    def record_event(self, step: int, event_type: str, details: str = "", reward: float = 0.0) -> None:
        """Record a game event for the event log (max 6 entries)."""
        self.event_history.append((step, f"{event_type} {details}", reward))
        if len(self.event_history) > 6:
            self.event_history.pop(0)

    def update_episode_metrics(self, reward: float, steps: int, crates: int, kills: int) -> None:
        """Update cumulative episode statistics."""
        self.episode_metrics["reward"] = reward
        self.episode_metrics["steps"] = steps
        self.episode_metrics["crates"] = crates
        self.episode_metrics["kills"] = kills

    def record_qvalues(self, q_values: list[float], selected_action: int,
                        valid_actions: list[int] | None = None) -> None:
        """Store Q-values for visualization.

        Args:
            q_values: Raw (unmasked) Q-values for all actions.
            selected_action: The action actually taken (must be a valid action).
            valid_actions: Indices of actions that weren't masked out. If None,
                all actions are treated as valid (no masking shown).
        """
        self.current_qvalues = q_values
        self.selected_action = selected_action
        self.valid_actions = (
            set(valid_actions) if valid_actions is not None else set(range(len(q_values)))
        )

    def _rect(self, row: int, column: int):
        """Calculate pygame rectangle for a grid cell."""
        return (column * self.cell, row * self.cell, self.cell, self.cell)

    def _center(self, row: int, column: int):
        """Calculate center pixel position for a grid cell."""
        return (int((column + 0.5) * self.cell), int((row + 0.5) * self.cell))

    def _screen_budget(self, max_window_size: tuple[int, int] | None) -> tuple[int, int]:
        """Return the usable window budget in pixels."""
        if max_window_size is not None:
            return max_window_size

        display_info = self.pygame.display.Info()
        return (
            int(display_info.current_w * _SCREEN_MARGIN),
            int(display_info.current_h * _SCREEN_MARGIN),
        )

    def _dashboard_width(self, config, screen_budget_w: int) -> int:
        """Decide total dashboard (panel) width directly from the screen budget
        and the board's cell count, with no dependency on cell size in pixels.

        This removes the old chicken-and-egg problem where panel width was
        estimated from `config.width * _MAX_CELL` before the real cell size
        was known, then recomputed afterward from the real (usually smaller)
        `board_w` -- two different numbers used to make two different
        decisions, which is what caused the small-board layout breakage.
        """
        min_total = _MIN_FEATURE_PANEL_WIDTH + _MIN_STATS_PANEL_WIDTH
        desired_w = max(min_total, int(config.width * 24))
        max_affordable = max(min_total, int(screen_budget_w * 0.45))
        return min(desired_w, max_affordable)

    def _split_dashboard_width(self, dashboard_w: int) -> tuple[int, int]:
        """Split the dashboard into feature/network and stats/events columns."""
        stats_w = max(_MIN_STATS_PANEL_WIDTH, min(int(dashboard_w * 0.38), _STATS_PANEL_WIDTH))
        feature_w = max(_MIN_FEATURE_PANEL_WIDTH, dashboard_w - stats_w)
        return feature_w, dashboard_w - feature_w

    def _fit_cell_size(
        self,
        config,
        cell_size: int | None,
        panel_w: int,
        screen_budget_w: int,
        screen_budget_h: int,
    ) -> int:
        """Choose the largest cell size that keeps the board+panel on-screen.

        `panel_w` must be the *real* dashboard width already decided by
        `_dashboard_width` (0 if show_state is False) -- not an estimate --
        so this is now a single algebraic pass with no re-fitting needed.
        """
        if cell_size is not None:
            return max(_MIN_CELL, min(_MAX_CELL, int(cell_size)))

        avail_w = max(_MIN_CELL, screen_budget_w - panel_w)
        avail_h = max(_MIN_CELL, screen_budget_h)

        fitted_cell = min(
            _MAX_CELL,
            avail_w // max(1, config.width),
            avail_h // max(1, config.height),
        )
        return max(_MIN_CELL, int(fitted_cell))

    def _calculate_panel_height(self, feature_panel_w: int, scale: float | None = None) -> int:
        """Calculate required height for the observation panel.

        Includes: grid visualization, scalars, network diagram.

        `feature_panel_w` must be the actual feature-panel pixel width that
        will be used to draw (from `_split_dashboard_width`), not the fixed
        `_FEATURE_PANEL_WIDTH` constant -- otherwise `ui_scale` is computed
        against a width that's wrong for small boards (where the panel gets
        floored to `_MIN_FEATURE_PANEL_WIDTH`), and content sized for the
        assumed width gets crammed into the real, narrower panel.
        """
        if scale is None:
            scale = self.ui_scale
        cell_size = max(2, int(6 * scale))
        gap = max(1, int(1 * scale))
        max_width = max(1, int((feature_panel_w - 20) * scale))
        layer_sizes = [512, 256, 128, 6]

        header_height = max(24, int(40 * scale))
        grid_section = max(90, int(220 * scale))
        scalars_section = max(90, int(200 * scale))
        divider_height = max(12, int(20 * scale))

        network_height = 0
        for total_neurons in layer_sizes:
            network_height += 18  # Label + spacing
            columns = min(total_neurons, max_width // (cell_size + gap))
            rows = (total_neurons + columns - 1) // columns
            network_height += rows * (cell_size + gap) + 6

        return header_height + grid_section + scalars_section + divider_height + network_height

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
            self._draw_game_stats(game)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_game_stats(self, game: Game) -> None:
        """Draw game statistics, event log, and Q-value bars in the stats panel."""
        font = self.font_small
        scale = self.ui_scale
        margin = max(6, int(12 * scale))
        line_height = font.get_height() + max(1, int(2 * scale))

        panel_left = self.left_region_w + self.feature_panel_w
        panel_rect = (panel_left, 0, self.stats_panel_w, self.panel_h)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect)

        stats_x = panel_left + margin
        stats_y = margin

        self._text(stats_x, stats_y, "GAME STATS", _COLORS["text_dim"], small=True)
        stats_y += line_height + max(2, int(4 * scale))

        agent_alive = "alive" if game.agent.alive else "dead"
        active_bombs = len(game.bombs)
        enemies_alive = sum(1 for enemy in game.enemies if enemy.alive)
        crates_remaining = sum(
            1 for row in range(game.config.height)
            for column in range(game.config.width)
            if game.grid[row, column] == Tile.CRATE
        )

        stats = [
            f"step: {game.step_count}/{game.config.max_steps}",
            f"agent: {agent_alive}",
            f"bombs: {active_bombs}",
            f"enemies: {enemies_alive}/{game.config.n_enemies}",
            f"crates: {crates_remaining}",
            f"reward: {self.episode_metrics['reward']:.2f}",
            f"crates_destroyed: {int(self.episode_metrics['crates'])}",
            f"enemies_killed: {int(self.episode_metrics['kills'])}",
        ]

        for stat in stats:
            surf = font.render(stat, True, _COLORS["text"])
            self.screen.blit(surf, (stats_x, stats_y))
            stats_y += line_height

        events_x = stats_x
        events_y = stats_y + max(5, int(10 * scale))

        self._text(events_x, events_y, "EVENTS", _COLORS["text_dim"], small=True)
        events_y += line_height + max(2, int(4 * scale))

        if self.event_history:
            for step, event_text, reward in self.event_history:
                if reward > 0:
                    color = _COLORS["reward"]
                elif reward < 0:
                    color = _COLORS["penalty"]
                else:
                    color = _COLORS["text"]
                display_text = f"{step}: {event_text}"
                surf = font.render(display_text, True, color)
                self.screen.blit(surf, (events_x, events_y))
                events_y += line_height
        else:
            surf = font.render("(no events)", True, _COLORS["text_dim"])
            self.screen.blit(surf, (events_x, events_y))
            events_y += line_height

        qval_y = max(stats_y, events_y) + max(4, int(8 * scale))
        self._text(stats_x, qval_y, "Q-VALUES", _COLORS["text_dim"], small=True)
        qval_y += line_height + max(2, int(4 * scale))

        self._draw_qvalue_bars(stats_x, qval_y)
        self.screen.set_clip(previous_clip)

    def _draw_qvalue_bars(self, start_x: int, start_y: int) -> None:
        """Draw horizontal bar chart of Q-values.

        Masked (invalid) actions are drawn as empty/dimmed bars with no
        numeric value, since their raw Q-values are meaningless (the agent
        could never select them) and would otherwise distort the min/max
        scale used to normalize the valid bars.
        """
        pygame = self.pygame
        font = self.font_small

        action_names = ["UP", "DN", "LT", "RT", "BOM", "WT"]
        available_w = max(80, self.stats_panel_w - (start_x - (self.left_region_w + self.feature_panel_w)) - max(16, int(28 * self.ui_scale)) - 54)
        max_bar_width = min(max(70, int(120 * self.ui_scale)), available_w)
        bar_height = max(6, int(10 * self.ui_scale))
        bar_gap = max(3, int(4 * self.ui_scale))

        valid_q_values = [
            q_value for i, q_value in enumerate(self.current_qvalues)
            if i in self.valid_actions
        ]
        if valid_q_values:
            q_min = min(valid_q_values)
            q_max = max(valid_q_values)
        else:
            q_min, q_max = 0.0, 0.0
        q_range = max(0.01, q_max - q_min)

        for i, (name, q_value) in enumerate(zip(action_names, self.current_qvalues)):
            y = start_y + i * (bar_height + bar_gap)
            is_valid = i in self.valid_actions
            is_selected = (i == self.selected_action)

            if not is_valid:
                color = _COLORS["text_dim"]
            else:
                color = _COLORS["ray"] if is_selected else _COLORS["text"]
            surf = font.render(f"{name}", True, color)
            self.screen.blit(surf, (start_x, y))

            bar_x = start_x + 35
            bar_width = max_bar_width
            pygame.draw.rect(self.screen, (40, 40, 50), (bar_x, y, bar_width, bar_height))

            if is_valid:
                normalized = (q_value - q_min) / q_range
                fill_width = int(normalized * bar_width)
                bar_color = _COLORS["ray"] if is_selected else (100, 150, 200)
                pygame.draw.rect(self.screen, bar_color, (bar_x, y, fill_width, bar_height))

                value_x = bar_x + bar_width + 8
                value_surf = font.render(f"{q_value:+.2f}", True, color)
                self.screen.blit(value_surf, (value_x, y))
            else:
                value_x = bar_x + bar_width + 8
                value_surf = font.render("masked", True, _COLORS["text_dim"])
                self.screen.blit(value_surf, (value_x, y))

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
        panel_left = self.left_region_w
        scale = self.ui_scale
        margin = max(6, int(12 * scale))
        obs_width = self.feature_panel_w - 2 * margin

        panel_rect = (panel_left, 0, self.panel_w, self.panel_h)
        pygame.draw.rect(
            self.screen,
            _COLORS["panel"],
            panel_rect,
        )
        pygame.draw.line(
            self.screen,
            (40, 40, 50),
            (panel_left + self.feature_panel_w, 0),
            (panel_left + self.feature_panel_w, self.panel_h),
            1,
        )
        previous_clip = self.screen.get_clip()
        feature_rect = (panel_left, 0, self.feature_panel_w, self.panel_h)
        self.screen.set_clip(feature_rect)

        features = obs_builder.features()
        height, width, _ = obs_builder.board_shape
        spatial_size = height * width * _N_CHANNELS
        spatial = features[:spatial_size].reshape(height, width, _N_CHANNELS)
        scalar_features = features[spatial_size:]

        grids_x = panel_left + margin
        text_y = max(5, int(10 * scale))
        text_y = self._text(
            grids_x, text_y,
            f"step {game.step_count}   obs dim {obs_builder.size}",
            _COLORS["text"],
        )
        text_y += max(1, int(2 * scale))

        channel_names = ["tile", "danger", "blast", "bomb_t", "entity"]
        mini = max(4, int(10 * scale))
        grid_gap_x = max(4, int(8 * scale))
        grid_gap_y = max(3, int(6 * scale))

        def draw_single_grid(channel_index, start_x, start_y):
            """Draw a single channel grid at the given position."""
            self._text(start_x, start_y, channel_names[channel_index], _COLORS["text_dim"], small=True)
            grid_y = start_y + max(8, int(12 * scale))

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
        row_1_y = row_0_y + max(8, int(12 * scale)) + grid_height + grid_gap_y

        draw_single_grid(0, column_positions[0], row_0_y)
        draw_single_grid(1, column_positions[1], row_0_y)
        draw_single_grid(2, column_positions[2], row_0_y)
        draw_single_grid(3, column_positions[0], row_1_y)
        draw_single_grid(4, column_positions[1], row_1_y)

        scalar_y = row_1_y + max(8, int(12 * scale)) + grid_height + max(4, int(8 * scale))
        self._text(grids_x, scalar_y, "SCALARS", _COLORS["text_dim"], small=True)
        scalar_y += max(10, int(14 * scale))

        font = self.font_small
        line_height = font.get_height() + max(1, int(scale))

        for name, value in zip(SCALAR_NAMES, scalar_features):
            surf = font.render(f"{name:<14} {float(value):5.2f}", True, _COLORS["text"])
            self.screen.blit(surf, (grids_x, scalar_y))
            scalar_y += line_height

        if network is not None:
            divider_y = scalar_y + max(5, int(10 * scale))
            pygame.draw.line(
                self.screen,
                (60, 60, 70),
                (panel_left + margin, divider_y),
                (panel_left + self.panel_w - margin, divider_y),
                2,
            )
            self._draw_network_diagram(obs_builder, network, grids_x, divider_y + max(6, int(10 * scale)), obs_width)

        self.screen.set_clip(previous_clip)

    def _activation_color(self, normalized: float) -> tuple[int, int, int]:
        """Map a normalized activation (0-1) to an RGB color.

        0 (lowest in layer): dark blue
        1 (highest in layer): bright orange
        """
        t = max(0.0, min(1.0, normalized))
        r = int(20 + t * 235)
        g = int(30 + t * 150)
        b = int(120 - t * 100)
        return (r, g, b)

    def _draw_network_diagram(self, obs_builder, network, start_x: int, start_y: int, max_width: int) -> None:
        """Draw neural network activation flow on the right side panel."""
        pygame = self.pygame

        state = obs_builder.features()
        state_cp = cp.array(state.reshape(1, -1), dtype=cp.float32)
        _, activations = network.forward_with_activations(state_cp)

        layer_names = ["H1", "H2", "H3", "OUTPUT"]
        layer_sizes = [512, 256, 128, 6]
        all_activations = [activation.get().flatten() for activation in activations]

        cell_size = max(2, int(6 * self.ui_scale))
        gap = max(1, int(1 * self.ui_scale))

        y = start_y
        total_neurons_all = sum(layer_sizes)
        y = self._text(start_x, y, f"NETWORK ({total_neurons_all} neurons)", _COLORS["text_dim"], small=True)
        y += max(3, int(6 * self.ui_scale))

        for name, layer_activations, total_neurons in zip(layer_names, all_activations, layer_sizes):
            y = self._text(start_x, y, f"{name} ({total_neurons})", _COLORS["text_dim"], small=True)
            y += 2

            columns = min(total_neurons, max_width // (cell_size + gap))
            rows = (total_neurons + columns - 1) // columns

            # Normalize per-layer so variations are visible regardless of scale
            layer_min = float(layer_activations.min())
            layer_max = float(layer_activations.max())
            layer_range = max(1e-6, layer_max - layer_min)

            for i in range(total_neurons):
                row = i // columns
                column = i % columns
                normalized = (float(layer_activations[i]) - layer_min) / layer_range
                color = self._activation_color(normalized)
                x = start_x + column * (cell_size + gap)
                cell_y = y + row * (cell_size + gap)
                pygame.draw.rect(self.screen, color, (x, cell_y, cell_size, cell_size))

            y += rows * (cell_size + gap) + max(3, int(6 * self.ui_scale))

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
