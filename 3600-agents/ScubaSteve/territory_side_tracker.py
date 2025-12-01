"""
Territory Side Tracker - Invasion Phase Management

Tracks which cells belong to "our side" vs "opponent side" based on spawn positions
and board geometry. Calculates saturation to trigger invasion mode.

Game Context:
- 80 rounds total (40 per agent)
- Agent A (white) goes first, Agent B (black) goes second
- Board is 8x8 (coordinates 0-7)

Strategy:
- Divide board based on spawn position (left vs right, top vs bottom)
- Track saturation of our side (% of cells with our eggs)
- Trigger invasion when our side is saturated
- Hard cutoff: stop invasion with 6-7 turns remaining (safety mode)
"""

from typing import Tuple, Set, Dict, Optional
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board


class TerritorySideTracker:
    """
    Tracks our side vs opponent side and calculates saturation.

    Division Strategy:
    - If spawned on left half (x < 4): our side = x < 4
    - If spawned on right half (x >= 4): our side = x >= 4
    - If spawned on top half (y < 4): our side = y < 4
    - If spawned on bottom half (y >= 4): our side = y >= 4

    Uses the dominant axis (horizontal or vertical split).
    """

    def __init__(self):
        self.our_side_cells: Set[Tuple[int, int]] = set()
        self.opponent_side_cells: Set[Tuple[int, int]] = set()
        self.division_axis = None  # 'horizontal' or 'vertical'
        self.initialized = False

    def initialize(self, spawn_position: Tuple[int, int]):
        """
        Initialize territory division based on spawn position.

        Args:
            spawn_position: Our spawn location (x, y)
        """
        if self.initialized:
            return

        spawn_x, spawn_y = spawn_position

        # Determine division axis based on spawn location
        # Horizontal division (left vs right)
        horizontal_distance = min(spawn_x, 7 - spawn_x)

        # Vertical division (top vs bottom)
        vertical_distance = min(spawn_y, 7 - spawn_y)

        # Choose axis that maximizes our territory (closer to edge)
        # AGGRESSIVE STRATEGY: Push into opponent territory - take 75% of board (6 columns/rows)
        # This restricts opponent's area and forces them to our side
        if horizontal_distance <= vertical_distance:
            # Horizontal division (left/right)
            self.division_axis = 'horizontal'
            if spawn_x < 4:
                # We're on left, AGGRESSIVELY take 6 columns (0-5) = 75% of board
                for x in range(8):
                    for y in range(8):
                        if x < 6:  # Changed from < 5 to < 6 (0,1,2,3,4,5)
                            self.our_side_cells.add((x, y))
                        else:
                            self.opponent_side_cells.add((x, y))
            else:
                # We're on right, AGGRESSIVELY take 6 columns (2-7) = 75% of board
                for x in range(8):
                    for y in range(8):
                        if x >= 2:  # Changed from >= 3 to >= 2 (2,3,4,5,6,7)
                            self.our_side_cells.add((x, y))
                        else:
                            self.opponent_side_cells.add((x, y))
        else:
            # Vertical division (top/bottom)
            self.division_axis = 'vertical'
            if spawn_y < 4:
                # We're on top, AGGRESSIVELY take 6 rows (0-5) = 75% of board
                for x in range(8):
                    for y in range(8):
                        if y < 6:  # Changed from < 5 to < 6 (0,1,2,3,4,5)
                            self.our_side_cells.add((x, y))
                        else:
                            self.opponent_side_cells.add((x, y))
            else:
                # We're on bottom, AGGRESSIVELY take 6 rows (2-7) = 75% of board
                for x in range(8):
                    for y in range(8):
                        if y >= 2:  # Changed from >= 3 to >= 2 (2,3,4,5,6,7)
                            self.our_side_cells.add((x, y))
                        else:
                            self.opponent_side_cells.add((x, y))

        self.initialized = True
        print(f"[TerritorySideTracker] ✓ Initialized with {self.division_axis} division")
        print(f"[TerritorySideTracker] Our side: {len(self.our_side_cells)} cells, Opponent side: {len(self.opponent_side_cells)} cells")
        print(f"[TerritorySideTracker] Spawn ({spawn_x},{spawn_y}), Our side sample: {list(self.our_side_cells)[:5]}")

    def is_our_side(self, position: Tuple[int, int]) -> bool:
        """Check if position is on our side."""
        if not self.initialized:
            return False
        return position in self.our_side_cells

    def is_opponent_side(self, position: Tuple[int, int]) -> bool:
        """Check if position is on opponent's side."""
        if not self.initialized:
            return False
        return position in self.opponent_side_cells

    def calculate_saturation(self, board: "game_board.Board") -> float:
        """
        Calculate saturation percentage of our side.

        Returns:
            Saturation percentage (0.0 to 100.0)
        """
        if not self.initialized:
            print(f"[SATURATION DEBUG] Not initialized!")
            return 0.0

        # Count how many cells on our side have our eggs
        our_eggs = set(board.eggs_player)
        cells_with_eggs = len(our_eggs & self.our_side_cells)

        # DEBUG: Print what we're seeing
        if board.turn_count % 20 == 0:  # Every 20 turns
            print(f"[SATURATION DEBUG Turn {board.turn_count}]")
            print(f"  Our eggs (board.eggs_player): {list(our_eggs)[:10]}...")  # First 10
            print(f"  Our side cells (sample): {list(self.our_side_cells)[:10]}...")
            print(f"  Cells with eggs: {cells_with_eggs}")
            print(f"  Total our side cells: {len(self.our_side_cells)}")

        # Calculate percentage
        total_cells = len(self.our_side_cells)
        if total_cells == 0:
            print(f"[SATURATION DEBUG] Total cells is 0!")
            return 0.0

        saturation = (cells_with_eggs / total_cells) * 100.0
        return saturation

    def get_unexplored_opponent_cells(self, board: "game_board.Board",
                                      exploration_tracker) -> Set[Tuple[int, int]]:
        """
        Get cells on opponent's side that we haven't visited or egged yet.

        Args:
            board: Current board state
            exploration_tracker: ExplorationTracker instance

        Returns:
            Set of unexplored opponent-side cells
        """
        if not self.initialized:
            return set()

        unexplored = set()
        our_eggs = set(board.eggs_player)

        for cell in self.opponent_side_cells:
            # Skip if we already have an egg there
            if cell in our_eggs:
                continue

            # Skip if blocked by turds
            if cell in board.turds_player or cell in board.turds_enemy:
                continue

            # Skip if recently visited (if tracker available)
            if exploration_tracker:
                if not exploration_tracker.has_egg(cell):
                    unexplored.add(cell)
            else:
                unexplored.add(cell)

        return unexplored

    def should_trigger_invasion(self, board: "game_board.Board",
                                player_turn: int,
                                current_eggs: int) -> bool:
        """
        Determine if invasion mode should be triggered.

        Conditions (ALL must be true):
        1. We have laid 15+ eggs (proven capability)
        2. 7-13 moves remaining (player turn 27-33)
        3. Our side saturation >= 70% OR we're looping (no eggs in last 5 turns)

        Args:
            board: Current board state
            player_turn: Our turn number (0-39, where 0 is our first move)
            current_eggs: Number of eggs we've laid

        Returns:
            True if invasion should be triggered
        """
        if not self.initialized:
            return False

        # Hard cutoff: Don't invade with less than 7 moves remaining
        moves_remaining = 40 - player_turn
        if moves_remaining < 7:
            return False

        # Don't invade too early (need at least 7 moves gone)
        if moves_remaining > 13:
            return False

        # Requirement: Must have at least 10 eggs (lowered for larger territory)
        if current_eggs < 10:
            return False

        # Check saturation
        saturation = self.calculate_saturation(board)

        # Trigger if saturation >= 17% (adjusted for 48-cell territory = 8 eggs)
        # Old: 20% of 40 cells = 8 eggs
        # New: 17% of 48 cells = 8 eggs (same absolute threshold)
        if saturation >= 17.0:
            print(f"[INVASION TRIGGER] Saturation {saturation:.1f}% >= 17%, triggering!")
            return True

        return False

    def should_end_invasion(self, player_turn: int) -> bool:
        """
        Check if invasion mode should end (safety cutoff).

        Args:
            player_turn: Our turn number (0-39)

        Returns:
            True if we should exit invasion mode
        """
        # Safety cutoff: End invasion with 6 moves remaining (player turn 34)
        moves_remaining = 40 - player_turn
        if moves_remaining <= 6:
            return True

        return False

    def should_avoid_our_side(self, board: "game_board.Board", player_turn: int) -> bool:
        """
        Check if we should avoid our side (already saturated).

        Used to prevent wasting moves revisiting already-filled territory.

        Args:
            board: Current board state
            player_turn: Our turn number (0-39)

        Returns:
            True if our side is saturated and we should explore elsewhere
        """
        # Don't activate until mid-game
        if player_turn < 15:
            return False

        # Check saturation
        saturation = self.calculate_saturation(board)

        # Avoid our side if >= 70% saturated
        return saturation >= 70.0

    def get_stats(self, board: "game_board.Board") -> Dict[str, any]:
        """Get current territory statistics."""
        if not self.initialized:
            return {
                'initialized': False
            }

        saturation = self.calculate_saturation(board)
        our_eggs = set(board.eggs_player)
        eggs_on_our_side = len(our_eggs & self.our_side_cells)
        eggs_on_opponent_side = len(our_eggs & self.opponent_side_cells)

        return {
            'initialized': True,
            'division_axis': self.division_axis,
            'our_side_cells': len(self.our_side_cells),
            'opponent_side_cells': len(self.opponent_side_cells),
            'saturation': saturation,
            'eggs_on_our_side': eggs_on_our_side,
            'eggs_on_opponent_side': eggs_on_opponent_side
        }

