"""
Debug Output Module for ScubaSteve V7
Provides clear, formatted status updates during gameplay
"""

from typing import Tuple, Dict, Optional
from game import board as game_board
from game.enums import MoveType


class DebugReporter:
    """
    Handles formatted debug output for agent decision-making

    Shows:
    - Move information (position, direction, action)
    - Evaluation breakdown (neural vs heuristic scores)
    - Performance metrics (eggs, efficiency, time)
    - Decision reasoning
    """

    def __init__(self):
        self.turn_count = 0
        self.last_report_turn = -1

    def report_move(
        self,
        board: "game_board.Board",
        current_loc: Tuple[int, int],
        dest_loc: Tuple[int, int],
        move: Tuple,
        time_left: float,
        neural_score: Optional[float] = None,
        heuristic_score: Optional[float] = None,
        reasoning: str = "",
        pos_history: list = None
    ):
        """
        Print comprehensive move report

        Args:
            board: Current board state
            current_loc: Starting position
            dest_loc: Destination position
            move: (Direction, MoveType) tuple
            time_left: Remaining time in seconds
            neural_score: Neural network evaluation (if available)
            heuristic_score: Heuristic evaluation
            reasoning: Text explanation of move choice
            pos_history: List of recent positions for efficiency calc
        """
        self.turn_count += 1

        # Calculate metrics
        eggs = board.chicken_player.eggs_laid
        turds_left = board.chicken_player.turds_left
        moves_remaining = 40 - self.turn_count
        egg_efficiency = (eggs / max(1, self.turn_count)) * 100

        # Movement efficiency
        if pos_history:
            unique_positions = len(set(pos_history))
            movement_efficiency = (unique_positions / max(1, len(pos_history))) * 100
        else:
            unique_positions = 0
            movement_efficiency = 0.0

        # Format output
        print(f"\n{'='*75}")
        print(f"[V7 Turn {self.turn_count}] ScubaSteve Decision")
        print(f"{'='*75}")

        # Move info
        direction_symbol = self._get_direction_symbol(current_loc, dest_loc)
        action_color = self._get_action_color(move[1])
        print(f"  Move: {current_loc} {direction_symbol} {dest_loc} [{action_color}{move[1].name}\033[0m]")

        # Evaluation breakdown
        if neural_score is not None and heuristic_score is not None:
            blended = 0.5 * neural_score + 0.5 * heuristic_score
            print(f"  Evaluation:")
            print(f"    Neural:    {neural_score:+8.2f}")
            print(f"    Heuristic: {heuristic_score:+8.2f}")
            print(f"    Blended:   {blended:+8.2f} (50-50 hybrid)")
        elif heuristic_score is not None:
            print(f"  Evaluation: {heuristic_score:+8.2f} (heuristic-only)")

        # Reasoning
        if reasoning:
            print(f"  Reasoning: {reasoning}")

        # Performance metrics
        print(f"\n  Performance Metrics:")
        print(f"    Eggs:      {eggs}/40 ({egg_efficiency:.1f}% efficiency)")
        print(f"    Movement:  {unique_positions}/{len(pos_history) if pos_history else 0} unique ({movement_efficiency:.1f}%)")
        print(f"    Turds:     {turds_left}/5 remaining ({5 - turds_left} used)")
        print(f"    Time:      {time_left:.1f}s | Moves remaining: {moves_remaining}")

        # Progress bar for eggs
        target_eggs = 25
        progress = min(100, (eggs / target_eggs) * 100)
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"    Progress:  [{bar}] {eggs}/{target_eggs} target")

        print(f"{'='*75}\n")

        self.last_report_turn = self.turn_count

    def report_quick_move(
        self,
        turn: int,
        current_loc: Tuple[int, int],
        dest_loc: Tuple[int, int],
        move_type: MoveType,
        eggs: int,
        reason: str = ""
    ):
        """
        Print quick one-line move report (for non-reporting turns)
        """
        direction_symbol = self._get_direction_symbol(current_loc, dest_loc)
        action = move_type.name
        print(f"[V7 T{turn:02d}] {current_loc} {direction_symbol} {dest_loc} [{action}] - {eggs} eggs {reason}")

    def _get_direction_symbol(self, from_loc: Tuple[int, int], to_loc: Tuple[int, int]) -> str:
        """Get arrow symbol for direction"""
        dx = to_loc[0] - from_loc[0]
        dy = to_loc[1] - from_loc[1]

        if dx > 0:
            return "→"
        elif dx < 0:
            return "←"
        elif dy > 0:
            return "↓"
        elif dy < 0:
            return "↑"
        else:
            return "•"

    def _get_action_color(self, move_type: MoveType) -> str:
        """Get ANSI color code for action type"""
        if move_type == MoveType.EGG:
            return "\033[92m"  # Green
        elif move_type == MoveType.TURD:
            return "\033[93m"  # Yellow
        else:
            return "\033[94m"  # Blue

    def report_opening_book(self, turn: int, move: Tuple):
        """Report opening book usage"""
        print(f"[V7 T{turn:02d}] 📖 Opening Book Move: {move}")

    def report_turd_blocked(self, turn: int, reason: str):
        """Report turd placement blocked"""
        print(f"[V7 T{turn:02d}] 🚫 Turd blocked: {reason}")

    def report_turd_approved(self, turn: int, score: float):
        """Report turd placement approved"""
        print(f"[V7 T{turn:02d}] ✓ Turd approved (strategic value: {score:.1f})")

    def report_safety_warning(self, position: Tuple[int, int], risk: float):
        """Report safety concern"""
        print(f"[V7 Safety] ⚠️  Position {position} has {risk*100:.1f}% trap risk")

    def report_death(self, trap_location: Tuple[int, int]):
        """Report death from trapdoor"""
        print(f"[V7 Safety] ☠️  DEATH! Trapdoor at {trap_location} - marked for avoidance")

