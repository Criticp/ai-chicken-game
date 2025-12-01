"""
Separator Planner - Strategic Wall Builder

Manages the separator wall strategy by building turd walls on rows/columns 2 & 5
to divide the board into territories and confuse opponents.

Strategy:
- Choose row-based (rows 2 & 5) OR column-based (columns 2 & 5) walls
- Place at least 2 turds in primary axis, then complement with secondary axis
- Track progress and manage phases (SELECTING → BUILDING → COMPLETED)
- Provide scoring bonuses for strategic wall positions
"""

from typing import Tuple, List, Set, Optional, Dict
from enum import Enum
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.board import manhattan_distance


class WallPhase(Enum):
    """Wall building phases"""
    SELECTING = "selecting"          # Deciding which axis to use
    BUILDING_PRIMARY = "building_primary"    # Building first wall (row 2 or col 2)
    BUILDING_SECONDARY = "building_secondary"  # Building second wall (row 5 or col 5)
    COMPLETED = "completed"          # Both walls have 2+ turds
    MAINTAINING = "maintaining"      # Walls built, maintain presence
    ABORTED = "aborted"             # Strategy failed/abandoned


class WallAxis(Enum):
    """Wall orientation"""
    ROWS = "rows"        # Horizontal walls (rows 2 & 5)
    COLUMNS = "columns"  # Vertical walls (columns 2 & 5)


class SeparatorPlanner:
    """
    Strategic wall builder for board division.
    
    Divides the 8×8 board using turd walls on strategic lines:
    - Row-based: Horizontal walls on rows 2 and 5
    - Column-based: Vertical walls on columns 2 and 5
    
    Selection criteria:
    - Choose axis closest to player spawn
    - Avoid axes with high trapdoor risk
    - Consider enemy position (stay away from their territory)
    """
    
    # Strategic positions
    PRIMARY_ROW = 2
    SECONDARY_ROW = 5
    PRIMARY_COL = 2
    SECONDARY_COL = 5
    
    # Scoring weights
    WALL_POSITION_VALUE = 300.0      # Base value for wall position
    COMPLETION_BONUS = 100.0         # Per turd in wall
    SYNERGY_BONUS = 200.0           # Both walls active
    STRATEGIC_BONUS = 150.0         # Wall creates board split
    
    # Thresholds
    MIN_TURDS_PER_WALL = 2          # Minimum turds to consider wall "built"
    MAX_FAILED_ATTEMPTS = 3         # Abort after this many failures
    ABORT_TURN_THRESHOLD = 15       # Abort if not 40% complete by this turn
    MIN_PROGRESS_THRESHOLD = 0.4    # 40% progress needed by abort turn
    
    # Complementary requirement: total turds across primary+secondary before committing to a single-wall completion
    COMPLEMENTARY_REQUIRED_TOTAL = 2

    def __init__(self, trapdoor_tracker=None):
        """
        Initialize separator planner.
        
        Args:
            trapdoor_tracker: Optional TrapdoorTracker for risk assessment
        """
        self.tracker = trapdoor_tracker
        
        # State
        self.phase = WallPhase.SELECTING
        self.axis = None  # WallAxis.ROWS or WallAxis.COLUMNS
        
        # Wall tracking
        self.primary_wall_turds: Set[Tuple[int, int]] = set()
        self.secondary_wall_turds: Set[Tuple[int, int]] = set()
        
        # Failure tracking
        self.failed_attempts = 0
        self.last_turd_placement_turn = -1
        
        # Strategy commitment
        self.committed = False
        self.commitment_turn = -1
        
    def select_axis(self, board: "game_board.Board") -> WallAxis:
        """
        Select optimal wall axis (rows vs columns).
        
        Decision factors:
        1. Distance from player position to each axis
        2. Enemy position (stay away from their side)
        3. Trapdoor risk on each axis
        4. Existing eggs/turds
        
        Args:
            board: Current board state
            
        Returns:
            WallAxis.ROWS or WallAxis.COLUMNS
        """
        my_pos = board.chicken_player.get_location()
        enemy_pos = board.chicken_enemy.get_location()
        
        # Calculate distance to primary lines
        dist_to_row_2 = abs(my_pos[1] - self.PRIMARY_ROW)
        dist_to_col_2 = abs(my_pos[0] - self.PRIMARY_COL)
        
        # Factor 1: Proximity score (closer = better)
        row_proximity_score = 8 - dist_to_row_2
        col_proximity_score = 8 - dist_to_col_2
        
        # Factor 2: Enemy avoidance (prefer axis away from enemy)
        enemy_row = enemy_pos[1]
        enemy_col = enemy_pos[0]
        
        # If enemy is near row 2 or 5, penalize row axis
        row_enemy_penalty = 0
        if enemy_row in [self.PRIMARY_ROW, self.SECONDARY_ROW]:
            row_enemy_penalty = -5
        elif abs(enemy_row - self.PRIMARY_ROW) <= 1 or abs(enemy_row - self.SECONDARY_ROW) <= 1:
            row_enemy_penalty = -2
            
        # If enemy is near col 2 or 5, penalize column axis
        col_enemy_penalty = 0
        if enemy_col in [self.PRIMARY_COL, self.SECONDARY_COL]:
            col_enemy_penalty = -5
        elif abs(enemy_col - self.PRIMARY_COL) <= 1 or abs(enemy_col - self.SECONDARY_COL) <= 1:
            col_enemy_penalty = -2
        
        # Factor 3: Trapdoor risk
        row_risk_penalty = 0
        col_risk_penalty = 0
        
        if self.tracker:
            # Check risk on wall lines
            for x in range(8):
                # Row 2
                risk = self.tracker.get_trapdoor_risk((x, self.PRIMARY_ROW))
                if risk > 0.3:
                    row_risk_penalty -= risk * 3
                    
                # Row 5
                risk = self.tracker.get_trapdoor_risk((x, self.SECONDARY_ROW))
                if risk > 0.3:
                    row_risk_penalty -= risk * 3
                    
            for y in range(8):
                # Column 2
                risk = self.tracker.get_trapdoor_risk((self.PRIMARY_COL, y))
                if risk > 0.3:
                    col_risk_penalty -= risk * 3
                    
                # Column 5
                risk = self.tracker.get_trapdoor_risk((self.SECONDARY_COL, y))
                if risk > 0.3:
                    col_risk_penalty -= risk * 3
        
        # Calculate total scores
        row_score = row_proximity_score + row_enemy_penalty + row_risk_penalty
        col_score = col_proximity_score + col_enemy_penalty + col_risk_penalty
        
        # Select best axis
        if row_score > col_score:
            selected = WallAxis.ROWS
            print(f"[SeparatorPlanner] Selected ROW axis (score: {row_score:.1f} vs {col_score:.1f})")
        else:
            selected = WallAxis.COLUMNS
            print(f"[SeparatorPlanner] Selected COLUMN axis (score: {col_score:.1f} vs {row_score:.1f})")
        
        return selected
    
    def get_next_wall_target(self, board: "game_board.Board") -> Optional[Tuple[int, int]]:
        """
        Get the next strategic wall position to target.

        Returns:
            (x, y) position for next turd placement, or None if strategy aborted
        """
        # Check if aborted
        if self.phase == WallPhase.ABORTED:
            return None

        # Select axis if not done
        if self.phase == WallPhase.SELECTING:
            self.axis = self.select_axis(board)
            self.phase = WallPhase.BUILDING_PRIMARY
            self.committed = True
            self.commitment_turn = board.turn_count
            print(f"[SeparatorPlanner] Committed to {self.axis.value} strategy on turn {self.commitment_turn}")

        # Check abort conditions
        if self._should_abort(board):
            self.phase = WallPhase.ABORTED
            print(f"[SeparatorPlanner] Strategy ABORTED (turn {board.turn_count})")
            return None

        # Get current player position
        my_pos = board.chicken_player.get_location()

        # Helper: combined turds on both walls
        combined_turds = len(self.primary_wall_turds) + len(self.secondary_wall_turds)

        # If we haven't satisfied the complementary requirement, prefer placing on the wall
        # line that currently has fewer turds (to enforce distribution across both lines).
        if combined_turds < self.COMPLEMENTARY_REQUIRED_TOTAL:
            # Decide which wall to target (prefer the one with fewer turds)
            primary_count = len(self.primary_wall_turds)
            secondary_count = len(self.secondary_wall_turds)

            # If primary has fewer (or equal) turds, target primary; otherwise target secondary
            if primary_count <= secondary_count:
                # Ensure phase reflects choice
                self.phase = WallPhase.BUILDING_PRIMARY
                target = self._get_best_primary_position(board, my_pos)
                print(f"[SeparatorPlanner] Complementary fill -> PRIMARY target {target} (primary:{primary_count} secondary:{secondary_count})")
                return target
            else:
                self.phase = WallPhase.BUILDING_SECONDARY
                target = self._get_best_secondary_position(board, my_pos)
                print(f"[SeparatorPlanner] Complementary fill -> SECONDARY target {target} (primary:{primary_count} secondary:{secondary_count})")
                return target

        # Building primary wall (after complementary requirement satisfied)
        if self.phase == WallPhase.BUILDING_PRIMARY:
            target = self._get_best_primary_position(board, my_pos)

            # Only switch to building secondary once primary has reached MIN_TURDS_PER_WALL
            if len(self.primary_wall_turds) >= self.MIN_TURDS_PER_WALL:
                self.phase = WallPhase.BUILDING_SECONDARY
                print(f"[SeparatorPlanner] Primary wall COMPLETED ({len(self.primary_wall_turds)} turds)")
                return self._get_best_secondary_position(board, my_pos)

            return target

        # Building secondary wall
        elif self.phase == WallPhase.BUILDING_SECONDARY:
            target = self._get_best_secondary_position(board, my_pos)

            # Check if secondary wall complete
            if len(self.secondary_wall_turds) >= self.MIN_TURDS_PER_WALL:
                self.phase = WallPhase.COMPLETED
                print(f"[SeparatorPlanner] Secondary wall COMPLETED ({len(self.secondary_wall_turds)} turds)")
                print(f"[SeparatorPlanner] BOARD DIVISION ACHIEVED!")
                return None

            return target

        # Walls completed or maintaining
        else:
            return None
    
    def _get_best_primary_position(self, board: "game_board.Board", 
                                   my_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get best position on primary wall line."""
        candidates = []
        
        if self.axis == WallAxis.ROWS:
            # Row 2 - iterate all columns
            for x in range(8):
                pos = (x, self.PRIMARY_ROW)
                score = self._score_wall_position(board, pos, my_pos, is_primary=True)
                candidates.append((pos, score))
        else:
            # Column 2 - iterate all rows
            for y in range(8):
                pos = (self.PRIMARY_COL, y)
                score = self._score_wall_position(board, pos, my_pos, is_primary=True)
                candidates.append((pos, score))
        
        # Filter out invalid/occupied positions
        valid_candidates = [(pos, score) for pos, score in candidates if score > -1000]
        
        if not valid_candidates:
            return None
        
        # Return highest scoring position
        best_pos, best_score = max(valid_candidates, key=lambda x: x[1])
        return best_pos
    
    def _get_best_secondary_position(self, board: "game_board.Board",
                                     my_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get best position on secondary wall line."""
        candidates = []
        
        if self.axis == WallAxis.ROWS:
            # Row 5 - iterate all columns
            for x in range(8):
                pos = (x, self.SECONDARY_ROW)
                score = self._score_wall_position(board, pos, my_pos, is_primary=False)
                candidates.append((pos, score))
        else:
            # Column 5 - iterate all rows
            for y in range(8):
                pos = (self.SECONDARY_COL, y)
                score = self._score_wall_position(board, pos, my_pos, is_primary=False)
                candidates.append((pos, score))
        
        # Filter out invalid/occupied positions
        valid_candidates = [(pos, score) for pos, score in candidates if score > -1000]
        
        if not valid_candidates:
            return None
        
        # Return highest scoring position
        best_pos, best_score = max(valid_candidates, key=lambda x: x[1])
        return best_pos
    
    def _score_wall_position(self, board: "game_board.Board", pos: Tuple[int, int],
                            my_pos: Tuple[int, int], is_primary: bool) -> float:
        """
        Score a potential wall position.
        
        Returns:
            Score (higher = better), or -10000 if invalid
        """
        # Check if occupied
        if (pos in board.eggs_player or pos in board.eggs_enemy or
            pos in board.turds_player or pos in board.turds_enemy):
            return -10000.0
        
        score = 0.0
        
        # Base proximity bonus (closer = better)
        dist = manhattan_distance(my_pos, pos)
        proximity_bonus = max(0, 10 - dist)
        score += proximity_bonus * 10
        
        # Trapdoor risk penalty
        if self.tracker:
            risk = self.tracker.get_trapdoor_risk(pos)
            score -= risk * 200  # Heavy penalty for risky positions
        
        # Enemy avoidance
        enemy_pos = board.chicken_enemy.get_location()
        enemy_dist = manhattan_distance(enemy_pos, pos)
        if enemy_dist < 3:
            score -= 50  # Avoid placing near enemy
        
        # Continuity bonus (connect to existing wall turds)
        wall_turds = self.primary_wall_turds if is_primary else self.secondary_wall_turds
        if wall_turds:
            min_wall_dist = min(manhattan_distance(pos, t) for t in wall_turds)
            if min_wall_dist <= 2:
                score += 30  # Bonus for extending existing wall
        
        return score
    
    def _should_abort(self, board: "game_board.Board") -> bool:
        """
        Check if strategy should be aborted.
        
        Abort conditions:
        - Too many failed attempts (3+)
        - Turn > 15 and progress < 40%
        - Enemy has destroyed wall turds
        """
        # Condition 1: Too many failures
        if self.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
            return True
        
        # Condition 2: Turn threshold with low progress
        if board.turn_count > self.ABORT_TURN_THRESHOLD:
            progress = self.get_progress()
            if progress < self.MIN_PROGRESS_THRESHOLD:
                return True
        
        # Condition 3: Check if enemy destroyed our walls
        # (We'd need to track previous turds and compare)
        # For now, skip this check
        
        return False
    
    def update(self, board: "game_board.Board", placed_turd: bool, turd_loc: Optional[Tuple[int, int]]):
        """
        Update planner state after a move.
        
        Args:
            board: Current board state
            placed_turd: Whether a turd was placed this turn
            turd_loc: Location of turd if placed
        """
        if self.phase == WallPhase.ABORTED or not self.committed:
            return
        
        # Update wall tracking
        if placed_turd and turd_loc:
            if self._is_on_primary_wall(turd_loc):
                self.primary_wall_turds.add(turd_loc)
                self.last_turd_placement_turn = board.turn_count
                print(f"[SeparatorPlanner] Primary wall +1 (total: {len(self.primary_wall_turds)})")
            elif self._is_on_secondary_wall(turd_loc):
                self.secondary_wall_turds.add(turd_loc)
                self.last_turd_placement_turn = board.turn_count
                print(f"[SeparatorPlanner] Secondary wall +1 (total: {len(self.secondary_wall_turds)})")
        else:
            # Track failed attempts (if we wanted to place turd but couldn't)
            if self.phase in [WallPhase.BUILDING_PRIMARY, WallPhase.BUILDING_SECONDARY]:
                # Only count as failure if it's been >3 turns since last wall turd
                if board.turn_count - self.last_turd_placement_turn > 3:
                    self.failed_attempts += 1
    
    def _is_on_primary_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on primary wall line."""
        if self.axis == WallAxis.ROWS:
            return pos[1] == self.PRIMARY_ROW
        else:
            return pos[0] == self.PRIMARY_COL
    
    def _is_on_secondary_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on secondary wall line."""
        if self.axis == WallAxis.ROWS:
            return pos[1] == self.SECONDARY_ROW
        else:
            return pos[0] == self.SECONDARY_COL
    
    def is_wall_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on any wall line."""
        if not self.committed or self.phase == WallPhase.ABORTED:
            return False
        return self._is_on_primary_wall(pos) or self._is_on_secondary_wall(pos)
    
    def get_wall_bonus(self, pos: Tuple[int, int]) -> float:
        """
        Get scoring bonus for placing turd at this position.
        
        Returns:
            Bonus score (0 if not a wall position)
        """
        if not self.is_wall_position(pos):
            return 0.0
        
        bonus = self.WALL_POSITION_VALUE
        
        # Completion bonus based on existing turds
        if self._is_on_primary_wall(pos):
            bonus += self.COMPLETION_BONUS * len(self.primary_wall_turds)
        elif self._is_on_secondary_wall(pos):
            bonus += self.COMPLETION_BONUS * len(self.secondary_wall_turds)
        
        # Synergy bonus if both walls partially built
        if len(self.primary_wall_turds) >= 1 and len(self.secondary_wall_turds) >= 1:
            bonus += self.SYNERGY_BONUS
        
        return bonus
    
    def get_progress(self) -> float:
        """
        Get wall building progress (0.0 to 1.0).
        
        Progress metric:
        - Primary wall: 0.0 to 0.5 (need 2 turds)
        - Secondary wall: 0.5 to 1.0 (need 2 turds)
        """
        if not self.committed or self.phase == WallPhase.ABORTED:
            return 0.0
        
        # Primary wall progress (0 to 0.5)
        primary_progress = min(len(self.primary_wall_turds) / self.MIN_TURDS_PER_WALL, 1.0) * 0.5
        
        # Secondary wall progress (0 to 0.5)
        secondary_progress = min(len(self.secondary_wall_turds) / self.MIN_TURDS_PER_WALL, 1.0) * 0.5
        
        total_progress = primary_progress + secondary_progress
        return total_progress
    
    def both_walls_active(self) -> bool:
        """Check if both walls have at least 1 turd."""
        return len(self.primary_wall_turds) >= 1 and len(self.secondary_wall_turds) >= 1
    
    def get_wall_positions(self) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Get current wall turd positions.
        
        Returns:
            Dict with 'primary' and 'secondary' wall turd sets
        """
        return {
            'primary': self.primary_wall_turds.copy(),
            'secondary': self.secondary_wall_turds.copy()
        }
    
    def is_active(self) -> bool:
        """Check if wall strategy is currently active."""
        return self.committed and self.phase not in [WallPhase.ABORTED, WallPhase.COMPLETED]
    
    def get_status(self) -> str:
        """Get human-readable status."""
        if not self.committed:
            return "Not committed"
        
        progress_pct = self.get_progress() * 100
        return f"{self.phase.value} - {progress_pct:.0f}% complete ({len(self.primary_wall_turds)}+{len(self.secondary_wall_turds)} turds)"
