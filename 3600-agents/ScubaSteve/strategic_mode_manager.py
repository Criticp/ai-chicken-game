from enum import Enum
from typing import Optional


class StrategyMode(Enum):
    HYBRID = "hybrid"
    WALL_BUILDING = "wall_building"
    EGG_MAXIMIZE = "egg_maximize"


class StrategyManager:
    """Simple strategy coordinator for ScubaSteve.

    Decides between wall-building (separator), egg-maximization, or hybrid mode.
    This is intentionally conservative and uses simple heuristics with hysteresis to
    avoid flipping too often.
    """

    def __init__(self, separator_planner=None):
        self.separator_planner = separator_planner

        # Hysteresis / counters
        self.last_mode: StrategyMode = StrategyMode.HYBRID
        self.mode_since_turn = 0
        self.wall_attempts = 0
        self.successful_wall_attempts = 0

    def select_strategy(self, board, move_count: int, separator_progress: float) -> StrategyMode:
        """Pick a strategy based on simple rules.

        Rules (conservative defaults):
        - If separator planner is active and progress is below 1.0, prefer WALL_BUILDING early-mid game
        - If we've been building for many turns with little progress, switch to EGG_MAXIMIZE
        - Otherwise, stay in HYBRID which lets the search engine decide
        """
        # If no separator planner, prefer eggs/hybrid
        if not self.separator_planner:
            self.last_mode = StrategyMode.HYBRID
            return self.last_mode

        # Prefer wall building early if the planner is active and not yet complete
        if self.separator_planner.is_active() and separator_progress < 0.9 and move_count < 40:
            # If recent attempts have failed, fall back to eggs
            if self.wall_attempts - self.successful_wall_attempts > 4:
                self.last_mode = StrategyMode.EGG_MAXIMIZE
            else:
                self.last_mode = StrategyMode.WALL_BUILDING

        # Mid-late game prefer egg maximization unless wall nearly complete
        elif move_count >= 40 and separator_progress < 0.6:
            self.last_mode = StrategyMode.EGG_MAXIMIZE

        else:
            # Hybrid default: let search decide, but keep occasional wall attempts
            self.last_mode = StrategyMode.HYBRID

        return self.last_mode

    def record_wall_attempt(self, success: bool):
        """Record a wall attempt outcome; used for hysteresis and fallback decisions."""
        self.wall_attempts += 1
        if success:
            self.successful_wall_attempts += 1
