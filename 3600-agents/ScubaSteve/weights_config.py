"""
Centralized Weights Configuration for ScubaSteve Agent

All heuristic weights and search parameters in one place for easy tuning.
Modify values here to test different strategies.
"""

# ═══════════════════════════════════════════════════════════════
# SEARCH PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Maximum depth for iterative deepening search
MAX_SEARCH_DEPTH = 8  # Increased from 6 to 8 for deeper search

# Time management
MAX_TIME_PER_MOVE = 5.0  # Maximum seconds per move
TIME_FRACTION_OF_REMAINING = 0.3  # Use max 30% of remaining time
TIME_MULTIPLIER_PER_MOVE = 2.0  # Or 2x average time per move


# ═══════════════════════════════════════════════════════════════
# MOVE ORDERING WEIGHTS (for Alpha-Beta optimization)
# ═══════════════════════════════════════════════════════════════

# Egg move priorities (negative = higher priority in sorting)
CORNER_EGG_PRIORITY = -10000.0  # Highest priority (4 eggs!)
REGULAR_EGG_PRIORITY = -1000.0  # High priority
SEPARATOR_EGG_BONUS = -100.0  # Extra bonus for eggs on separator lines

# Plain move priority
PLAIN_MOVE_PRIORITY = 0.0  # Neutral

# Turd move priorities (positive = lower priority)
TURD_ABSOLUTE_BAN_PRIORITY = 10000.0  # Turns 0-3 (absolute ban)
TURD_EARLY_GAME_PRIORITY = 1000.0  # Turns 4-9 (very low)
TURD_SEPARATOR_PRIORITY = 500.0  # Turns 10+ on separator
TURD_NON_SEPARATOR_PRIORITY = 800.0  # Turns 10+ non-separator


# ═══════════════════════════════════════════════════════════════
# MOVE SCORING WEIGHTS (in search tree)
# ═══════════════════════════════════════════════════════════════

# Egg bonuses
EGG_CORNER_BONUS = 400.0  # Bonus for laying egg in corner
EGG_REGULAR_BONUS = 250.0  # Bonus for laying regular egg

# Plain move penalty
PLAIN_MOVE_PENALTY = 150.0  # Penalty for plain moves

# Turd scoring by game phase
TURD_ABSOLUTE_BAN_PENALTY = 10000.0  # Turns 0-3 (absolute ban)
TURD_EARLY_HEAVY_PENALTY = 1000.0  # Turns 4-9 (heavy penalty)
TURD_MID_SAVE_LAST_PENALTY = 500.0  # Turns 10-29 with ≤1 turd left
TURD_MID_GENERAL_PENALTY = 200.0  # Turns 10-29 general use
TURD_NON_SEPARATOR_PENALTY = 300.0  # Non-separator placement

# Turd bonuses (late game and strategic use)
TURD_EARLY_SEPARATOR_BONUS = 50.0  # Turns 4-9 on separator with ≥4 turds
TURD_MID_SEPARATOR_BONUS = 150.0  # Turns 10-29 on separator with ≥2 turds
TURD_LATE_SEPARATOR_BONUS = 300.0  # Turns 30+ on separator
TURD_LATE_NON_SEPARATOR_BONUS = 50.0  # Turns 30+ non-separator


# ═══════════════════════════════════════════════════════════════
# GAME PHASE THRESHOLDS
# ═══════════════════════════════════════════════════════════════

ABSOLUTE_BAN_TURD_TURNS = 3  # No turds until turn 4
EARLY_GAME_THRESHOLD = 10  # Turns 0-9 = early game
MID_GAME_THRESHOLD = 30  # Turns 10-29 = mid game, 30+ = late game

# Turd conservation thresholds
EARLY_GAME_MIN_TURDS = 4  # Need ≥4 turds to use in early game
MID_GAME_MIN_TURDS = 2  # Need ≥2 turds to use separators in mid game


# ═══════════════════════════════════════════════════════════════
# TERMINAL STATE SCORES
# ═══════════════════════════════════════════════════════════════

WIN_SCORE = 10000  # Score for winning position
BLOCKED_PENALTY_MULTIPLIER = 5.0  # Multiply by egg_diff_weight (5 egg penalty)


# ═══════════════════════════════════════════════════════════════
# SAFETY PROTOCOL
# ═══════════════════════════════════════════════════════════════

LAVA_FLOOR_ENABLED = True  # Enable safety filtering
SAFETY_THRESHOLD = 0.05  # 5% max trap risk tolerance


# ═══════════════════════════════════════════════════════════════
# BOARD GEOMETRY
# ═══════════════════════════════════════════════════════════════

BOARD_SIZE = 8  # 8x8 board
SEPARATOR_LINES = [2, 5]  # x=2, x=5, y=2, y=5 are separator lines
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]  # Corner positions

