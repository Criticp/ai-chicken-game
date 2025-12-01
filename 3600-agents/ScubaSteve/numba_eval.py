"""
JIT-compiled evaluation function for maximum performance.

Reads weights from weights_config.py and evaluates board positions.
All functions are Numba-compatible for compilation to machine code.
"""

import numpy as np
from numba import njit

# Import weights as module-level constants (Numba can access these)
from .weights_config import (
    # Material weights
    EGG_CORNER_BONUS,
    EGG_REGULAR_BONUS,

    # Mobility
    WIN_SCORE,
    BLOCKED_PENALTY_MULTIPLIER,

    # Board geometry
    SEPARATOR_LINES,
    CORNERS,
)

from .numba_types import (
    EMPTY, ME, ENEMY, MY_EGG, ENEMY_EGG, MY_TURD, ENEMY_TURD,
    is_valid_position,
)


# ═══════════════════════════════════════════════════════════════
# EVALUATION WEIGHTS AS NUMBA-COMPATIBLE CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Material evaluation
W_MY_EGG = np.float32(100.0)  # Points per egg I have
W_ENEMY_EGG = np.float32(-120.0)  # Penalty per enemy egg

# Positional evaluation
W_CORNER_CONTROL = np.float32(50.0)  # Bonus for controlling corners
W_CENTER_CONTROL = np.float32(10.0)  # Bonus for being near center
W_SEPARATOR_CONTROL = np.float32(15.0)  # Bonus for eggs on separators

# Mobility evaluation
W_VALID_MOVE = np.float32(5.0)  # Points per available move
W_NO_MOVES = np.float32(-50000.0)  # Immediate loss if blocked

# Turd evaluation
W_MY_TURD_VALUE = np.float32(-10.0)  # Slight penalty for using turds early
W_ENEMY_TURD_THREAT = np.float32(-20.0)  # Penalty for enemy turds near me

# Distance evaluation
W_DISTANCE_TO_ENEMY = np.float32(-2.0)  # Penalty for being close to enemy


# ═══════════════════════════════════════════════════════════════
# JIT-COMPILED HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@njit
def count_pieces(board_arr: np.ndarray, piece_type: np.int8) -> np.int32:
    """Count number of pieces of given type on board"""
    count = 0
    for r in range(8):
        for c in range(8):
            if board_arr[r, c] == piece_type:
                count += 1
    return count


@njit
def find_piece_position(board_arr: np.ndarray, piece_type: np.int8) -> tuple:
    """Find position of a piece (assumes only one exists)"""
    for r in range(8):
        for c in range(8):
            if board_arr[r, c] == piece_type:
                return (r, c)
    return (-1, -1)


@njit
def manhattan_distance(r1: np.int8, c1: np.int8, r2: np.int8, c2: np.int8) -> np.int32:
    """Calculate Manhattan distance between two positions"""
    return abs(r1 - r2) + abs(c1 - c2)


@njit
def count_valid_moves_simple(board_arr: np.ndarray,
                            my_row: np.int8,
                            my_col: np.int8,
                            enemy_row: np.int8,
                            enemy_col: np.int8) -> np.int32:
    """
    Simple move counter for evaluation (doesn't consider parity/turds).
    Just counts how many directions we can move.
    """
    move_count = 0

    # Pre-calculate dangerous squares (adjacent to enemy turds)
    danger = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for c in range(8):
            if board_arr[r, c] == ENEMY_TURD:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if is_valid_position(nr, nc):
                            danger[nr, nc] = 1

    # Check all 4 directions
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in deltas:
        new_row = my_row + dr
        new_col = my_col + dc

        # Check bounds
        if not is_valid_position(new_row, new_col):
            continue

        # Check if square is blocked
        target = board_arr[new_row, new_col]
        if target == ENEMY or target == ENEMY_EGG or target == ENEMY_TURD:
            continue

        # Check danger zone
        if danger[new_row, new_col] == 1:
            continue

        move_count += 1

    return move_count


@njit
def is_corner(row: np.int8, col: np.int8) -> bool:
    """Check if position is a corner"""
    return (row == 0 or row == 7) and (col == 0 or col == 7)


@njit
def is_separator(row: np.int8, col: np.int8) -> bool:
    """Check if position is on a separator line (row/col 2 or 5)"""
    return row == 2 or row == 5 or col == 2 or col == 5


@njit
def distance_to_center(row: np.int8, col: np.int8) -> np.float32:
    """Calculate distance to center of board (3.5, 3.5)"""
    center_r = 3.5
    center_c = 3.5
    return abs(row - center_r) + abs(col - center_c)


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════

@njit
def evaluate_jit(board_arr: np.ndarray,
                my_row: np.int8,
                my_col: np.int8,
                enemy_row: np.int8,
                enemy_col: np.int8,
                my_turds_left: np.int8,
                enemy_turds_left: np.int8,
                depth: np.int32) -> np.float32:
    """
    JIT-compiled evaluation function.

    Called at depth 0 (leaf nodes) to score board position.

    Args:
        board_arr: 8x8 board state
        my_row, my_col: My chicken position
        enemy_row, enemy_col: Enemy chicken position
        my_turds_left: My remaining turds
        enemy_turds_left: Enemy remaining turds
        depth: Current depth (for debugging)

    Returns:
        Score from my perspective (positive = good for me)
    """
    score = np.float32(0.0)

    # ═══════════════════════════════════════════════════════════════
    # 1. MOBILITY CHECK (most critical)
    # ═══════════════════════════════════════════════════════════════
    my_moves = count_valid_moves_simple(board_arr, my_row, my_col, enemy_row, enemy_col)
    enemy_moves = count_valid_moves_simple(board_arr, enemy_row, enemy_col, my_row, my_col)

    if my_moves == 0:
        return W_NO_MOVES  # Immediate loss

    if enemy_moves == 0:
        score += W_NO_MOVES * -1.0  # Enemy blocked = huge bonus

    # Mobility differential
    score += my_moves * W_VALID_MOVE
    score -= enemy_moves * W_VALID_MOVE * 0.5  # Enemy mobility is bad (but less important)

    # ═══════════════════════════════════════════════════════════════
    # 2. MATERIAL COUNT (eggs)
    # ═══════════════════════════════════════════════════════════════
    my_eggs = count_pieces(board_arr, MY_EGG)
    enemy_eggs = count_pieces(board_arr, ENEMY_EGG)

    score += my_eggs * W_MY_EGG
    score += enemy_eggs * W_ENEMY_EGG  # Negative weight

    # ═══════════════════════════════════════════════════════════════
    # 3. POSITIONAL BONUSES
    # ═══════════════════════════════════════════════════════════════

    # Corner control (count eggs in corners)
    corner_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for r, c in corner_positions:
        if board_arr[r, c] == MY_EGG:
            score += W_CORNER_CONTROL
        elif board_arr[r, c] == ENEMY_EGG:
            score -= W_CORNER_CONTROL

    # Separator control (eggs on strategic lines)
    for r in range(8):
        for c in range(8):
            if is_separator(r, c):
                if board_arr[r, c] == MY_EGG:
                    score += W_SEPARATOR_CONTROL
                elif board_arr[r, c] == ENEMY_EGG:
                    score -= W_SEPARATOR_CONTROL

    # Center control (being near center is slightly good)
    my_center_dist = distance_to_center(my_row, my_col)
    if my_center_dist <= 2.0:
        score += W_CENTER_CONTROL

    # ═══════════════════════════════════════════════════════════════
    # 4. DISTANCE EVALUATION
    # ═══════════════════════════════════════════════════════════════

    # Distance to enemy (too close = risky)
    dist = manhattan_distance(my_row, my_col, enemy_row, enemy_col)
    if dist <= 2:
        score += W_DISTANCE_TO_ENEMY * (3 - dist)  # Penalty increases with proximity

    # ═══════════════════════════════════════════════════════════════
    # 5. TURD EVALUATION
    # ═══════════════════════════════════════════════════════════════

    # Penalty for using turds (we want to save them strategically)
    turds_used = 5 - my_turds_left
    score += turds_used * W_MY_TURD_VALUE

    # Enemy turds near me are threatening
    for r in range(8):
        for c in range(8):
            if board_arr[r, c] == ENEMY_TURD:
                turd_dist = manhattan_distance(my_row, my_col, r, c)
                if turd_dist <= 3:
                    score += W_ENEMY_TURD_THREAT

    return score


# ═══════════════════════════════════════════════════════════════
# WARMUP FUNCTION
# ═══════════════════════════════════════════════════════════════

def warmup_eval_jit():
    """Warm up evaluation JIT compilation"""
    print("[Numba Eval] Warming up evaluation JIT...")

    dummy_board = np.zeros((8, 8), dtype=np.int8)
    dummy_board[0, 0] = ME
    dummy_board[7, 7] = ENEMY
    dummy_board[0, 7] = MY_EGG
    dummy_board[7, 0] = ENEMY_EGG

    _ = evaluate_jit(dummy_board, 0, 0, 7, 7, 5, 5, 0)

    print("[Numba Eval] Evaluation JIT ready!")

