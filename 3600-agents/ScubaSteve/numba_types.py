"""
Numba-optimized board representation and utilities for JIT compilation.

Converts Python Board objects to NumPy arrays for 10-100x faster search.
"""

import numpy as np
from numba import njit
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Direction, MoveType

# ═══════════════════════════════════════════════════════════════
# INTEGER CONSTANTS FOR NUMBA (Board cell types)
# ═══════════════════════════════════════════════════════════════
EMPTY = np.int8(0)
ME = np.int8(1)
ENEMY = np.int8(2)
MY_EGG = np.int8(3)
ENEMY_EGG = np.int8(4)
MY_TURD = np.int8(5)
ENEMY_TURD = np.int8(6)

# Move type integers
PLAIN = np.int8(0)
EGG = np.int8(1)
TURD = np.int8(2)

# Direction integers (matching game.enums.Direction)
UP = np.int8(0)
DOWN = np.int8(1)
LEFT = np.int8(2)
RIGHT = np.int8(3)

# Transposition table flags
TT_EXACT = np.int8(0)
TT_LOWERBOUND = np.int8(1)
TT_UPPERBOUND = np.int8(2)


# ═══════════════════════════════════════════════════════════════
# ZOBRIST HASH TABLE (Global, pre-computed)
# ═══════════════════════════════════════════════════════════════
# Initialize with fixed seed for deterministic hashing
_rng = np.random.default_rng(42)
ZOBRIST_TABLE = _rng.integers(0, 2**63 - 1, size=(8, 8, 7), dtype=np.int64)


# ═══════════════════════════════════════════════════════════════
# BOARD CONVERSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def board_to_array(board: "game_board.Board") -> np.ndarray:
    """
    Convert Board object to 8x8 NumPy array of int8.

    Cell values:
    0 = EMPTY
    1 = ME (player chicken)
    2 = ENEMY (opponent chicken)
    3 = MY_EGG
    4 = ENEMY_EGG
    5 = MY_TURD
    6 = ENEMY_TURD

    Returns:
        np.ndarray[8, 8, dtype=int8]
    """
    arr = np.zeros((8, 8), dtype=np.int8)

    # Place chickens
    px, py = board.chicken_player.get_location()
    arr[py, px] = ME

    ex, ey = board.chicken_enemy.get_location()
    arr[ey, ex] = ENEMY

    # Place eggs
    for (x, y) in board.eggs_player:
        arr[y, x] = MY_EGG

    for (x, y) in board.eggs_enemy:
        arr[y, x] = ENEMY_EGG

    # Place turds
    for (x, y) in board.turds_player:
        arr[y, x] = MY_TURD

    for (x, y) in board.turds_enemy:
        arr[y, x] = ENEMY_TURD

    return arr


def extract_board_metadata(board: "game_board.Board") -> tuple:
    """
    Extract metadata needed for search but not in array.

    Returns:
        (my_pos, enemy_pos, my_turds_left, enemy_turds_left, turn_count, is_my_turn)
    """
    my_pos = board.chicken_player.get_location()
    enemy_pos = board.chicken_enemy.get_location()
    my_turds = board.chicken_player.get_turds_left()
    enemy_turds = board.chicken_enemy.get_turds_left()
    turn_count = board.turn_count
    can_lay_even = board.chicken_player.can_lay_egg_on_even()

    return (my_pos, enemy_pos, my_turds, enemy_turds, turn_count, can_lay_even)


# ═══════════════════════════════════════════════════════════════
# JIT-COMPILED HASH FUNCTION
# ═══════════════════════════════════════════════════════════════

@njit
def compute_hash_fast(board_arr: np.ndarray) -> np.int64:
    """
    Compute Zobrist hash of board state.

    JIT-compiled for maximum speed.

    Args:
        board_arr: 8x8 array of cell types

    Returns:
        64-bit hash value
    """
    hash_val = np.int64(0)

    for row in range(8):
        for col in range(8):
            piece = board_arr[row, col]
            if piece != EMPTY:
                hash_val ^= ZOBRIST_TABLE[row, col, piece]

    return hash_val


@njit
def update_hash_incremental(old_hash: np.int64,
                           old_pos: tuple,
                           new_pos: tuple,
                           old_piece: np.int8,
                           new_piece: np.int8) -> np.int64:
    """
    Update hash incrementally after a move.

    XOR out old piece, XOR in new piece.

    Args:
        old_hash: Previous hash value
        old_pos: (row, col) of piece before move
        new_pos: (row, col) of piece after move
        old_piece: Piece type at old position (will be removed or replaced)
        new_piece: Piece type at new position (egg/turd deposited)

    Returns:
        Updated hash value
    """
    hash_val = old_hash

    # XOR out old position (chicken was here, now it's gone or has egg/turd)
    old_r, old_c = old_pos
    if old_piece != EMPTY:
        hash_val ^= ZOBRIST_TABLE[old_r, old_c, old_piece]

    # XOR in new position (chicken moved here)
    new_r, new_c = new_pos
    if new_piece != EMPTY:
        hash_val ^= ZOBRIST_TABLE[new_r, new_c, new_piece]

    return hash_val


# ═══════════════════════════════════════════════════════════════
# MOVE CONVERSION UTILITIES
# ═══════════════════════════════════════════════════════════════

def direction_to_int(direction: Direction) -> np.int8:
    """Convert Direction enum to integer"""
    return np.int8(direction.value)


def int_to_direction(val: int) -> Direction:
    """Convert integer to Direction enum"""
    return Direction(val)


def movetype_to_int(move_type: MoveType) -> np.int8:
    """Convert MoveType enum to integer"""
    return np.int8(move_type.value)


def int_to_movetype(val: int) -> MoveType:
    """Convert integer to MoveType enum"""
    return MoveType(val)


@njit
def get_new_position(row: np.int8, col: np.int8, direction: np.int8) -> tuple:
    """
    Get new position after moving in direction.

    Directions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

    Returns:
        (new_row, new_col)
    """
    if direction == UP:
        return (row - 1, col)
    elif direction == DOWN:
        return (row + 1, col)
    elif direction == LEFT:
        return (row, col - 1)
    else:  # RIGHT
        return (row, col + 1)


@njit
def is_valid_position(row: np.int8, col: np.int8) -> bool:
    """Check if position is within board bounds"""
    return 0 <= row < 8 and 0 <= col < 8


# ═══════════════════════════════════════════════════════════════
# MOVE GENERATION (JIT-COMPILED)
# ═══════════════════════════════════════════════════════════════

@njit
def generate_valid_moves(board_arr: np.ndarray,
                        my_row: np.int8,
                        my_col: np.int8,
                        enemy_row: np.int8,
                        enemy_col: np.int8,
                        my_turds_left: np.int8,
                        can_lay_even: bool,
                        is_my_turn: bool) -> np.ndarray:
    """
    Generate all valid moves for current player.

    Returns array of moves: Nx3 array where each row is [direction, move_type, priority]
    Priority is used for move ordering (lower = better)

    Args:
        board_arr: 8x8 board state
        my_row, my_col: Current player position
        enemy_row, enemy_col: Opponent position
        my_turds_left: Number of turds available
        can_lay_even: True if player lays eggs on even parity squares
        is_my_turn: True if generating for ME, False for ENEMY

    Returns:
        Nx3 array of [direction, move_type, priority]
    """
    moves = []

    # Determine which pieces block movement
    if is_my_turn:
        my_piece = ME
        enemy_piece = ENEMY
        enemy_egg = ENEMY_EGG
        enemy_turd = ENEMY_TURD
        my_turd = MY_TURD
    else:
        my_piece = ENEMY
        enemy_piece = ME
        enemy_egg = MY_EGG
        enemy_turd = MY_TURD
        my_turd = ENEMY_TURD

    # Pre-calculate dangerous squares (adjacent to enemy turds)
    danger = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for c in range(8):
            if board_arr[r, c] == enemy_turd:
                # Mark all adjacent squares as dangerous
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if is_valid_position(nr, nc):
                            danger[nr, nc] = 1

    # Try all 4 directions
    for direction in range(4):
        new_row, new_col = get_new_position(my_row, my_col, direction)

        # Check bounds
        if not is_valid_position(new_row, new_col):
            continue

        # Check if square is blocked
        target = board_arr[new_row, new_col]
        if target == enemy_piece or target == enemy_egg or target == enemy_turd:
            continue

        # Check danger zone (adjacent to enemy turd)
        if danger[new_row, new_col] == 1:
            continue

        # 1. PLAIN MOVE (always valid if we got here)
        moves.append([direction, PLAIN, 0])  # Priority 0 (neutral)

        # 2. EGG MOVE (check parity and occupancy)
        parity = (new_row + new_col) % 2
        can_lay = (can_lay_even and parity == 0) or (not can_lay_even and parity == 1)

        current_cell = board_arr[my_row, my_col]
        occupied = current_cell in [MY_EGG, ENEMY_EGG, MY_TURD, ENEMY_TURD]

        if can_lay and not occupied:
            # Priority based on corner bonus
            is_corner = (my_row == 0 or my_row == 7) and (my_col == 0 or my_col == 7)
            priority = -10000 if is_corner else -1000  # Negative = high priority
            moves.append([direction, EGG, priority])

        # 3. TURD MOVE (check turd availability and adjacency)
        if my_turds_left > 0 and not occupied:
            # Check if adjacent to enemy
            adjacent_to_enemy = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    check_r, check_c = my_row + dr, my_col + dc
                    if check_r == enemy_row and check_c == enemy_col:
                        adjacent_to_enemy = True
                        break
                if adjacent_to_enemy:
                    break

            if not adjacent_to_enemy:
                moves.append([direction, TURD, 500])  # Low priority (positive = bad)

    # Convert to NumPy array
    if len(moves) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    return np.array(moves, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════
# WARMUP FUNCTION
# ═══════════════════════════════════════════════════════════════

def warmup_jit():
    """
    Warm up JIT compilation by calling functions with dummy data.

    This ensures first real move doesn't experience compilation lag.
    """
    print("[Numba] Warming up JIT compilation...")

    # Create dummy board
    dummy_board = np.zeros((8, 8), dtype=np.int8)
    dummy_board[0, 0] = ME
    dummy_board[7, 7] = ENEMY

    # Warm up hash
    _ = compute_hash_fast(dummy_board)

    # Warm up move generation
    _ = generate_valid_moves(dummy_board, 0, 0, 7, 7, 5, True, True)

    print("[Numba] JIT compilation complete!")

