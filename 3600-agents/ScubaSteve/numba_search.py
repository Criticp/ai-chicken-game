"""
JIT-compiled Negamax search with Alpha-Beta pruning.

This is the core search engine - fully compiled to machine code for maximum speed.
"""

import numpy as np
from numba import njit
import time

from .numba_types import (
    EMPTY, ME, ENEMY, MY_EGG, ENEMY_EGG, MY_TURD, ENEMY_TURD,
    PLAIN, EGG, TURD,
    UP, DOWN, LEFT, RIGHT,
    TT_EXACT, TT_LOWERBOUND, TT_UPPERBOUND,
    compute_hash_fast,
    generate_valid_moves,
    get_new_position,
    is_valid_position,
)

from .numba_eval import evaluate_jit
from .numba_tt import tt_store, tt_lookup, TRANSPOSITION_TABLE

from .weights_config import (
    WIN_SCORE,
    ABSOLUTE_BAN_TURD_TURNS,
)


# ═══════════════════════════════════════════════════════════════
# BOARD SIMULATION (MAKE/UNDO MOVES)
# ═══════════════════════════════════════════════════════════════

@njit
def apply_move(board_arr: np.ndarray,
              old_row: np.int8,
              old_col: np.int8,
              new_row: np.int8,
              new_col: np.int8,
              move_type: np.int8,
              is_my_turn: bool) -> tuple:
    """
    Apply a move to the board (MODIFIES board_arr in place).

    Returns information needed to undo the move.

    Args:
        board_arr: 8x8 board state (will be modified)
        old_row, old_col: Current position
        new_row, new_col: Destination position
        move_type: PLAIN, EGG, or TURD
        is_my_turn: True if this is player's move

    Returns:
        (old_cell_value,) - info needed to undo
    """
    # Determine piece types
    if is_my_turn:
        moving_piece = ME
        egg_piece = MY_EGG
        turd_piece = MY_TURD
    else:
        moving_piece = ENEMY
        egg_piece = ENEMY_EGG
        turd_piece = ENEMY_TURD

    # Save old cell value for undo
    old_cell_value = board_arr[old_row, old_col]

    # Remove chicken from old position (may leave egg/turd)
    if move_type == EGG:
        board_arr[old_row, old_col] = egg_piece
    elif move_type == TURD:
        board_arr[old_row, old_col] = turd_piece
    else:  # PLAIN
        board_arr[old_row, old_col] = EMPTY

    # Place chicken at new position
    board_arr[new_row, new_col] = moving_piece

    return (old_cell_value,)


@njit
def undo_move(board_arr: np.ndarray,
             old_row: np.int8,
             old_col: np.int8,
             new_row: np.int8,
             new_col: np.int8,
             old_cell_value: np.int8):
    """
    Undo a move (restore board to previous state).

    Args:
        board_arr: 8x8 board state (will be modified)
        old_row, old_col: Original position before move
        new_row, new_col: Position moved to
        old_cell_value: Value that was at old position before move
    """
    # Restore old position
    board_arr[old_row, old_col] = old_cell_value

    # Clear new position
    board_arr[new_row, new_col] = EMPTY


# ═══════════════════════════════════════════════════════════════
# JIT-COMPILED NEGAMAX WITH ALPHA-BETA
# ═══════════════════════════════════════════════════════════════

@njit
def negamax_jit(board_arr: np.ndarray,
               depth: np.int32,
               alpha: np.float32,
               beta: np.float32,
               my_row: np.int8,
               my_col: np.int8,
               enemy_row: np.int8,
               enemy_col: np.int8,
               my_turds_left: np.int8,
               enemy_turds_left: np.int8,
               turn_count: np.int32,
               can_lay_even: bool,
               is_my_turn: bool,
               tt_array: np.ndarray,
               start_time: np.float64,
               time_limit: np.float64) -> tuple:
    """
    JIT-compiled Negamax search with Alpha-Beta pruning and TT.

    Returns:
        (score, nodes_searched, tt_hits)
    """
    nodes_searched = 1
    tt_hits = 0

    # ═══════════════════════════════════════════════════════════════
    # TIME CHECK
    # ═══════════════════════════════════════════════════════════════
    current_time = time.time()
    if current_time - start_time >= time_limit:
        # Time's up - return quick evaluation
        score = evaluate_jit(board_arr, my_row, my_col, enemy_row, enemy_col,
                            my_turds_left, enemy_turds_left, depth)
        if not is_my_turn:
            score = -score
        return (score, nodes_searched, tt_hits)

    # ═══════════════════════════════════════════════════════════════
    # TRANSPOSITION TABLE LOOKUP
    # ═══════════════════════════════════════════════════════════════
    board_hash = compute_hash_fast(board_arr)

    found, tt_score, new_alpha, new_beta = tt_lookup(tt_array, board_hash, depth, alpha, beta)

    if found:
        tt_hits += 1
        # Transposition table cutoff
        return (tt_score, nodes_searched, tt_hits)

    # Update alpha/beta from TT bounds
    alpha = new_alpha
    beta = new_beta

    # ═══════════════════════════════════════════════════════════════
    # BASE CASE: DEPTH 0 (Leaf node - evaluate position)
    # ═══════════════════════════════════════════════════════════════
    if depth == 0:
        score = evaluate_jit(board_arr, my_row, my_col, enemy_row, enemy_col,
                            my_turds_left, enemy_turds_left, depth)

        # Negamax: negate if enemy's turn
        if not is_my_turn:
            score = -score

        # Store in TT as exact value
        tt_store(tt_array, board_hash, depth, TT_EXACT, score)

        return (score, nodes_searched, tt_hits)

    # ═══════════════════════════════════════════════════════════════
    # GENERATE MOVES
    # ═══════════════════════════════════════════════════════════════
    if is_my_turn:
        moves = generate_valid_moves(board_arr, my_row, my_col, enemy_row, enemy_col,
                                     my_turds_left, can_lay_even, True)
    else:
        moves = generate_valid_moves(board_arr, enemy_row, enemy_col, my_row, my_col,
                                     enemy_turds_left, not can_lay_even, False)

    # Filter turds in early game (ABSOLUTE BAN)
    if turn_count <= ABSOLUTE_BAN_TURD_TURNS:
        filtered_moves = []
        for i in range(moves.shape[0]):
            if moves[i, 1] != TURD:  # Not a turd move
                filtered_moves.append(moves[i])

        if len(filtered_moves) > 0:
            moves = np.array(filtered_moves, dtype=np.int32)
        # else: keep all moves (emergency fallback)

    # ═══════════════════════════════════════════════════════════════
    # TERMINAL CASE: NO MOVES (Blocked)
    # ═══════════════════════════════════════════════════════════════
    if moves.shape[0] == 0:
        # Being blocked is very bad
        score = evaluate_jit(board_arr, my_row, my_col, enemy_row, enemy_col,
                            my_turds_left, enemy_turds_left, depth)
        score -= 500.0  # Blocked penalty

        if not is_my_turn:
            score = -score

        tt_store(tt_array, board_hash, depth, TT_EXACT, score)
        return (score, nodes_searched, tt_hits)

    # ═══════════════════════════════════════════════════════════════
    # MOVE ORDERING (Sort by priority - lower = better)
    # ═══════════════════════════════════════════════════════════════
    # Sort moves by priority column (column 2)
    sorted_indices = np.argsort(moves[:, 2])
    moves = moves[sorted_indices]

    # ═══════════════════════════════════════════════════════════════
    # RECURSIVE SEARCH
    # ═══════════════════════════════════════════════════════════════
    best_score = -np.inf
    original_alpha = alpha

    for i in range(moves.shape[0]):
        direction = moves[i, 0]
        move_type = moves[i, 1]

        # Get positions
        if is_my_turn:
            old_r, old_c = my_row, my_col
        else:
            old_r, old_c = enemy_row, enemy_col

        new_r, new_c = get_new_position(old_r, old_c, direction)

        # Apply move
        undo_info = apply_move(board_arr, old_r, old_c, new_r, new_c, move_type, is_my_turn)

        # Update positions and turd counts
        if is_my_turn:
            new_my_row, new_my_col = new_r, new_c
            new_enemy_row, new_enemy_col = enemy_row, enemy_col
            new_my_turds = my_turds_left - (1 if move_type == TURD else 0)
            new_enemy_turds = enemy_turds_left
        else:
            new_my_row, new_my_col = my_row, my_col
            new_enemy_row, new_enemy_col = new_r, new_c
            new_my_turds = my_turds_left
            new_enemy_turds = enemy_turds_left - (1 if move_type == TURD else 0)

        # Recurse (negate for negamax)
        child_score, child_nodes, child_hits = negamax_jit(
            board_arr, depth - 1, -beta, -alpha,
            new_my_row, new_my_col, new_enemy_row, new_enemy_col,
            new_my_turds, new_enemy_turds, turn_count + 1,
            can_lay_even, not is_my_turn,
            tt_array, start_time, time_limit
        )

        score = -child_score  # Negamax negation

        # Undo move
        undo_move(board_arr, old_r, old_c, new_r, new_c, undo_info[0])

        # Update statistics
        nodes_searched += child_nodes
        tt_hits += child_hits

        # Alpha-beta logic
        if score > best_score:
            best_score = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Beta cutoff
            break

    # ═══════════════════════════════════════════════════════════════
    # STORE IN TRANSPOSITION TABLE
    # ═══════════════════════════════════════════════════════════════
    if best_score <= original_alpha:
        # Failed low - upper bound
        tt_store(tt_array, board_hash, depth, TT_UPPERBOUND, best_score)
    elif best_score >= beta:
        # Failed high - lower bound
        tt_store(tt_array, board_hash, depth, TT_LOWERBOUND, best_score)
    else:
        # Exact score
        tt_store(tt_array, board_hash, depth, TT_EXACT, best_score)

    return (best_score, nodes_searched, tt_hits)


# ═══════════════════════════════════════════════════════════════
# WARMUP FUNCTION
# ═══════════════════════════════════════════════════════════════

def warmup_negamax_jit():
    """Warm up negamax JIT compilation"""
    print("[Negamax] Warming up negamax JIT...")

    dummy_board = np.zeros((8, 8), dtype=np.int8)
    dummy_board[0, 0] = ME
    dummy_board[7, 7] = ENEMY

    dummy_tt = np.zeros((1000, 4), dtype=np.int64)

    _ = negamax_jit(
        dummy_board, 2, -1000.0, 1000.0,
        0, 0, 7, 7,
        5, 5, 0, True, True,
        dummy_tt, time.time(), 1.0
    )

    print("[Negamax] Negamax JIT ready!")

