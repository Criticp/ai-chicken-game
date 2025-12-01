"""
Iterative Deepening wrapper for Numba search engine.

Handles root move unrolling and time management.
"""

import numpy as np
import time
from typing import Tuple, Callable

from game.enums import Direction, MoveType
from game import board as game_board

from .numba_types import (
    EMPTY, ME, ENEMY, MY_EGG, ENEMY_EGG, MY_TURD, ENEMY_TURD,
    PLAIN, EGG, TURD,
    UP, DOWN, LEFT, RIGHT,
    board_to_array,
    extract_board_metadata,
    generate_valid_moves,
    get_new_position,
    int_to_direction,
    int_to_movetype,
)

from .numba_search import negamax_jit, apply_move, undo_move
from .numba_tt import TRANSPOSITION_TABLE, clear_transposition_table, get_tt_stats

from .weights_config import (
    MAX_SEARCH_DEPTH,
    MAX_TIME_PER_MOVE,
    TIME_FRACTION_OF_REMAINING,
    TIME_MULTIPLIER_PER_MOVE,
    ABSOLUTE_BAN_TURD_TURNS,
)


class NumbaSearchEngine:
    """
    High-performance search engine using JIT-compiled Numba.

    Implements iterative deepening with root move unrolling.
    """

    def __init__(self):
        """Initialize search engine"""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.max_depth_reached = 0

    def search(self,
              board: "game_board.Board",
              time_left: Callable) -> Tuple[Direction, MoveType]:
        """
        Perform iterative deepening search to find best move.

        This is the ROOT of the search tree - it unrolls the first layer
        to track which move produced the best score.

        Args:
            board: Current board state
            time_left: Function that returns remaining time in seconds

        Returns:
            (Direction, MoveType) tuple representing best move
        """
        start_time = time.time()

        # Calculate time budget
        moves_remaining = max(board.turns_left_player, 1)
        time_budget = min(
            MAX_TIME_PER_MOVE,
            time_left() * TIME_FRACTION_OF_REMAINING,
            time_left() / moves_remaining * TIME_MULTIPLIER_PER_MOVE
        )

        print(f"[NumbaSearch] Time budget: {time_budget:.3f}s, Remaining: {time_left():.1f}s")

        # ═══════════════════════════════════════════════════════════════
        # CONVERT BOARD TO NUMPY ARRAY
        # ═══════════════════════════════════════════════════════════════
        board_arr = board_to_array(board)
        my_pos, enemy_pos, my_turds, enemy_turds, turn_count, can_lay_even = extract_board_metadata(board)

        my_row, my_col = my_pos[1], my_pos[0]  # Convert (x,y) to (row,col)
        enemy_row, enemy_col = enemy_pos[1], enemy_pos[0]

        # ═══════════════════════════════════════════════════════════════
        # GENERATE ROOT MOVES
        # ═══════════════════════════════════════════════════════════════
        moves = generate_valid_moves(
            board_arr,
            np.int8(my_row), np.int8(my_col),
            np.int8(enemy_row), np.int8(enemy_col),
            np.int8(my_turds),
            can_lay_even,
            True
        )

        # Filter turds in early game
        if turn_count <= ABSOLUTE_BAN_TURD_TURNS:
            filtered_moves = []
            for i in range(moves.shape[0]):
                if moves[i, 1] != TURD:
                    filtered_moves.append(moves[i])

            if len(filtered_moves) > 0:
                moves = np.array(filtered_moves, dtype=np.int32)

        if moves.shape[0] == 0:
            # No valid moves - return default
            print("[NumbaSearch] WARNING: No valid moves!")
            return (Direction.UP, MoveType.PLAIN)

        if moves.shape[0] == 1:
            # Only one move - return it immediately
            direction = int_to_direction(int(moves[0, 0]))
            move_type = int_to_movetype(int(moves[0, 1]))
            print(f"[NumbaSearch] Only one move: {direction.name} {move_type.name}")
            return (direction, move_type)

        # ═══════════════════════════════════════════════════════════════
        # ITERATIVE DEEPENING LOOP
        # ═══════════════════════════════════════════════════════════════
        best_move_idx = 0
        best_score = -np.inf

        self.nodes_searched = 0
        self.tt_hits = 0

        for depth in range(1, MAX_SEARCH_DEPTH + 1):
            # Check if we have time for this depth
            if time.time() - start_time >= time_budget * 0.9:
                print(f"[NumbaSearch] Time budget exhausted before depth {depth}")
                break

            self.max_depth_reached = depth
            depth_start = time.time()

            # ═══════════════════════════════════════════════════════════════
            # ROOT MOVE UNROLLING
            # ═══════════════════════════════════════════════════════════════
            depth_best_idx = 0
            depth_best_score = -np.inf

            alpha = np.float32(-100000.0)
            beta = np.float32(100000.0)

            moves_evaluated = 0

            for move_idx in range(moves.shape[0]):
                # Check time
                if time.time() - start_time >= time_budget:
                    print(f"[NumbaSearch] Time expired at depth {depth}, move {move_idx}/{moves.shape[0]}")
                    break

                direction = moves[move_idx, 0]
                move_type = moves[move_idx, 1]

                # Make a copy of the board for this simulation
                sim_board = board_arr.copy()

                # Apply move
                new_row, new_col = get_new_position(np.int8(my_row), np.int8(my_col), np.int8(direction))
                undo_info = apply_move(
                    sim_board,
                    np.int8(my_row), np.int8(my_col),
                    new_row, new_col,
                    np.int8(move_type),
                    True
                )

                # Update metadata for recursive search
                new_my_turds = my_turds - (1 if move_type == TURD else 0)

                # Search from opponent's perspective (negamax negation)
                score, nodes, hits = negamax_jit(
                    sim_board,
                    np.int32(depth - 1),
                    -beta, -alpha,
                    new_row, new_col,
                    np.int8(enemy_row), np.int8(enemy_col),
                    np.int8(new_my_turds),
                    np.int8(enemy_turds),
                    np.int32(turn_count + 1),
                    can_lay_even,
                    False,  # Enemy's turn next
                    TRANSPOSITION_TABLE,
                    start_time,
                    time_budget
                )

                # Negate score (negamax)
                score = -score

                # Undo move (restore board)
                undo_move(sim_board, np.int8(my_row), np.int8(my_col), new_row, new_col, undo_info[0])

                # Update statistics
                self.nodes_searched += nodes
                self.tt_hits += hits
                moves_evaluated += 1

                # Track best move at this depth
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_idx = move_idx

                # Update alpha
                if score > alpha:
                    alpha = score

            # If we completed this depth, update global best
            if moves_evaluated == moves.shape[0]:
                best_move_idx = depth_best_idx
                best_score = depth_best_score

                depth_time = time.time() - depth_start
                print(f"[NumbaSearch] Depth {depth}: score={best_score:.1f}, "
                      f"nodes={self.nodes_searched}, tt_hits={self.tt_hits}, "
                      f"time={depth_time:.3f}s")
            else:
                # Incomplete depth - use previous result
                print(f"[NumbaSearch] Depth {depth} incomplete, using depth {depth-1} result")
                break

        # ═══════════════════════════════════════════════════════════════
        # RETURN BEST MOVE
        # ═══════════════════════════════════════════════════════════════
        best_direction = int_to_direction(int(moves[best_move_idx, 0]))
        best_move_type = int_to_movetype(int(moves[best_move_idx, 1]))

        elapsed = time.time() - start_time
        tt_entries, tt_util = get_tt_stats()

        print(f"[NumbaSearch] FINAL: {best_direction.name} {best_move_type.name}, "
              f"score={best_score:.1f}, depth={self.max_depth_reached}, "
              f"nodes={self.nodes_searched}, tt_util={tt_util:.1f}%, "
              f"time={elapsed:.3f}s")

        return (best_direction, best_move_type)

