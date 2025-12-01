"""
ScubaSteve Numba Agent - High-Performance JIT-Compiled Search

Integrates all Numba components with proper warmup and initialization.
"""

from collections.abc import Callable
from typing import Tuple
import time

from game import board as game_board
from game.enums import Direction, MoveType

# Import Numba components
from .numba_types import warmup_jit
from .numba_eval import warmup_eval_jit
from .numba_tt import init_transposition_table, warmup_tt_jit, clear_transposition_table
from .numba_search import warmup_negamax_jit
from .numba_engine import NumbaSearchEngine

from .weights_config import *


class PlayerAgent:
    """
    ScubaSteve V6 Numba Edition

    High-performance agent using JIT-compiled search with:
    - Iterative Deepening (1 to MAX_SEARCH_DEPTH plies)
    - Negamax with Alpha-Beta Pruning
    - Transposition Table with Zobrist Hashing
    - Pure heuristic evaluation (no neural network)
    - All weights centralized in weights_config.py

    Expected performance:
    - Early game: Depth 6-8 in 2-3 seconds
    - Mid game: Depth 8-10 in 2-3 seconds
    - Late game: Depth 10-15 in 2-3 seconds
    - 10-100x faster than Python implementation
    """

    def __init__(self, board: "game_board.Board", time_left: Callable):
        """
        Initialize agent and warm up JIT compilation.

        This takes 1-2 seconds on first run to compile all functions.
        """
        print("=" * 70)
        print("ScubaSteve V6 Numba Edition - Initializing")
        print("=" * 70)

        init_start = time.time()

        # Initialize transposition table
        init_transposition_table(size=1_000_000)

        # Warm up all JIT functions (compiles to machine code)
        print("\n[Init] Warming up JIT compilation (this takes ~1-2 seconds)...")
        warmup_start = time.time()

        warmup_jit()
        warmup_eval_jit()
        warmup_tt_jit()
        warmup_negamax_jit()

        warmup_time = time.time() - warmup_start
        print(f"[Init] JIT warmup complete in {warmup_time:.2f}s")

        # Initialize search engine
        self.search_engine = NumbaSearchEngine()

        # Agent metadata
        self.name = "ScubaSteve_Numba"
        self.move_count = 0

        init_time = time.time() - init_start
        print(f"\n[Init] Agent ready in {init_time:.2f}s")
        print(f"[Init] Configuration: MAX_DEPTH={MAX_SEARCH_DEPTH}, "
              f"TIME_LIMIT={MAX_TIME_PER_MOVE}s")
        print("=" * 70)

    def play(self,
            board: "game_board.Board",
            sensor_data: dict,
            time_left: Callable) -> Tuple[Direction, MoveType]:
        """
        Choose and execute a move using Numba-accelerated search.

        Args:
            board: Current game board state
            sensor_data: Sensor information (unused)
            time_left: Function returning remaining time in seconds

        Returns:
            (Direction, MoveType) tuple
        """
        self.move_count += 1

        print(f"\n{'='*70}")
        print(f"TURN {self.move_count} - Time remaining: {time_left():.1f}s")
        print(f"{'='*70}")

        move_start = time.time()

        # Perform search
        direction, move_type = self.search_engine.search(board, time_left)

        move_time = time.time() - move_start
        print(f"\n[Agent] Move selected in {move_time:.3f}s: {direction.name} {move_type.name}")
        print(f"{'='*70}\n")

        return (direction, move_type)

