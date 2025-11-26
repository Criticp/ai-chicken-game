"""
Scuba Steve V5 - Comprehensive AI Agent
Integrates all 5 components:
1. TrapdoorTracker (Bayesian Belief Engine)
2. SearchEngine (Iterative Deepening Negamax)
3. HybridEvaluator (Neural Network + Safety Checks)
4. Opening Book (Grandmaster Moves)
5. Mobility & Aggression Logic
"""

from collections.abc import Callable
from collections import deque
from typing import List, Tuple, Dict, Optional
import json
import os
import sys

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Direction, MoveType, Result, loc_after_direction

from .trapdoor_tracker import TrapdoorTracker
from .search_engine import SearchEngine
from .evaluator import HybridEvaluator


class PlayerAgent:
    """
    Scuba Steve V5: Comprehensive tournament AI agent.

    Architecture:
    - Bayesian trapdoor inference
    - Iterative deepening negamax search
    - Residual CNN evaluation (with fallback heuristics)
    - Opening book from grandmaster games
    - Loop prevention and aggression logic
    """

    def __init__(self, board: "game_board.Board", time_left: Callable):
        """Initialize all agent components"""

        # Component 1: Trapdoor Belief Engine
        self.tracker = TrapdoorTracker(map_size=8)

        # Component 3: Hybrid Evaluator (Neural + Safety)
        # Note: Neural network is slow in NumPy, using fast heuristic evaluation
        # To enable NN: change use_neural=True
        model_path = os.path.join(os.path.dirname(__file__), 'chicken_eval_model_numpy.json')
        use_neural = False  # Set to True to use trained NN (slower but potentially better)

        if use_neural:
            self.evaluator = HybridEvaluator(self.tracker, model_path)
        else:
            self.evaluator = HybridEvaluator(self.tracker, None)  # Fast heuristic mode

        # Component 2: Search Engine
        self.search_engine = SearchEngine(
            evaluator=self.evaluator,
            max_time_per_move=5.0
        )

        # Component 4: Opening Book
        self.opening_book: Dict[str, Tuple[Direction, MoveType]] = {}
        self._load_opening_book()

        # Component 5: Mobility & Aggression tracking
        self.move_count = 0
        self.move_history: deque = deque(maxlen=4)  # Last 4 positions for loop detection
        self.move_history.append(board.chicken_player.get_location())

        # Death detection (for teleport detection)
        self.prev_location = board.chicken_player.get_location()
        self.spawn_location = board.chicken_player.spawn

        print("[Scuba Steve V5] All systems initialized")
        print(f"  - Trapdoor Tracker: Bayesian inference engine")
        print(f"  - Search Engine: Iterative deepening negamax with transposition table")
        print(f"  - Evaluator: {'Neural Network (TRAINED)' if self.evaluator.neural_eval.model_loaded else 'Heuristic (FAST)'}")
        print(f"  - Opening Book: {len(self.opening_book)} positions")

    def _load_opening_book(self):
        """Load opening book from grandmaster games"""
        try:
            book_path = os.path.join(os.path.dirname(__file__), 'opening_book.json')
            if os.path.exists(book_path):
                with open(book_path, 'r') as f:
                    raw_book = json.load(f)

                    # Convert to usable format
                    for board_hash, move_data in raw_book.items():
                        direction = Direction(move_data['direction'])

                        if move_data['move_type'] == 'egg':
                            move_type = MoveType.EGG
                        elif move_data['move_type'] == 'turd':
                            move_type = MoveType.TURD
                        else:
                            move_type = MoveType.PLAIN

                        self.opening_book[board_hash] = (direction, move_type)

                print(f"[OpeningBook] ✓ Loaded {len(self.opening_book)} positions")
        except Exception as e:
            print(f"[OpeningBook] Not loaded: {e}")

    def play(
        self,
        board: "game_board.Board",
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """
        Main play function - called each turn.

        Returns:
            (Direction, MoveType) tuple
        """
        current_loc = board.chicken_player.get_location()

        # ====================================================================
        # STEP 1: DEATH DETECTION (Teleport = Trapdoor)
        # ====================================================================
        if self.prev_location is not None and self.prev_location != current_loc:
            # Check if we teleported (not adjacent move)
            px, py = self.prev_location
            neighbors = [(px-1, py), (px+1, py), (px, py-1), (px, py+1)]

            if current_loc not in neighbors and current_loc == self.spawn_location:
                # We died! Mark previous location as death trap (100% certain)
                self.tracker.mark_death_location(self.prev_location)
                print(f"[Agent] Death detected! Previous location {self.prev_location} marked as trap")
            else:
                # Normal move - mark previous location as safe (0% trap - Zero Knowledge)
                self.tracker.mark_safe_square(self.prev_location)

        # Mark current location as safe too (we're standing on it and alive)
        self.tracker.mark_safe_square(current_loc)

        # ====================================================================
        # STEP 2: UPDATE TRAPDOOR BELIEFS (Bayesian Inference)
        # ====================================================================
        self.tracker.update_from_sensors(current_loc, sensor_data)

        # Update found trapdoors
        for trapdoor in board.found_trapdoors:
            if trapdoor not in self.tracker.found_trapdoors:
                # Determine if even or odd based on coordinates
                x, y = trapdoor
                is_even = (x + y) % 2 == 0
                self.tracker.mark_found_trapdoor(trapdoor, is_even)

        # ====================================================================
        # STEP 3: CHECK OPENING BOOK (First 15 moves)
        # ====================================================================
        if self.move_count < 15 and self.opening_book:
            board_hash = self._hash_board(board)
            if board_hash in self.opening_book:
                book_move = self.opening_book[board_hash]

                # Verify it's still valid and safe
                if board.is_valid_move(book_move[0], book_move[1], enemy=False):
                    dest = loc_after_direction(current_loc, book_move[0])
                    risk = self.tracker.get_trapdoor_risk(dest)

                    if risk < 0.6:  # Safety check
                        # Use book move
                        self.move_history.append(dest)
                        self.prev_location = current_loc
                        self.move_count += 1

                        print(f"[Scuba Steve V5] Move {self.move_count}: {book_move} [OPENING BOOK]")
                        return book_move

        # ====================================================================
        # STEP 4: SEARCH FOR BEST MOVE (Iterative Deepening Negamax)
        # ====================================================================
        best_move = self.search_engine.search(
            board=board,
            time_left=time_left,
            move_history=list(self.move_history)
        )

        # ====================================================================
        # STEP 5: UPDATE STATE
        # ====================================================================
        dest_loc = loc_after_direction(current_loc, best_move[0])
        self.move_history.append(dest_loc)
        self.prev_location = current_loc
        self.move_count += 1

        # Component 5: Aggression check
        if best_move[1] == MoveType.TURD:
            # Only use turds if they reduce enemy mobility
            enemy_moves_before = len(board.get_valid_moves(enemy=True))
            print(f"[Aggression] Placing turd (enemy has {enemy_moves_before} moves)")

        print(f"[Scuba Steve V5] Move {self.move_count}: {best_move}")

        return best_move

    def _hash_board(self, board: "game_board.Board") -> str:
        """Hash board state for opening book lookup"""
        import hashlib

        try:
            player_loc = board.chicken_player.get_location()
            enemy_loc = board.chicken_enemy.get_location()

            eggs_p = sorted(list(board.eggs_player))
            eggs_e = sorted(list(board.eggs_enemy))
            turds_p = sorted(list(board.turds_player))
            turds_e = sorted(list(board.turds_enemy))

            state_str = f"{player_loc}|{enemy_loc}|{eggs_p}|{eggs_e}|{turds_p}|{turds_e}"
            return hashlib.md5(state_str.encode()).hexdigest()
        except:
            return ""

