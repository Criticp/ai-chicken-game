"""
Scuba Steve V7 - FACELIFT EDITION
Streamlined 50-50 Neural-Heuristic Hybrid

Core Philosophy:
- EGG MAXIMIZATION: Place eggs on every safe, unvisited cell
- MOVEMENT EFFICIENCY: 95%+ unique positions, minimal backtracking
- STRATEGIC TURDS: Block opponent access at choke points (15+ turns)
- HYBRID EVALUATION: 50% neural network + 50% heuristic
- CLEAR DEBUG OUTPUT: Every move explained with reasoning
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
from game.board import manhattan_distance

from .trapdoor_tracker import TrapdoorTracker
from .search_engine import SearchEngine
from .evaluator import HybridEvaluator
from .turd_advisor import TurdAdvisor
from .debug_reporter import DebugReporter


class PlayerAgent:
    """
    Scuba Steve V7: Streamlined Hybrid AI

    Components:
    1. TrapdoorTracker - Bayesian inference for trap detection
    2. HybridEvaluator - 50-50 neural/heuristic blend
    3. SearchEngine - Iterative deepening negamax
    4. TurdAdvisor - Strategic turd placement

    Removed complexity from V6:
    - No exploration frenzy mode
    - No corner trap detection
    - No invasion mode overrides
    - No separator wall planning
    - Simplified to core egg maximization
    """

    def __init__(self, board: "game_board.Board", time_left: Callable):
        """Initialize streamlined agent components"""

        # Component 1: Trapdoor Tracker (Bayesian Belief Engine)
        self.tracker = TrapdoorTracker(map_size=8)

        # Component 2: Hybrid Evaluator (50-50 Neural + Heuristic)
        model_path = os.path.join(os.path.dirname(__file__), 'chicken_eval_model_numpy.json')

        # ENABLE NEURAL NETWORK for 50-50 hybrid
        use_neural = True  # Changed from False

        if use_neural and os.path.exists(model_path):
            self.evaluator = HybridEvaluator(self.tracker, model_path)
            print("[ScubaSteve V7] Neural network ENABLED (50-50 hybrid mode)")
        else:
            self.evaluator = HybridEvaluator(self.tracker, None)
            print("[ScubaSteve V7] Heuristic-only mode (neural model not found)")

        # Component 3: Search Engine
        self.search_engine = SearchEngine(
            evaluator=self.evaluator,
            max_time_per_move=5.0
        )

        # Component 4: Turd Advisor
        try:
            self.turd_advisor = TurdAdvisor(self.evaluator.neural_eval)
        except Exception as e:
            print(f"[TurdAdvisor] Failed to initialize: {e}")
            self.turd_advisor = None

        # Opening Book
        self.opening_book: Dict[str, Tuple[Direction, MoveType]] = {}
        self._load_opening_book()

        # State tracking
        self.move_count = 0
        self.pos_history = []  # Last 12 positions for anti-looping
        self.move_history: deque = deque(maxlen=8)

        # Initialization flag
        self.prev_location = None
        self.spawn_location = None
        self._initialized = False

        # Performance metrics & Debugging
        self.eggs_per_turn = []
        self.reporter = DebugReporter()

        print("[ScubaSteve V7] ✓ All systems initialized")
        print(f"  - Neural Network: {'ACTIVE' if self.evaluator.neural_eval.model_loaded else 'INACTIVE'}")
        print(f"  - Opening Book: {len(self.opening_book)} positions")
        print(f"  - Strategy: Egg Maximization + Strategic Turd Blocking")

    def _load_opening_book(self):
        """Load opening book from grandmaster games"""
        try:
            book_path = os.path.join(os.path.dirname(__file__), 'opening_book.json')
            if os.path.exists(book_path):
                with open(book_path, 'r') as f:
                    raw_book = json.load(f)

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
        Main decision function - called each turn

        Returns:
            (Direction, MoveType) tuple
        """
        current_loc = board.chicken_player.get_location()
        current_eggs = board.chicken_player.eggs_laid
        moves_remaining = 40 - self.move_count

        # ================================================================
        # STEP 1: INITIALIZATION (First Turn)
        # ================================================================
        if not self._initialized:
            self.prev_location = current_loc
            self.spawn_location = board.chicken_player.get_spawn()
            self.move_history.append(current_loc)
            self._initialized = True
            print(f"[V7 Init] Spawn: {self.spawn_location}, Starting at: {current_loc}")

        # ================================================================
        # STEP 2: TRAPDOOR DETECTION & BELIEF UPDATE
        # ================================================================

        # Death detection (teleport = trapdoor)
        if self.prev_location is not None and self.prev_location != current_loc:
            px, py = self.prev_location
            neighbors = [(px-1, py), (px+1, py), (px, py-1), (px, py+1)]

            if current_loc not in neighbors and current_loc == self.spawn_location:
                # We died! Mark previous location as death trap
                self.tracker.mark_death_location(self.prev_location)
                self.reporter.report_death(self.prev_location)
            else:
                # Normal move - mark previous as safe
                self.tracker.mark_safe_square(self.prev_location)

        # Mark current location as safe (we're alive)
        self.tracker.mark_safe_square(current_loc)

        # Update beliefs from sensors
        self.tracker.update_from_sensors(current_loc, sensor_data)

        # Update found trapdoors
        for trapdoor in board.found_trapdoors:
            if trapdoor not in self.tracker.found_trapdoors:
                x, y = trapdoor
                is_even = (x + y) % 2 == 0
                self.tracker.mark_found_trapdoor(trapdoor, is_even)

        # ================================================================
        # STEP 3: OPENING BOOK (First 10 Moves)
        # ================================================================
        if self.move_count < 10 and self.opening_book:
            board_hash = self._hash_board(board)
            if board_hash in self.opening_book:
                book_move = self.opening_book[board_hash]

                # Verify it's valid and safe
                if board.is_valid_move(book_move[0], book_move[1], enemy=False):
                    dest = loc_after_direction(current_loc, book_move[0])
                    risk = self.tracker.get_trapdoor_risk(dest)

                    if risk < 0.5:  # Safety check
                        self._update_state(current_loc, dest, book_move)
                        self.reporter.report_opening_book(self.move_count, book_move)
                        return book_move

        # ================================================================
        # STEP 4: EGG MAXIMIZATION - Fast Egg Forcing
        # ================================================================
        # CRITICAL: Always prefer eggs when possible (no expensive search needed)

        forced_egg_move = None
        if board.can_lay_egg():
            valid_moves = board.get_valid_moves(enemy=False)
            egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]

            if egg_moves:
                # Fast heuristic: pick best egg move
                best_egg_move = None
                best_score = -float('inf')

                for move in egg_moves:
                    dest = loc_after_direction(current_loc, move[0])
                    score = 200  # Base egg value

                    # Safety check
                    risk = self.tracker.get_trapdoor_risk(dest)
                    if risk > 0.5:
                        continue  # Skip unsafe
                    score -= risk * 400

                    # HUGE bonus for unvisited cells
                    if dest not in self.pos_history:
                        score += 300

                    # Penalty for already-egged cells
                    if dest in board.eggs_player:
                        score -= 1000

                    # Bonus for distance from spawn (explore outward)
                    if self.spawn_location:
                        spawn_dist = manhattan_distance(dest, self.spawn_location)
                        score += spawn_dist * 5

                    # Avoid recent positions
                    if dest in self.pos_history[-6:]:
                        score -= 100

                    if score > best_score:
                        best_score = score
                        best_egg_move = move

                if best_egg_move and best_score > -500:
                    forced_egg_move = best_egg_move

        # If we have a forced egg move, use it (skip expensive search)
        if forced_egg_move:
            dest = loc_after_direction(current_loc, forced_egg_move[0])
            self._update_state(current_loc, dest, forced_egg_move)

            # Debug output
            reason = f"Forced egg (score: {best_score:.1f}, safety: {(1-risk)*100:.1f}%)"
            if self.move_count % 5 == 0 or self.move_count < 5:
                self.reporter.report_move(
                    board, current_loc, dest, forced_egg_move, time_left(),
                    heuristic_score=best_score, reasoning=reason, pos_history=self.pos_history
                )
            else:
                self.reporter.report_quick_move(
                    self.move_count, current_loc, dest, forced_egg_move[1], current_eggs + 1, reason
                )

            return forced_egg_move

        # ================================================================
        # STEP 5: SEARCH FOR BEST MOVE (Hybrid Eval)
        # ================================================================
        # Pass position history for anti-looping
        best_move, eval_details = self.search_engine.search(
            board=board,
            time_left=time_left,
            move_history=list(self.move_history),
            pos_history=self.pos_history
        )

        # ================================================================
        # STEP 6: TURD FILTERING (Strategic Use Only)
        # ================================================================
        if best_move[1] == MoveType.TURD:
            # Conservative turd policy: only after turn 15
            if self.move_count < 15:
                reason = f"Too early (turn {self.move_count} < 15)"
                self.reporter.report_turd_blocked(self.move_count, reason)
                # Replace with egg or plain
                if board.can_lay_egg():
                    best_move = (best_move[0], MoveType.EGG)
                else:
                    best_move = (best_move[0], MoveType.PLAIN)

            # Use turd advisor for strategic approval
            elif self.turd_advisor:
                try:
                    current_score = self.turd_advisor.score_cell(board, current_loc)

                    # Only use turd if score is good (50+)
                    if current_score.score < 50:
                        reason = f"Low value (score: {current_score.score:.1f})"
                        self.reporter.report_turd_blocked(self.move_count, reason)
                        # Replace with egg
                        if board.can_lay_egg():
                            best_move = (best_move[0], MoveType.EGG)
                        else:
                            best_move = (best_move[0], MoveType.PLAIN)
                    else:
                        self.reporter.report_turd_approved(self.move_count, current_score.score)
                except Exception as e:
                    print(f"[V7 Turd] Error in advisor: {e}")

        # ================================================================
        # STEP 7: UPDATE STATE & RETURN
        # ================================================================
        dest = loc_after_direction(current_loc, best_move[0])
        self._update_state(current_loc, dest, best_move)

        # Debug output every 5 turns
        if self.move_count % 5 == 0:
            self.reporter.report_move(
                board, current_loc, dest, best_move, time_left(),
                neural_score=eval_details.get('neural_score'),
                heuristic_score=eval_details.get('heuristic_score'),
                reasoning=f"Search (depth {eval_details.get('depth')})",
                pos_history=self.pos_history
            )
        else:
            self.reporter.report_quick_move(
                self.move_count, current_loc, dest, best_move[1], board.chicken_player.eggs_laid
            )

        return best_move

    def _update_state(self, current_loc, dest_loc, move):
        """Update internal state after move"""
        self.pos_history.append(current_loc)
        if len(self.pos_history) > 12:
            self.pos_history = self.pos_history[-12:]

        self.move_history.append(dest_loc)
        self.prev_location = current_loc
        self.move_count += 1

        # Track metrics
        if move[1] == MoveType.EGG:
            self.eggs_per_turn.append(self.move_count)

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

