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
from typing import List, Tuple, Dict
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
from .turd_advisor import TurdAdvisor

# ScubaSteve V6 components (local modules)
from .separator_planner import SeparatorPlanner
from .exploration_tracker import ExplorationTracker
from .path_planner import PathPlanner
from .strategic_mode_manager import StrategyManager, StrategyMode


class PlayerAgent:
    """
    Scuba Steve V6: Strategic Enhancement Architecture

    V5 Core:
    - Bayesian trapdoor inference
    - Iterative deepening negamax search
    - Residual CNN evaluation (with fallback heuristics)
    - Opening book from grandmaster games
    - Loop prevention and aggression logic

    V6 Enhancements:
    - Separator wall strategy (rows/columns 2 & 5)
    - Exploration tracking (visited cells, egg placement history)
    - Intelligent pathfinding (opportunistic egg laying)
    - Strategic mode management (wall building vs egg maximization)
    - Parity-based turd placement
    """

    def __init__(self, board: "game_board.Board", time_left: Callable):
        """Initialize all agent components (V5 + V6)"""

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

        # ═══════════════════════════════════════════════════════════════
        # V6 NEW COMPONENTS
        # ═══════════════════════════════════════════════════════════════

        # Component 7: Separator Planner (Strategic wall building)
        try:
            self.separator_planner = SeparatorPlanner(trapdoor_tracker=self.tracker)
        except Exception as e:
            print(f"[SeparatorPlanner] Failed to initialize: {e}")
            self.separator_planner = None

        # Component 8: Exploration Tracker (Memory system)
        try:
            self.exploration_tracker = ExplorationTracker()
        except Exception as e:
            print(f"[ExplorationTracker] Failed to initialize: {e}")
            self.exploration_tracker = None

        # Component 9: Path Planner (Intelligent navigation)
        try:
            self.path_planner = PathPlanner(
                exploration_tracker=self.exploration_tracker,
                trapdoor_tracker=self.tracker
            )
        except Exception as e:
            print(f"[PathPlanner] Failed to initialize: {e}")
            self.path_planner = None

        # Component 10: Strategy Manager (Mode coordination)
        try:
            self.strategy_manager = StrategyManager(separator_planner=self.separator_planner)
        except Exception as e:
            print(f"[StrategyManager] Failed to initialize: {e}")
            self.strategy_manager = None

        # Component 2: Search Engine
        self.search_engine = SearchEngine(
            evaluator=self.evaluator,
            max_time_per_move=5.0
        )

        # Component 6: Turd Advisor (Enhanced with V6 features)
        try:
            self.turd_advisor = TurdAdvisor(
                self.evaluator.neural_eval,
                separator_planner=self.separator_planner,
                exploration_tracker=self.exploration_tracker
            )
        except Exception as e:
            print(f"[TurdAdvisor] Failed to initialize: {e}")
            self.turd_advisor = None

        # Component 4: Opening Book
        self.opening_book: Dict[str, Tuple[Direction, MoveType]] = {}
        self._load_opening_book()

        # Component 5: Mobility & Aggression tracking
        self.move_count = 0

        # FINAL POLISH: The Breadcrumb Trail (Anti-Looping)
        # Keep last 8 positions to prevent oscillation
        self.pos_history = []  # Will store last 8 positions
        self.move_history: deque = deque(maxlen=4)  # Legacy for compatibility

        # NEW: Stuck detection counters
        self.stuck_counter = 0
        self.force_egg_mode_until = -1  # Turn number when forced egg mode expires

        # Death detection (for teleport detection)
        # NOTE: These will be initialized on first play() call when chicken has been positioned
        self.prev_location = None
        self.spawn_location = None
        self._initialized = False  # Flag to track first-turn initialization

        print("[Scuba Steve V6] All systems initialized")
        print(f"  - Trapdoor Tracker: Bayesian inference engine")
        print(f"  - Search Engine: Iterative deepening negamax with transposition table")
        print(f"  - Evaluator: {'Neural Network (TRAINED)' if self.evaluator.neural_eval.model_loaded else 'Heuristic (FAST)'}")
        print(f"  - Turd Advisor: Enhanced with parity + wall + exploration bonuses")
        print(f"  - Opening Book: {len(self.opening_book)} positions")
        print(f"  - Separator Planner: Strategic wall builder (rows/columns 2 & 5)")
        print(f"  - Exploration Tracker: Memory system for efficient pathfinding")
        print(f"  - Path Planner: Opportunistic egg placement during traversal")
        print(f"  - Strategy Manager: Dynamic mode switching (wall vs eggs)")

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
        # STEP 0: FIRST-TURN INITIALIZATION
        # ====================================================================
        if not self._initialized:
            self.prev_location = current_loc
            self.spawn_location = board.chicken_player.get_spawn()
            self.move_history.append(current_loc)
            self._initialized = True

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
        # STEP 2.5: UPDATE EXPLORATION TRACKER (V6 NEW)
        # ====================================================================
        if self.exploration_tracker:
            self.exploration_tracker.update(current_loc, board.turn_count)

            # Track egg placements
            if self.prev_location and self.prev_location in board.eggs_player:
                self.exploration_tracker.mark_egged(self.prev_location)

        # ====================================================================
        # STEP 2.75: STUCK DETECTION (V6 ANTI-LOOP) - ENHANCED
        # ====================================================================
        # Detect if we're oscillating between same positions
        if len(self.pos_history) >= 3:  # Changed from 4 to 3 - faster detection
            # Check if current position appeared in last 2 positions (was 3)
            if current_loc in self.pos_history[-2:]:  # More aggressive threshold
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

            # If stuck for 2+ consecutive detections (was 3), force egg mode
            if self.stuck_counter >= 2 and self.force_egg_mode_until < board.turn_count:
                self.force_egg_mode_until = board.turn_count + 10  # Force for 10 turns (was 5)
                print(f"[STUCK DETECTOR] Loop detected! Forcing EGG_MAXIMIZE mode for 10 turns")
                # Invalidate path cache to force replanning
                if self.path_planner:
                    self.path_planner.invalidate_cache()
                # Also try to find nearest fresh cell
                if self.exploration_tracker:
                    fresh_cell = self.exploration_tracker.find_nearest_unexplored(current_loc, board)
                    if fresh_cell:
                        print(f"[STUCK DETECTOR] Routing to fresh cell at {fresh_cell}")
                self.stuck_counter = 0

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

                    if risk < 0.6: # Safety check
                        # Use book move
                        self.move_history.append(dest)
                        self.prev_location = current_loc
                        self.move_count += 1

                        print(f"[Scuba Steve V6] Move {self.move_count}: {book_move} [OPENING BOOK]")
                        return book_move

        # ====================================================================
        # STEP 3.5: STRATEGIC MODE SELECTION (V6 NEW)
        # ====================================================================
        current_strategy = StrategyMode.HYBRID  # Default
        path_plan = None
        wall_target = None

        # Check if stuck detector forced egg mode
        if self.force_egg_mode_until >= board.turn_count:
            current_strategy = StrategyMode.EGG_MAXIMIZE
            print(f"[V6 Strategy] FORCED EGG MODE (stuck detector) - {self.force_egg_mode_until - board.turn_count} turns remaining")
        elif self.strategy_manager and self.separator_planner:
            # Get current separator progress
            separator_progress = self.separator_planner.get_progress()

            # Select strategy based on game state
            current_strategy = self.strategy_manager.select_strategy(
                board, self.move_count, separator_progress
            )

            # If in wall-building mode, get target
            if current_strategy == StrategyMode.WALL_BUILDING and self.separator_planner.is_active():
                wall_target = self.separator_planner.get_next_wall_target(board)

                # Plan route to wall target WITH FORCED EGG PLACEMENT
                if wall_target and self.path_planner:
                    path_plan = self.path_planner.plan_route(
                        current_loc, wall_target, board, self.exploration_tracker,
                        force_egg_on_route=True  # NEW: Force eggs while routing to wall
                    )

                    if path_plan:
                        print(f"[V6 Strategy] {current_strategy.value} - routing to wall {wall_target} (path length: {len(path_plan)}, EGGS ON ROUTE)")

        # ====================================================================
        # STEP 4: SEARCH FOR BEST MOVE (Iterative Deepening Negamax)
        # ====================================================================
        # FINAL POLISH: Pass position history for anti-looping
        best_move = self.search_engine.search(
            board=board,
            time_left=time_left,
            move_history=list(self.move_history),
            pos_history=self.pos_history  # NEW: Breadcrumb trail
        )

        # ====================================================================
        # STEP 4.5: GUARDIAN WRAPPER - Protect Against Max's Turd Traps
        # ====================================================================
        enemy_loc = board.chicken_enemy.get_location()

        # Check if our best_move is a "trap door" (allows enemy checkmate next turn)
        if not self._is_move_safe_from_turd_trap(board, best_move, current_loc, enemy_loc):
            print(f"[GUARDIAN] ⚠️  Best move {best_move} is a TRAP DOOR! Filtering for safety...")

            # Get all valid moves and filter out trap doors
            valid_moves = board.get_valid_moves(enemy=False)
            safe_moves = self._filter_moves_for_safety(board, valid_moves, current_loc, enemy_loc)

            # If we found safer alternatives, use them
            if safe_moves and best_move not in safe_moves:
                # Prefer egg-laying moves from safe options
                egg_moves = [m for m in safe_moves if m[1] == MoveType.EGG]
                if egg_moves and board.can_lay_egg():
                    best_move = egg_moves[0]
                    print(f"[GUARDIAN] ✓ Using safe egg move: {best_move}")
                else:
                    best_move = safe_moves[0]
                    print(f"[GUARDIAN] ✓ Using safe move: {best_move}")

        # ====================================================================
        # STEP 4.75: TURD ADVISOR - STRATEGIC DECISION MAKER
        # ====================================================================
        if best_move[1] == MoveType.TURD and self.turd_advisor is not None:
            # Agent wants to place turd - use TurdAdvisor to evaluate if it's the BEST place
            try:
                # Get the advisor's recommendation for best turd placement
                recommendation = self.turd_advisor.recommend(board)

                if recommendation:
                    # Check if current location matches advisor's top recommendation
                    # or if advisor scores this location highly
                    current_score = self.turd_advisor.score_cell(board, current_loc)

                    # If current location is good enough (within 80% of best score), approve
                    score_threshold = recommendation.score * 0.8

                    if current_score.score >= score_threshold:
                        print(f"[TURD ADVISOR] ✓ APPROVED - Score: {current_score.score:.1f} (Threshold: {score_threshold:.1f})")
                        # Keep the turd move
                    else:
                        # Current location not ideal - suggest alternative
                        print(f"[TURD ADVISOR] ✗ REJECTED - Score: {current_score.score:.1f} < Threshold: {score_threshold:.1f}")
                        print(f"[TURD ADVISOR] Best location would be {recommendation.cell} (score: {recommendation.score:.1f})")

                        # Replace turd with alternative move (egg or plain in same direction)
                        if board.is_valid_move(best_move[0], MoveType.EGG, enemy=False):
                            best_move = (best_move[0], MoveType.EGG)
                            print(f"[TURD ADVISOR] → Using EGG instead")
                        elif board.is_valid_move(best_move[0], MoveType.PLAIN, enemy=False):
                            best_move = (best_move[0], MoveType.PLAIN)
                            print(f"[TURD ADVISOR] → Using PLAIN instead")
                        else:
                            # Get any egg move as fallback
                            all_moves = board.get_valid_moves(enemy=False)
                            egg_moves = [m for m in all_moves if m[1] == MoveType.EGG]
                            if egg_moves:
                                best_move = egg_moves[0]
                                print(f"[TURD ADVISOR] → Using alternative EGG move")
                            else:
                                # Keep turd as last resort
                                print(f"[TURD ADVISOR] → No alternatives, keeping turd")
                else:
                    # No recommendation available, trust search
                    print(f"[TURD ADVISOR] No recommendation available, trusting search")
            except Exception as e:
                print(f"[TURD ADVISOR] Error: {e}, trusting search")

        # ====================================================================
        # STEP 5: UPDATE STATE
        # ====================================================================
        dest_loc = loc_after_direction(current_loc, best_move[0])

        # FINAL POLISH: Update breadcrumb trail (keep last 8 positions)
        self.pos_history.append(current_loc)
        if len(self.pos_history) > 8:
            self.pos_history = self.pos_history[-8:]  # Keep only last 8

        # Legacy tracking (for compatibility)
        self.move_history.append(dest_loc)
        self.prev_location = current_loc
        self.move_count += 1

        # Component 5: Aggression check
        if best_move[1] == MoveType.TURD:
            # Only use turds if they reduce enemy mobility
            enemy_moves_before = len(board.get_valid_moves(enemy=False))
            print(f"[Aggression] Placing turd (enemy has {enemy_moves_before} moves)")

        # ====================================================================
        # STEP 5.5: UPDATE V6 COMPONENTS (NEW)
        # ====================================================================

        # Update separator planner
        if self.separator_planner:
            placed_turd = (best_move[1] == MoveType.TURD)
            turd_loc = current_loc if placed_turd else None
            self.separator_planner.update(board, placed_turd, turd_loc)

            # Record turd in exploration tracker as well
            if placed_turd and self.exploration_tracker and turd_loc:
                self.exploration_tracker.mark_turd(turd_loc)

        # Update strategy manager
        if self.strategy_manager and self.separator_planner:
            if best_move[1] == MoveType.TURD and wall_target:
                # Check if this was a successful wall placement
                success = (current_loc == wall_target)
                self.strategy_manager.record_wall_attempt(success)

        print(f"[Scuba Steve V6] Move {self.move_count}: {best_move}")

        # V6 status report
        if self.separator_planner and self.move_count % 5 == 0:
            print(f"[V6 Status] {self.separator_planner.get_status()}")
            if self.exploration_tracker:
                stats = self.exploration_tracker.get_exploration_stats()
                print(f"[V6 Exploration] Visited: {stats['visited_cells']}, Egged: {stats['egged_cells']}")

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

    # ========================================================================
    # GUARDIAN WRAPPER: Turd Trap Detection (vs Max)
    # ========================================================================

    def _is_move_safe_from_turd_trap(self, board: "game_board.Board",
                                      move: Tuple[Direction, MoveType],
                                      my_loc: Tuple[int, int],
                                      enemy_loc: Tuple[int, int]) -> bool:
        """
        GUARDIAN WRAPPER: Check if a move allows enemy to trap us with turd next turn.

        Simulates worst-case scenario:
        1. We move to next_loc
        2. Enemy moves adjacent and drops turd at their current position
        3. Check if we'd have 0 valid moves (checkmate)

        Args:
            board: Current board state
            move: Our proposed move (Direction, MoveType)
            my_loc: Our current location
            enemy_loc: Enemy current location

        Returns:
            True if move is safe, False if it's a trap door
        """
        from game.board import manhattan_distance

        # Calculate where we'd be after this move
        next_loc = loc_after_direction(my_loc, move[0])

        # Check all possible enemy moves
        enemy_directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

        for enemy_dir in enemy_directions:
            enemy_next_loc = loc_after_direction(enemy_loc, enemy_dir)

            # Skip invalid enemy moves
            if not board.is_valid_cell(enemy_next_loc):
                continue
            if enemy_next_loc == next_loc:  # Enemy can't move on top of us
                continue
            if enemy_next_loc in board.eggs_player:
                continue
            if enemy_next_loc in board.turds_player:
                continue

            # Simulate enemy dropping turd at their CURRENT location (where they WERE)
            # Game rule: turd is placed at departure square, not destination
            simulated_enemy_turd = enemy_loc

            # Check if turd placement would be valid for enemy
            if manhattan_distance(simulated_enemy_turd, next_loc) < 2:
                # Enemy can't place turd adjacent to where we'll be
                continue

            # Check if enemy has turds left
            if board.chicken_enemy.get_turds_left() <= 0:
                continue

            # Now check: how many valid moves would WE have from next_loc
            # after enemy places this turd?
            our_valid_moves_count = 0

            for our_dir in enemy_directions:
                our_future_loc = loc_after_direction(next_loc, our_dir)

                # Check if this move would be valid
                if not board.is_valid_cell(our_future_loc):
                    continue
                if our_future_loc == enemy_next_loc:  # Enemy will be there
                    continue
                if our_future_loc in board.eggs_enemy:
                    continue

                # CRITICAL: Check if adjacent to simulated enemy turd
                # We cannot move to squares adjacent to enemy turds
                if manhattan_distance(our_future_loc, simulated_enemy_turd) <= 1:
                    continue  # Blocked by turd zone

                # Also check existing enemy turds
                is_blocked_by_existing_turds = False
                for existing_turd in board.turds_enemy:
                    if manhattan_distance(our_future_loc, existing_turd) <= 1:
                        is_blocked_by_existing_turds = True
                        break

                if is_blocked_by_existing_turds:
                    continue

                # Also check our own turds (can't move onto them)
                if our_future_loc in board.turds_player:
                    continue

                # This move would be valid!
                our_valid_moves_count += 1

            # If enemy can force us to 0 moves, this is a TRAP DOOR
            if our_valid_moves_count == 0:
                return False  # NOT SAFE - enemy can checkmate us next turn

        # Checked all enemy responses, none lead to our checkmate
        return True  # SAFE

    def _filter_moves_for_safety(self, board: "game_board.Board",
                                 valid_moves: List[Tuple[Direction, MoveType]],
                                 my_loc: Tuple[int, int],
                                 enemy_loc: Tuple[int, int]) -> List[Tuple[Direction, MoveType]]:
        """
        GUARDIAN WRAPPER: Filter moves to remove trap doors.

        If all moves are dangerous, picks the one with most "breathing room"
        (highest number of future moves in worst case).

        Args:
            board: Current board state
            valid_moves: List of all valid moves
            my_loc: Our current location
            enemy_loc: Enemy current location

        Returns:
            List of safe moves, or least-dangerous move if all are traps
        """
        safe_moves = []
        move_safety_scores = {}  # Move -> breathing room score

        for move in valid_moves:
            if self._is_move_safe_from_turd_trap(board, move, my_loc, enemy_loc):
                safe_moves.append(move)
            else:
                # This move is a trap door - calculate "breathing room"
                # Count how many neighbors the destination has (more = safer)
                next_loc = loc_after_direction(my_loc, move[0])

                breathing_room = 0
                for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                    test_loc = loc_after_direction(next_loc, direction)
                    if board.is_valid_cell(test_loc):
                        if test_loc not in board.eggs_enemy:
                            if test_loc not in board.turds_enemy:
                                breathing_room += 1

                move_safety_scores[move] = breathing_room

        # If we have safe moves, return only those
        if safe_moves:
            return safe_moves

        # ALL moves are trap doors! Pick the one with most breathing room
        if move_safety_scores:
            best_bad_move = max(move_safety_scores.items(), key=lambda x: x[1])[0]
            print(f"[GUARDIAN] ⚠️  WARNING: All moves are trap doors! Picking least bad: {best_bad_move} (breathing room: {move_safety_scores[best_bad_move]})")
            return [best_bad_move]

        # Ultimate fallback: return original moves
        return valid_moves

    # ========================================================================
    # LIGHTWEIGHT TURD FILTER: Prevent Wasteful Turds (Trust Search for Strategy)
    # ========================================================================

    def _lightweight_turd_filter(self, board: "game_board.Board",
                                 current_loc: Tuple[int, int]) -> Tuple[bool, str]:
        """
        LIGHTWEIGHT TURD FILTER: Only reject obviously wasteful turds.

        Philosophy: Trust the search engine's turd decisions (it has good heuristics).
        Only block turds that are clearly wasteful:
        - Very early game (first 8 turns - need to explore)
        - Far from any action (>6 from enemy AND >4 from our territory)
        - Last turd wasted on low-value position

        DEFAULT: APPROVE (let the agent's search make strategic decisions)

        Args:
            board: Current board state
            current_loc: Position where turd would be placed

        Returns:
            (approved: bool, reason: str)
        """
        from game.board import manhattan_distance

        turn = board.turn_count
        turds_left = board.chicken_player.get_turds_left()
        enemy_pos = board.chicken_enemy.get_location()

        # RULE 1: No turds in first 8 turns (let agent explore and establish territory)
        if turn < 8:
            return False, f"Turn {turn} < 8 (explore first)"

        # RULE 2: Reserve last turd for critical moments only
        if turds_left <= 1:
            dist_to_enemy = manhattan_distance(current_loc, enemy_pos)
            if dist_to_enemy > 4:
                return False, f"Last turd - too far from enemy ({dist_to_enemy} squares)"

        # RULE 3: Reject turds that are wastefully far from action
        dist_to_enemy = manhattan_distance(current_loc, enemy_pos)

        # Check distance to our nearest egg (are we in/near our territory?)
        min_dist_to_our_eggs = float('inf')
        if board.eggs_player:
            for egg in board.eggs_player:
                dist = manhattan_distance(current_loc, egg)
                min_dist_to_our_eggs = min(min_dist_to_our_eggs, dist)

        # Wasteful if far from enemy AND far from our territory
        if dist_to_enemy > 6 and min_dist_to_our_eggs > 4:
            return False, f"Wasteful (enemy:{dist_to_enemy}, our territory:{min_dist_to_our_eggs})"

        # DEFAULT: APPROVE - trust the search engine
        return True, "✓ Approved"

    # ========================================================================
    # TURD STRATEGY MANAGER: Smart Turd Conservation Helper Methods
    # ========================================================================

    def _should_use_turd(self, board: "game_board.Board",
                         move: Tuple[Direction, MoveType],
                         current_loc: Tuple[int, int],
                         enemy_loc: Tuple[int, int]) -> bool:
        """
        TURD STRATEGY MANAGER: Decides if a turd move should be allowed.

        Philosophy: Turds are PRECIOUS (only 5 total)
        - Early game (turns 1-20): NO turds unless life-threatening
        - Mid game (turns 21-35): Only for high-value tactical blocks
        - Late game (turns 36+): Aggressive turd warfare

        High-value situations:
        1. KILL SHOT: Enemy would have 0 moves after our turd
        2. LIFE SAVER: We're about to be trapped, turd gives escape
        3. MAJOR MOBILITY REDUCTION: Turd removes 4+ enemy moves

        Args:
            board: Current board state
            move: Proposed turd move
            current_loc: Our location
            enemy_loc: Enemy location

        Returns:
            True if turd should be used, False if should be saved
        """
        from game.board import manhattan_distance

        # Get game state
        turn_number = self.move_count
        turds_remaining = board.chicken_player.get_turds_left()

        # ====================================================================
        # RULE 1: Protect Last 2 Turds (Kill Shots Only)
        # ====================================================================
        if turds_remaining <= 2:
            if self._is_turd_kill_shot(board, current_loc, enemy_loc):
                print(f"[TURD MANAGER] Last {turds_remaining} turds - KILL SHOT approved!")
                return True
            print(f"[TURD MANAGER] Protecting last {turds_remaining} turds for kill shots")
            return False

        # ====================================================================
        # RULE 2: Early Game (Turns 1-20) - STRICT Conservation
        # ====================================================================
        if turn_number <= 20:
            # Exception 1: Kill shot
            if self._is_turd_kill_shot(board, current_loc, enemy_loc):
                print(f"[TURD MANAGER] Early game KILL SHOT approved!")
                return True

            # Exception 2: Life-threatening (we have ≤2 moves)
            our_mobility = len(board.get_valid_moves(enemy=False))
            if our_mobility <= 2:
                print(f"[TURD MANAGER] Early game LIFE SAVER approved (mobility={our_mobility})")
                return True

            print(f"[TURD MANAGER] Early game - conserving turds")
            return False

        # ====================================================================
        # RULE 3: Mid Game (Turns 21-35) - Tactical Use
        # ====================================================================
        if turn_number <= 35:
            # Allow turds if they significantly reduce enemy mobility
            enemy_moves_before = len(board.get_valid_moves(enemy=False))

            # Exception 1: Kill shot
            if self._is_turd_kill_shot(board, current_loc, enemy_loc):
                print(f"[TURD MANAGER] Tactical KILL SHOT approved!")
                return True

            # Exception 2: Major mobility reduction (cripple enemy movement)
            if enemy_moves_before > 4:
                print(f"[TURD MANAGER] Tactical turd - reduces enemy moves from {enemy_moves_before} to 0!")
                return True

            print(f"[TURD MANAGER] Mid game - conservative turd use")
            return False

        # ====================================================================
        # RULE 4: Late Game (Turns 36+) - Aggressive Turd Warfare
        # ====================================================================
        print(f"[TURD MANAGER] Late game - aggressive turd warfare")
        return True

    def _is_turd_kill_shot(self, board: "game_board.Board",
                           turd_loc: Tuple[int, int],
                           enemy_loc: Tuple[int, int]) -> bool:
        """
        TURD STRATEGY MANAGER: Check if placing a turd here would result in enemy having 0 moves.

        Args:
            board: Current board state
            turd_loc: Location where turd is proposed
            enemy_loc: Enemy current location

        Returns:
            True if this is a kill shot turd placement, False otherwise
        """
        from game.board import manhattan_distance

        # Simulate placing the turd
        original_turds_enemy = board.turds_enemy.copy()
        board.turds_enemy.append(turd_loc)

        # Check all possible enemy moves
        enemy_directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        for enemy_dir in enemy_directions:
            enemy_next_loc = loc_after_direction(enemy_loc, enemy_dir)

            # Skip invalid enemy moves
            if not board.is_valid_cell(enemy_next_loc):
                continue
            if enemy_next_loc == turd_loc:  # Enemy can't move on top of turd
                continue
            if enemy_next_loc in board.eggs_player:
                continue
            if enemy_next_loc in board.turds_player:
                continue

            # If enemy has ANY valid move, it's not a kill shot
            board.turds_enemy = original_turds_enemy  # Restore state
            return False

        # If we checked all directions and enemy has no moves, it's a kill shot
        board.turds_enemy = original_turds_enemy  # Restore state
        return True

