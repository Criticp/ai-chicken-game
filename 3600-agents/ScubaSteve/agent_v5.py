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
            max_time_per_move=5.0,
            trapdoor_tracker=self.tracker  # Pass tracker for tactical engine
        )

        # Component 4: Opening Book
        self.opening_book: Dict[str, Tuple[Direction, MoveType]] = {}
        self._load_opening_book()

        # Component 5: Mobility & Aggression tracking
        self.move_count = 0

        # FINAL POLISH: The Breadcrumb Trail (Anti-Looping)
        # Keep last 8 positions to prevent oscillation
        self.pos_history = []  # Will store last 8 positions
        self.move_history: deque = deque(maxlen=4)  # Legacy for compatibility

        # Death detection (for teleport detection)
        # NOTE: These will be initialized on first play() call when chicken has been positioned
        self.prev_location = None
        self.spawn_location = None
        self._initialized = False  # Flag to track first-turn initialization

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
            # Check A: Kill shot
            if self._is_turd_kill_shot(board, current_loc, enemy_loc):
                print(f"[TURD MANAGER] Mid game KILL SHOT approved!")
                return True

            # Check B: Major mobility reduction (removes 4+ enemy moves)
            enemy_mobility_before = len(board.get_valid_moves(enemy=True))
            enemy_mobility_after = self._simulate_enemy_mobility_after_turd(board, current_loc, enemy_loc)
            mobility_reduction = enemy_mobility_before - enemy_mobility_after

            if mobility_reduction >= 4:
                print(f"[TURD MANAGER] Mid game TACTICAL turd approved (reduces enemy by {mobility_reduction} moves)")
                return True

            # Check C: Defensive (we're low on mobility)
            our_mobility = len(board.get_valid_moves(enemy=False))
            if our_mobility <= 3:
                print(f"[TURD MANAGER] Mid game DEFENSIVE turd approved (our mobility={our_mobility})")
                return True

            print(f"[TURD MANAGER] Mid game - turd not high-value enough (reduction={mobility_reduction})")
            return False

        # ====================================================================
        # RULE 4: Late Game (Turns 36+) - Aggressive Warfare
        # ====================================================================

        # Priority A: Kill shots
        if self._is_turd_kill_shot(board, current_loc, enemy_loc):
            print(f"[TURD MANAGER] Late game KILL SHOT approved!")
            return True

        # Priority B: Significant mobility reduction (3+ moves)
        enemy_mobility_before = len(board.get_valid_moves(enemy=True))
        enemy_mobility_after = self._simulate_enemy_mobility_after_turd(board, current_loc, enemy_loc)
        mobility_reduction = enemy_mobility_before - enemy_mobility_after

        if mobility_reduction >= 3:
            print(f"[TURD MANAGER] Late game AGGRESSIVE turd approved (reduces enemy by {mobility_reduction})")
            return True

        print(f"[TURD MANAGER] Late game - turd not valuable enough (reduction={mobility_reduction})")
        return False

    def _is_turd_kill_shot(self, board: "game_board.Board",
                           my_loc: Tuple[int, int],
                           enemy_loc: Tuple[int, int]) -> bool:
        """
        Check if placing turd at current location would reduce enemy to 0 moves (checkmate).

        Returns:
            True if this turd would checkmate the enemy
        """
        from game.board import manhattan_distance

        # Verify turd placement is valid
        if manhattan_distance(my_loc, enemy_loc) < 2:
            return False

        # Simulate turd at our current location
        simulated_turds = board.turds_player.copy()
        simulated_turds.add(my_loc)

        # Count enemy moves after this turd
        enemy_valid_moves = 0

        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            enemy_next_loc = loc_after_direction(enemy_loc, direction)

            # Check if move is valid
            if not board.is_valid_cell(enemy_next_loc):
                continue
            if enemy_next_loc in board.eggs_player:
                continue

            # Check if blocked by our simulated turd
            is_blocked = False
            for check_dir in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                adj_loc = loc_after_direction(enemy_next_loc, check_dir)
                if adj_loc in simulated_turds:
                    is_blocked = True
                    break

            # Also check existing turds
            if not is_blocked:
                for existing_turd in board.turds_player:
                    if manhattan_distance(enemy_next_loc, existing_turd) <= 1:
                        is_blocked = True
                        break

            if not is_blocked:
                enemy_valid_moves += 1

        # Kill shot = enemy has 0 moves
        return enemy_valid_moves == 0

    def _simulate_enemy_mobility_after_turd(self, board: "game_board.Board",
                                            my_loc: Tuple[int, int],
                                            enemy_loc: Tuple[int, int]) -> int:
        """
        Simulate how many moves enemy would have after we place turd at my_loc.

        Returns:
            Number of valid enemy moves after turd placement
        """
        from game.board import manhattan_distance

        # Simulate turd at our location
        simulated_turds = board.turds_player.copy()
        simulated_turds.add(my_loc)

        # Count enemy moves
        enemy_moves = 0

        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            enemy_next_loc = loc_after_direction(enemy_loc, direction)

            if not board.is_valid_cell(enemy_next_loc):
                continue
            if enemy_next_loc in board.eggs_player:
                continue

            # Check if blocked by simulated turd
            is_blocked = False
            for turd in simulated_turds:
                if manhattan_distance(enemy_next_loc, turd) <= 1:
                    is_blocked = True
                    break

            if not is_blocked:
                enemy_moves += 1

        return enemy_moves
