"""
Component 2: The Search Engine (Iterative Deepening Negamax)

Implements:
- Negamax with Alpha-Beta Pruning
- Iterative Deepening (IDDFS)
- Transposition Table with Zobrist Hashing
- Heuristic Move Ordering (Eggs → Plains → Turds)
- Time Management
"""

from typing import List, Tuple, Dict, Callable, Optional
import time
import random
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Direction, MoveType, Result, loc_after_direction

# Import centralized weights configuration
from .weights_config import *


class TranspositionTable:
    """
    Cache for board positions to avoid re-evaluation.
    Uses Zobrist hashing for fast board state identification.
    """

    def __init__(self):
        self.table: Dict[int, Tuple[float, int, Tuple[Direction, MoveType]]] = {}
        self.zobrist_table = self._init_zobrist()

    def _init_zobrist(self) -> Dict:
        """Initialize Zobrist hash table with random 64-bit integers"""
        random.seed(42)  # Deterministic for consistency

        zobrist = {
            'player_pos': [[random.getrandbits(64) for _ in range(8)] for _ in range(8)],
            'enemy_pos': [[random.getrandbits(64) for _ in range(8)] for _ in range(8)],
            'egg_player': [[random.getrandbits(64) for _ in range(8)] for _ in range(8)],
            'egg_enemy': [[random.getrandbits(64) for _ in range(8)] for _ in range(8)],
            'turd_player': [[random.getrandbits(64) for _ in range(8)] for _ in range(8)],
            'turd_enemy': [[random.getrandbits(64) for _ in range(8)] for _ in range(8)],
        }
        return zobrist

    def compute_hash(self, board: "game_board.Board") -> int:
        """Compute Zobrist hash for board state"""
        hash_value = 0

        # Hash player position
        px, py = board.chicken_player.get_location()
        hash_value ^= self.zobrist_table['player_pos'][py][px]

        # Hash enemy position
        ex, ey = board.chicken_enemy.get_location()
        hash_value ^= self.zobrist_table['enemy_pos'][ey][ex]

        # Hash eggs
        for (x, y) in board.eggs_player:
            hash_value ^= self.zobrist_table['egg_player'][y][x]
        for (x, y) in board.eggs_enemy:
            hash_value ^= self.zobrist_table['egg_enemy'][y][x]

        # Hash turds
        for (x, y) in board.turds_player:
            hash_value ^= self.zobrist_table['turd_player'][y][x]
        for (x, y) in board.turds_enemy:
            hash_value ^= self.zobrist_table['turd_enemy'][y][x]

        return hash_value

    def store(self, hash_key: int, score: float, depth: int, best_move: Tuple[Direction, MoveType]):
        """Store position evaluation in transposition table"""
        self.table[hash_key] = (score, depth, best_move)

    def lookup(self, hash_key: int, depth: int) -> Optional[Tuple[float, Tuple[Direction, MoveType]]]:
        """
        Lookup position in transposition table.
        Returns (score, best_move) if found at sufficient depth, else None.
        """
        if hash_key in self.table:
            stored_score, stored_depth, stored_move = self.table[hash_key]
            if stored_depth >= depth:
                return (stored_score, stored_move)
        return None

    def clear(self):
        """Clear the transposition table"""
        self.table.clear()


class SearchEngine:
    """
    Iterative Deepening Negamax search with Alpha-Beta pruning.
    Includes time management and move ordering for optimal performance.
    """

    def __init__(self, evaluator, max_time_per_move: float = MAX_TIME_PER_MOVE):
        """
        Args:
            evaluator: Evaluation function that takes (board, depth) and returns score
            max_time_per_move: Maximum time to spend on a single move (seconds)
        """
        self.evaluator = evaluator
        self.max_time_per_move = max_time_per_move
        self.transposition_table = TranspositionTable()

        # Statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.max_depth_reached = 0


    def search(self, board: "game_board.Board", time_left: Callable,
               move_history: Optional[List] = None,
               pos_history: Optional[List] = None) -> Tuple[Direction, MoveType]:
        """
        Perform iterative deepening search to find best move.

        Args:
            board: Current board state
            time_left: Callable that returns remaining time in seconds
            move_history: Recent move history for loop prevention (legacy)
            pos_history: Position history for breadcrumb trail anti-looping (NEW)

        Returns:
            (Direction, MoveType) tuple representing best move
        """
        # Store pos_history for evaluator access
        self.pos_history = pos_history if pos_history is not None else []

        start_time = time.time()

        # Calculate time budget for this move
        moves_remaining = max(board.turns_left_player, 1)
        time_budget = min(
            self.max_time_per_move,
            time_left() * TIME_FRACTION_OF_REMAINING,
            time_left() / moves_remaining * TIME_MULTIPLIER_PER_MOVE
        )
        deadline = start_time + time_budget

        # Get valid moves
        valid_moves = board.get_valid_moves(enemy=False)

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL FIX: ABSOLUTE BAN on turds for turns 0-3
        # ═══════════════════════════════════════════════════════════════
        if board.turn_count <= ABSOLUTE_BAN_TURD_TURNS:
            valid_moves = [(d, t) for d, t in valid_moves if t != MoveType.TURD]
            # If somehow ONLY turd moves exist (shouldn't happen), allow them
            if not valid_moves:
                valid_moves = board.get_valid_moves(enemy=False)

        # ═══════════════════════════════════════════════════════════════
        # LAVA FLOOR PROTOCOL: Filter out unsafe moves (risk > 5%)
        # ═══════════════════════════════════════════════════════════════
        if LAVA_FLOOR_ENABLED:
            safe_moves = self._filter_safe_moves(board, valid_moves, enemy=False)

            # Exception: If NO safe moves exist, use risky moves
            # (Being blocked = -5 eggs, worse than trap = -4 eggs)
            if safe_moves:
                valid_moves = safe_moves
            # else: keep all valid_moves (emergency fallback)

        # Prune loops if history provided
        if move_history:
            valid_moves = self._prune_loops(board, valid_moves, move_history)

        if not valid_moves:
            # Fallback
            valid_moves = board.get_valid_moves(enemy=False)
            if not valid_moves:
                return (Direction.UP, MoveType.PLAIN)

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Iterative deepening
        best_move = valid_moves[0]
        best_score = -float('inf')

        self.nodes_searched = 0
        self.tt_hits = 0

        # Start from depth 1, go up to MAX_SEARCH_DEPTH (or until time runs out)
        for depth in range(1, MAX_SEARCH_DEPTH + 1):
            if time.time() >= deadline:
                break

            self.max_depth_reached = depth
            iteration_start = time.time()

            # Search at this depth
            alpha = -float('inf')
            beta = float('inf')
            current_best = None
            current_score = -float('inf')

            # Order moves for better alpha-beta cutoffs
            ordered_moves = self._order_moves(board, valid_moves)

            for move in ordered_moves:
                if time.time() >= deadline:
                    break

                # Make move
                sim_board = self._copy_board(board)
                if not self._apply_move(sim_board, move, enemy=False):
                    continue

                # Negamax: opponent's score from their perspective, negated
                score = -self._negamax(sim_board, depth - 1, -beta, -alpha,
                                      False, deadline)

                # Apply move scoring heuristics from config
                direction, move_type = move
                current_loc = board.chicken_player.get_location()

                if move_type == MoveType.EGG:
                    if self._is_corner(current_loc):
                        score += EGG_CORNER_BONUS
                    else:
                        score += EGG_REGULAR_BONUS

                elif move_type == MoveType.TURD:
                    x, y = current_loc
                    is_separator = (x in SEPARATOR_LINES or y in SEPARATOR_LINES)
                    turds_left = board.chicken_player.get_turds_left()
                    turn = board.turn_count

                    # Absolute ban on early turds
                    if turn <= ABSOLUTE_BAN_TURD_TURNS:
                        score -= TURD_ABSOLUTE_BAN_PENALTY

                    elif turn < EARLY_GAME_THRESHOLD:
                        # Early game: Very conservative
                        if is_separator and turds_left >= EARLY_GAME_MIN_TURDS:
                            score += TURD_EARLY_SEPARATOR_BONUS
                        else:
                            score -= TURD_EARLY_HEAVY_PENALTY

                    elif turn < MID_GAME_THRESHOLD:
                        # Mid game: Strategic separator use
                        if is_separator and turds_left >= MID_GAME_MIN_TURDS:
                            score += TURD_MID_SEPARATOR_BONUS
                        elif turds_left <= 1:
                            score -= TURD_MID_SAVE_LAST_PENALTY
                        else:
                            score -= TURD_MID_GENERAL_PENALTY

                    else:
                        # Late game: Active turd use
                        if is_separator:
                            score += TURD_LATE_SEPARATOR_BONUS
                        else:
                            score += TURD_LATE_NON_SEPARATOR_BONUS

                    # Non-separator penalty
                    if not is_separator:
                        score -= TURD_NON_SEPARATOR_PENALTY

                elif move_type == MoveType.PLAIN:
                    score -= PLAIN_MOVE_PENALTY

                if score > current_score:
                    current_score = score
                    current_best = move

                alpha = max(alpha, score)
                if alpha >= beta:
                    break  # Beta cutoff

            # If we completed this depth iteration, update best move
            if current_best is not None and time.time() < deadline:
                best_move = current_best
                best_score = current_score
            else:
                # Time ran out, use previous depth's result
                break

        elapsed = time.time() - start_time
        print(f"[SearchEngine] Depth={self.max_depth_reached} Nodes={self.nodes_searched} "
              f"TT_hits={self.tt_hits} Time={elapsed:.3f}s")

        return best_move

    def _negamax(self, board: "game_board.Board", depth: int, alpha: float, beta: float,
                 is_player_turn: bool, deadline: float) -> float:
        """
        Negamax with alpha-beta pruning.

        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_player_turn: True if it's player's turn
            deadline: Time to stop searching

        Returns:
            Score from current player's perspective
        """
        self.nodes_searched += 1

        # Check time
        if time.time() >= deadline:
            return self.evaluator(board, depth)

        # Terminal node checks
        if board.winner is not None:
            if board.winner == Result.PLAYER:
                return WIN_SCORE - depth if is_player_turn else -WIN_SCORE + depth
            elif board.winner == Result.ENEMY:
                return -WIN_SCORE + depth if is_player_turn else WIN_SCORE - depth
            else:  # Tie
                return 0

        # Depth limit reached
        if depth == 0:
            score = self.evaluator(board, depth)
            # Negamax: negate if enemy's turn
            return score if is_player_turn else -score

        # Check transposition table
        board_hash = self.transposition_table.compute_hash(board)
        tt_result = self.transposition_table.lookup(board_hash, depth)
        if tt_result is not None:
            self.tt_hits += 1
            return tt_result[0]

        # Get valid moves
        valid_moves = board.get_valid_moves(enemy=not is_player_turn)

        # CRITICAL FIX: ABSOLUTE BAN on turds for turns 0-3
        if board.turn_count <= ABSOLUTE_BAN_TURD_TURNS:
            valid_moves = [(d, t) for d, t in valid_moves if t != MoveType.TURD]
            if not valid_moves:  # Emergency fallback
                valid_moves = board.get_valid_moves(enemy=not is_player_turn)

        # No moves available (blocked)
        if not valid_moves:
            # Blocked penalty = -5 eggs worth of score
            # Using average egg value (can be tuned in weights_config.py)
            penalty_score = -BLOCKED_PENALTY_MULTIPLIER * 100.0  # Approximate egg value
            score = self.evaluator(board, depth) + penalty_score
            return score if is_player_turn else -score

        # Recurse on all moves
        best_score = -float('inf')
        best_move = valid_moves[0]

        # Order moves for better pruning
        ordered_moves = self._order_moves(board, valid_moves)

        for move in ordered_moves:
            if time.time() >= deadline:
                break

            sim_board = self._copy_board(board)
            if not self._apply_move(sim_board, move, enemy=not is_player_turn):
                continue

            # Negamax recursion: negate and swap alpha/beta
            score = -self._negamax(sim_board, depth - 1, -beta, -alpha,
                                  not is_player_turn, deadline)

            # Apply move scoring heuristics from config
            direction, move_type = move
            if is_player_turn:
                current_loc = board.chicken_player.get_location()

                if move_type == MoveType.EGG:
                    if self._is_corner(current_loc):
                        score += EGG_CORNER_BONUS
                    else:
                        score += EGG_REGULAR_BONUS

                elif move_type == MoveType.TURD:
                    x, y = current_loc
                    is_separator = (x in SEPARATOR_LINES or y in SEPARATOR_LINES)
                    turds_left = board.chicken_player.get_turds_left()
                    turn = board.turn_count

                    # Absolute ban on early turds
                    if turn <= ABSOLUTE_BAN_TURD_TURNS:
                        score -= TURD_ABSOLUTE_BAN_PENALTY

                    elif turn < EARLY_GAME_THRESHOLD:
                        # Early game: Very conservative
                        if is_separator and turds_left >= EARLY_GAME_MIN_TURDS:
                            score += TURD_EARLY_SEPARATOR_BONUS
                        else:
                            score -= TURD_EARLY_HEAVY_PENALTY

                    elif turn < MID_GAME_THRESHOLD:
                        # Mid game: Strategic separator use
                        if is_separator and turds_left >= MID_GAME_MIN_TURDS:
                            score += TURD_MID_SEPARATOR_BONUS
                        elif turds_left <= 1:
                            score -= TURD_MID_SAVE_LAST_PENALTY
                        else:
                            score -= TURD_MID_GENERAL_PENALTY

                    else:
                        # Late game: Active turd use
                        if is_separator:
                            score += TURD_LATE_SEPARATOR_BONUS
                        else:
                            score += TURD_LATE_NON_SEPARATOR_BONUS

                    # Non-separator penalty
                    if not is_separator:
                        score -= TURD_NON_SEPARATOR_PENALTY

                elif move_type == MoveType.PLAIN:
                    score -= PLAIN_MOVE_PENALTY

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff

        # Store in transposition table
        self.transposition_table.store(board_hash, best_score, depth, best_move)

        return best_score

    def _order_moves(self, board: "game_board.Board",
                    moves: List[Tuple[Direction, MoveType]]) -> List[Tuple[Direction, MoveType]]:
        """
        Order moves for optimal alpha-beta pruning.
        Uses pure heuristic ordering based on move type and position.

        Priority (higher priority searched first for better alpha-beta cutoffs):
        1. Corner eggs (4 eggs!)
        2. Separator eggs (strategic territory control)
        3. Regular eggs
        4. Plain moves
        5. Strategic late-game turds
        6. Early-game turds (strongly discouraged)
        """
        current_loc = board.chicken_player.get_location()
        turn = board.turn_count

        def move_priority(move: Tuple[Direction, MoveType]) -> float:
            direction, move_type = move

            if move_type == MoveType.EGG:
                # Check if laying in corner (worth 4 eggs!)
                if self._is_corner(current_loc):
                    priority = CORNER_EGG_PRIORITY  # Highest priority
                else:
                    priority = REGULAR_EGG_PRIORITY  # High priority

                # Separator bonus
                x, y = current_loc
                if x in SEPARATOR_LINES or y in SEPARATOR_LINES:
                    priority += SEPARATOR_EGG_BONUS  # Extra priority for separators

                return priority

            elif move_type == MoveType.PLAIN:
                return PLAIN_MOVE_PRIORITY  # Neutral

            else:  # MoveType.TURD
                # Turd priority depends heavily on game phase
                if turn <= ABSOLUTE_BAN_TURD_TURNS:
                    return TURD_ABSOLUTE_BAN_PRIORITY  # Lowest priority (absolute ban)
                elif turn < EARLY_GAME_THRESHOLD:
                    return TURD_EARLY_GAME_PRIORITY  # Very low priority
                else:
                    # Check if separator
                    dest_x, dest_y = loc_after_direction(current_loc, direction)
                    if dest_x in SEPARATOR_LINES or dest_y in SEPARATOR_LINES:
                        return TURD_SEPARATOR_PRIORITY  # Low but acceptable for separator
                    return TURD_NON_SEPARATOR_PRIORITY  # Very low priority for non-separator

        return sorted(moves, key=move_priority)

    def _prune_loops(self, board: "game_board.Board",
                    moves: List[Tuple[Direction, MoveType]],
                    history: List[Tuple[int, int]]) -> List[Tuple[Direction, MoveType]]:
        """
        Remove moves that would revisit recent positions (loop prevention).

        Args:
            board: Current board
            moves: List of valid moves
            history: Recent position history

        Returns:
            Filtered list of moves
        """
        if not history:
            return moves

        current_loc = board.chicken_player.get_location()
        non_loop_moves = []

        for direction, move_type in moves:
            dest = loc_after_direction(current_loc, direction)
            if dest not in history:
                non_loop_moves.append((direction, move_type))

        # If all moves filtered out, allow loops as fallback
        return non_loop_moves if non_loop_moves else moves

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        """Check if location is a corner"""
        x, y = loc
        return (x == 0 or x == 7) and (y == 0 or y == 7)


    def _copy_board(self, board_obj: "game_board.Board") -> "game_board.Board":
        """Deep copy board for simulation"""
        new_board = game_board.Board(board_obj.game_map, copy=True)

        new_board.eggs_player = board_obj.eggs_player.copy()
        new_board.eggs_enemy = board_obj.eggs_enemy.copy()
        new_board.turds_player = board_obj.turds_player.copy()
        new_board.turds_enemy = board_obj.turds_enemy.copy()
        new_board.found_trapdoors = board_obj.found_trapdoors.copy()

        new_board.chicken_player = self._copy_chicken(board_obj.chicken_player)
        new_board.chicken_enemy = self._copy_chicken(board_obj.chicken_enemy)

        new_board.turn_count = board_obj.turn_count
        new_board.turns_left_player = board_obj.turns_left_player
        new_board.turns_left_enemy = board_obj.turns_left_enemy
        new_board.winner = board_obj.winner
        new_board.win_reason = board_obj.win_reason

        return new_board

    def _copy_chicken(self, chick) -> "chicken.Chicken":
        """Copy chicken object"""
        from game import chicken
        new = chicken.Chicken(copy=True)
        new.loc = chick.loc
        new.spawn = chick.spawn
        new.even_chicken = chick.even_chicken
        new.turds_left = chick.turds_left
        new.eggs_laid = chick.eggs_laid
        return new

    def _apply_move(self, board: "game_board.Board",
                   move: Tuple[Direction, MoveType], enemy: bool) -> bool:
        """
        Apply move to board with all bonuses.

        Returns:
            True if move applied successfully, False otherwise
        """
        direction, move_type = move

        try:
            if not board.is_valid_move(direction, move_type, enemy):
                return False

            if enemy:
                old = board.chicken_enemy.get_location()
                new = loc_after_direction(old, direction)
                board.chicken_enemy.loc = new

                if move_type == MoveType.EGG:
                    board.eggs_enemy.add(old)
                    # Corner bonus: +3 extra eggs
                    if self._is_corner(old):
                        board.chicken_enemy.eggs_laid += 4
                    else:
                        board.chicken_enemy.eggs_laid += 1
                elif move_type == MoveType.TURD:
                    board.turds_enemy.add(old)
                    board.chicken_enemy.turds_left -= 1

                board.turns_left_enemy -= 1
            else:
                old = board.chicken_player.get_location()
                new = loc_after_direction(old, direction)
                board.chicken_player.loc = new

                if move_type == MoveType.EGG:
                    board.eggs_player.add(old)
                    # Corner bonus: +3 extra eggs
                    if self._is_corner(old):
                        board.chicken_player.eggs_laid += 4
                    else:
                        board.chicken_player.eggs_laid += 1
                elif move_type == MoveType.TURD:
                    board.turds_player.add(old)
                    board.chicken_player.turds_left -= 1

                board.turns_left_player -= 1

            board.turn_count += 1

            # Check for blocking bonus: +5 eggs
            if not enemy:
                enemy_moves = board.get_valid_moves(enemy=True)
                if not enemy_moves and board.turns_left_enemy > 0:
                    board.chicken_player.eggs_laid += 5
            else:
                player_moves = board.get_valid_moves(enemy=False)
                if not player_moves and board.turns_left_player > 0:
                    board.chicken_enemy.eggs_laid += 5

            # Check game end
            if board.turns_left_player <= 0 and board.turns_left_enemy <= 0:
                if board.chicken_player.eggs_laid > board.chicken_enemy.eggs_laid:
                    board.winner = Result.PLAYER
                elif board.chicken_enemy.eggs_laid > board.chicken_player.eggs_laid:
                    board.winner = Result.ENEMY
                else:
                    board.winner = Result.TIE

            return True
        except Exception as e:
            return False

    def _filter_safe_moves(self, board: "game_board.Board",
                          valid_moves: List[Tuple],
                          enemy: bool = False) -> List[Tuple]:
        """
        LAVA FLOOR PROTOCOL: Filter moves to only include SAFE destinations.

        Returns only moves where destination has risk <= SAFETY_THRESHOLD (5%).
        If NO safe moves exist, returns empty list (caller handles fallback).
        """
        from game.enums import loc_after_direction

        # Try to get tracker from evaluator
        tracker = None
        try:
            # Try different possible locations for tracker
            if hasattr(self.evaluator, 'tracker'):
                tracker = self.evaluator.tracker
            elif hasattr(self.evaluator, 'neural_eval') and hasattr(self.evaluator.neural_eval, 'tracker'):
                tracker = self.evaluator.neural_eval.tracker
        except:
            pass

        # If no tracker available, return all moves (safety check disabled)
        if tracker is None:
            return valid_moves

        # Get current position
        if enemy:
            current_loc = board.chicken_enemy.get_location()
        else:
            current_loc = board.chicken_player.get_location()

        safe_moves = []
        for direction, move_type in valid_moves:
            # Calculate destination
            new_loc = loc_after_direction(current_loc, direction)

            # Check if destination is safe using tracker
            if tracker.is_safe(new_loc):
                safe_moves.append((direction, move_type))

        return safe_moves
