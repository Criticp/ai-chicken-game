"""
Tactical Engine: Quiescence Search & Selective Extensions

Implements:
1. Quiescence Search - Don't stop at tactical positions (turds/eggs)
2. Selective Extensions - Search deeper for critical turd placements
3. Turd Evaluator - Specialized evaluation for turd impact
4. Move Ordering - Prioritize aggressive moves in tactical sequences

Philosophy:
- Turds are PERMANENT - need deeper search (10-12 ply)
- Eggs create territory - need tactical evaluation (6-8 ply)
- Plain moves are positional - use normal depth (3-5 ply)
"""

from typing import Tuple, Optional
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Direction, MoveType, loc_after_direction
from game.board import manhattan_distance


class TurdEvaluator:
    """
    Specialized evaluator for turd placements.

    Evaluates turd impact on:
    - Enemy mobility reduction
    - Territory control
    - Checkmate potential
    - Strategic board control
    """

    def __init__(self, trapdoor_tracker):
        self.tracker = trapdoor_tracker

    def evaluate_turd_placement(self, board: "game_board.Board",
                               turd_loc: Tuple[int, int],
                               is_our_turd: bool) -> float:
        """
        Evaluate the strategic value of a turd placement.

        Args:
            board: Current board state
            turd_loc: Location where turd will be placed
            is_our_turd: True if we're placing it, False if enemy

        Returns:
            Score representing turd value (-100 to +100)
        """
        score = 0.0

        enemy_loc = board.chicken_enemy.get_location() if is_our_turd else board.chicken_player.get_location()
        our_loc = board.chicken_player.get_location() if is_our_turd else board.chicken_enemy.get_location()

        # ================================================================
        # FACTOR 1: MOBILITY REDUCTION (Most Important)
        # ================================================================
        # Count enemy moves before and after turd
        enemy_moves_before = self._count_valid_moves(board, enemy_loc, not is_our_turd, set())

        # Simulate turd placement
        simulated_turds = board.turds_player.copy() if is_our_turd else board.turds_enemy.copy()
        simulated_turds.add(turd_loc)
        enemy_moves_after = self._count_valid_moves(board, enemy_loc, not is_our_turd, simulated_turds)

        mobility_reduction = enemy_moves_before - enemy_moves_after

        # CRITICAL: Checkmate detection (enemy 0 moves)
        if enemy_moves_after == 0:
            score += 1000.0  # Instant win
        elif enemy_moves_after == 1:
            score += 300.0   # Near checkmate
        elif enemy_moves_after == 2:
            score += 150.0   # Severe restriction
        else:
            # Score based on mobility reduction
            score += mobility_reduction * 25.0

        # ================================================================
        # FACTOR 2: TERRITORY CONTROL
        # ================================================================
        # Turds near center are more valuable (control more space)
        center_x, center_y = 3.5, 3.5
        dist_to_center = abs(turd_loc[0] - center_x) + abs(turd_loc[1] - center_y)
        centrality = (7.0 - dist_to_center) / 7.0  # 0.0 to 1.0
        score += centrality * 15.0

        # ================================================================
        # FACTOR 3: PROXIMITY TO ENEMY (Offensive Value)
        # ================================================================
        dist_to_enemy = manhattan_distance(turd_loc, enemy_loc)

        if dist_to_enemy == 2:
            # Adjacent to enemy's reachable squares - very aggressive
            score += 40.0
        elif dist_to_enemy == 3:
            # Close range - good pressure
            score += 20.0
        elif dist_to_enemy >= 6:
            # Too far - wasted turd
            score -= 30.0

        # ================================================================
        # FACTOR 4: EGG PROTECTION
        # ================================================================
        # Turds near our eggs protect them
        our_eggs = board.eggs_player if is_our_turd else board.eggs_enemy
        for egg in our_eggs:
            dist = manhattan_distance(turd_loc, egg)
            if dist == 2:
                score += 10.0  # Forms protective barrier

        # ================================================================
        # FACTOR 5: CHOKE POINT CONTROL
        # ================================================================
        # Turds in narrow corridors are more effective
        neighbors = [
            (turd_loc[0] + dx, turd_loc[1] + dy)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
        ]

        blocked_sides = 0
        for neighbor in neighbors:
            if not board.is_valid_cell(neighbor):
                blocked_sides += 1
            elif neighbor in board.turds_player or neighbor in board.turds_enemy:
                blocked_sides += 1

        if blocked_sides >= 2:
            # Creating choke point
            score += 20.0 * blocked_sides

        # ================================================================
        # FACTOR 6: AVOID WASTING TURDS
        # ================================================================
        # Penalize turds that don't do much
        if mobility_reduction == 0 and dist_to_enemy >= 4:
            score -= 50.0  # Useless turd

        return score

    def _count_valid_moves(self, board: "game_board.Board",
                          location: Tuple[int, int],
                          is_enemy: bool,
                          extra_turds: set) -> int:
        """
        Count valid moves from a location, considering extra turds.

        Args:
            board: Board state
            location: Location to check from
            is_enemy: True if checking enemy moves
            extra_turds: Additional turds to consider (for simulation)

        Returns:
            Number of valid moves
        """
        count = 0

        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            dest = loc_after_direction(location, direction)

            # Check validity
            if not board.is_valid_cell(dest):
                continue

            # Check eggs
            if is_enemy:
                if dest in board.eggs_player:
                    continue
            else:
                if dest in board.eggs_enemy:
                    continue

            # Check turds (including simulated ones)
            is_blocked = False

            # Check all turds (existing + simulated)
            all_turds = (board.turds_enemy if is_enemy else board.turds_player) | extra_turds

            for turd in all_turds:
                if manhattan_distance(dest, turd) <= 1:
                    is_blocked = True
                    break

            if is_blocked:
                continue

            # Valid move
            count += 1

        return count


class TacticalEngine:
    """
    Handles quiescence search and selective extensions.

    Used by SearchEngine to:
    1. Continue search in tactical positions (don't stop at turds/eggs)
    2. Extend search for critical moves (deep turd analysis)
    3. Order moves for better alpha-beta cutoffs
    """

    def __init__(self, evaluator, trapdoor_tracker):
        """
        Args:
            evaluator: Main HybridEvaluator instance
            trapdoor_tracker: TrapdoorTracker for safety checks
        """
        self.evaluator = evaluator
        self.tracker = trapdoor_tracker
        self.turd_evaluator = TurdEvaluator(trapdoor_tracker)

        # Configuration
        self.quiescence_max_depth = 6  # Max quiescence ply
        self.turd_extension_depth = 4  # Extra ply for turd moves
        self.egg_extension_depth = 2   # Extra ply for egg moves

        # Statistics
        self.tactical_extensions = 0  # Count extensions used

    def is_tactical_move(self, move: Tuple[Direction, MoveType]) -> bool:
        """
        Check if a move is tactical (requires quiescence/extension).

        Args:
            move: (Direction, MoveType) tuple

        Returns:
            True if move is tactical (turd or egg)
        """
        return move[1] in [MoveType.TURD, MoveType.EGG]

    def should_extend_search(self, move: Tuple[Direction, MoveType],
                            board: "game_board.Board") -> bool:
        """
        Determine if search should be extended for this move.

        Extensions for:
        - Turds (permanent, need deep analysis)
        - Virgin eggs (high value)
        - Checkmate threats

        Args:
            move: Move being considered
            board: Current board state

        Returns:
            True if search should be extended
        """
        # Always extend turds (permanent moves)
        if move[1] == MoveType.TURD:
            return True

        # Extend virgin eggs (edges/corners)
        if move[1] == MoveType.EGG:
            current_loc = board.chicken_player.get_location()
            dest = loc_after_direction(current_loc, move[0])

            # Virgin egg check
            if dest not in board.eggs_player and dest not in board.eggs_enemy:
                x, y = dest
                if x == 0 or x == 7 or y == 0 or y == 7:
                    return True  # Edge egg

        return False

    def get_extension_depth(self, move: Tuple[Direction, MoveType],
                           board: "game_board.Board") -> int:
        """
        Get extension depth for a move.

        Args:
            move: Move being extended
            board: Current board state

        Returns:
            Number of extra ply to search
        """
        if move[1] == MoveType.TURD:
            # Turds get deepest extension (permanent impact)
            return self.turd_extension_depth

        if move[1] == MoveType.EGG:
            # Eggs get moderate extension
            return self.egg_extension_depth

        return 0

    def quiescence_search(self, board: "game_board.Board",
                         alpha: float, beta: float,
                         maximizing: bool,
                         depth_remaining: int,
                         deadline: float) -> float:
        """
        Quiescence search - continue searching tactical moves.

        Prevents horizon effect by searching turd/egg sequences
        until a quiet position is reached.

        Args:
            board: Board state
            alpha: Alpha value
            beta: Beta value
            maximizing: True if maximizing player (our turn)
            depth_remaining: Remaining quiescence depth
            deadline: Time deadline

        Returns:
            Evaluation score
        """
        import time

        # Check time
        if time.time() >= deadline:
            return self.evaluator.evaluate(board, 0)

        # Base case: no more quiescence depth
        if depth_remaining <= 0:
            return self.evaluator.evaluate(board, 0)

        # Stand-pat evaluation
        stand_pat = self.evaluator(board, 0)

        if maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        # Get tactical moves only
        valid_moves = board.get_valid_moves(enemy=not maximizing)
        tactical_moves = [m for m in valid_moves if self.is_tactical_move(m)]

        if not tactical_moves:
            return stand_pat

        # Order tactical moves (turds first, then eggs)
        ordered_moves = self._order_tactical_moves(board, tactical_moves, maximizing)

        # Search tactical moves
        for move in ordered_moves:
            if time.time() >= deadline:
                break

            # Apply move
            sim_board = self._copy_board(board)
            if not self._apply_move(sim_board, move, enemy=not maximizing):
                continue

            # Recursive quiescence
            score = self.quiescence_search(sim_board, alpha, beta,
                                          not maximizing, depth_remaining - 1,
                                          deadline)

            if maximizing:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score

        return alpha if maximizing else beta

    def _order_tactical_moves(self, board: "game_board.Board",
                             moves: list,
                             maximizing: bool) -> list:
        """
        Order tactical moves for better alpha-beta cutoffs.

        Priority:
        1. Checkmate turds (enemy 0 moves)
        2. Aggressive turds (high mobility reduction)
        3. Virgin eggs (corners/edges)
        4. Expansion eggs
        5. Other turds

        Args:
            board: Board state
            moves: List of moves to order
            maximizing: True if maximizing player

        Returns:
            Ordered list of moves
        """
        scored_moves = []

        for move in moves:
            score = 0.0

            if move[1] == MoveType.TURD:
                # Evaluate turd quality
                current_loc = board.chicken_player.get_location() if maximizing else board.chicken_enemy.get_location()
                turd_score = self.turd_evaluator.evaluate_turd_placement(board, current_loc, maximizing)
                score = turd_score

            elif move[1] == MoveType.EGG:
                # Evaluate egg quality
                current_loc = board.chicken_player.get_location() if maximizing else board.chicken_enemy.get_location()
                dest = loc_after_direction(current_loc, move[0])

                # Virgin egg bonus
                if dest not in board.eggs_player and dest not in board.eggs_enemy:
                    x, y = dest
                    if (x == 0 or x == 7) and (y == 0 or y == 7):
                        score = 100.0  # Corner
                    elif x == 0 or x == 7 or y == 0 or y == 7:
                        score = 50.0   # Edge

                # Expansion egg bonus
                for adj_dir in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                    adj_loc = loc_after_direction(dest, adj_dir)
                    our_eggs = board.eggs_player if maximizing else board.eggs_enemy
                    if adj_loc in our_eggs:
                        score += 30.0

            scored_moves.append((score, move))

        # Sort descending
        scored_moves.sort(reverse=True, key=lambda x: x[0])

        return [move for _, move in scored_moves]

    def _copy_board(self, board: "game_board.Board") -> "game_board.Board":
        """Create a deep copy of the board for simulation"""
        new_board = game_board.Board(board.game_map, copy=True)

        new_board.eggs_player = board.eggs_player.copy()
        new_board.eggs_enemy = board.eggs_enemy.copy()
        new_board.turds_player = board.turds_player.copy()
        new_board.turds_enemy = board.turds_enemy.copy()
        new_board.found_trapdoors = board.found_trapdoors.copy()

        new_board.chicken_player = self._copy_chicken(board.chicken_player)
        new_board.chicken_enemy = self._copy_chicken(board.chicken_enemy)

        new_board.turn_count = board.turn_count
        new_board.turns_left_player = board.turns_left_player
        new_board.turns_left_enemy = board.turns_left_enemy
        new_board.winner = board.winner
        new_board.win_reason = board.win_reason

        return new_board

    def _copy_chicken(self, chick) -> "chicken.Chicken":
        """Copy chicken object"""
        from game import chicken
        new = chicken.Chicken(copy=True)
        new.location = chick.location  # Use location property
        new.spawn = chick.spawn
        new.even_chicken = chick.even_chicken
        new.turds_left = chick.turds_left
        new.eggs_laid = chick.eggs_laid
        return new

    def _apply_move(self, board: "game_board.Board",
                   move: Tuple[Direction, MoveType],
                   enemy: bool) -> bool:
        """
        Apply a move to the board.

        Args:
            board: Board to modify
            move: Move to apply
            enemy: True if enemy move, False if our move

        Returns:
            True if move was applied successfully
        """
        direction, move_type = move

        try:
            # Update chicken position
            chicken = board.chicken_enemy if enemy else board.chicken_player
            current_loc = chicken.get_location()
            new_loc = loc_after_direction(current_loc, direction)

            # Move chicken
            chicken.location = new_loc

            # Handle move type
            if move_type == MoveType.EGG:
                if enemy:
                    board.eggs_enemy.add(new_loc)
                    chicken.eggs_left -= 1
                else:
                    board.eggs_player.add(new_loc)
                    chicken.eggs_left -= 1

            elif move_type == MoveType.TURD:
                if enemy:
                    board.turds_enemy.add(current_loc)
                    chicken.turds_left -= 1
                else:
                    board.turds_player.add(current_loc)
                    chicken.turds_left -= 1

            return True

        except Exception as e:
            return False

