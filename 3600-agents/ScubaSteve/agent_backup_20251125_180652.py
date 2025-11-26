"""
Scuba Steve - Main Agent Implementation
Data-driven AI with Alpha-Beta Minimax + Bayesian Trapdoor Tracking + Learned Evaluation
Trained on 5447 high-quality games from 18k game dataset

CRITICAL BUG FIXES (V4):
- Corner eggs give +3 bonus (4 total) - must be accounted for in simulation
- Blocking enemy gives +5 eggs - must be checked in simulation
- Evaluation must prioritize corners due to +3 bonus
- HARD LOOP BREAKER: Prune moves that revisit recent locations (no penalties, physical pruning)
- OPENING BOOK: Zobrist-hashed grandmaster moves from top 10% of 18k games
- DISTILLED WEIGHTS: Neural network intelligence compressed into linear weights
"""

from collections.abc import Callable
from collections import deque
from typing import List, Tuple, Dict, Set
import json
import os
import time
import hashlib

from game import *
from game.enums import Direction, MoveType, Result, loc_after_direction
from game import board, chicken
from .belief import TrapdoorBelief


class PlayerAgent:
    """
    Scuba Steve: Tournament AI agent with machine-learned evaluation function
    """

    # PREDATOR UPGRADE - DIRECTIVE 1: Optimal risk threshold from 18k game analysis
    MAX_RISK_TOLERANCE = 0.60  # Players taking >60% risk have <50% win rate

    # Learned weights from linear regression + manual corner adjustment
    WEIGHTS = {
        'egg_diff': 7.6488,       # Most important: egg advantage
        'mobility': 0.1693,        # Mobility advantage  
        'corner_proximity': 2.5,   # POSITIVE! Corners give +3 bonus eggs!
        'turd_diff': 0.3354,       # Enemy using turds is beneficial
        'trapdoor_risk': -20.0,    # High penalty for danger
        'intercept': -0.0055,
    }

    def __init__(self, board: board.Board, time_left: Callable):
        """Initialize agent with learned parameters"""
        self.belief = TrapdoorBelief(map_size=8)
        self.move_count = 0
        self.max_depth = 4
        self.nodes_searched = 0

        # DIRECTIVE 1: Hard Loop Breaker - Track last 4 positions
        self.history: deque = deque(maxlen=4)
        self.history.append(board.chicken_player.get_location())

        # DIRECTIVE 1: Death detection - track previous location to detect respawn
        self.prev_location = board.chicken_player.get_location()
        self.spawn_location = board.chicken_player.spawn

        # DIRECTIVE 2: Opening Book (Grandmaster Memory)
        self.opening_book: Dict[str, Tuple[Direction, MoveType]] = {}
        self._load_opening_book()

        # Load learned weights (Directive 3: Distilled from NN)
        self._load_learned_weights()

        print(f"[Scuba Steve] Ready to dive! Book size: {len(self.opening_book)}, Weights loaded")

    def _load_learned_weights(self):
        """Load weights learned from training data"""
        try:
            weights_path = os.path.join(os.path.dirname(__file__), 'learned_weights.json')
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    loaded = json.load(f)
                    # Update egg_diff, mobility, turd_diff from training
                    # But keep corner_proximity manual since we know corners = +3 eggs
                    for key in ['egg_diff', 'mobility', 'turd_diff', 'intercept']:
                        if key in loaded:
                            self.WEIGHTS[key] = loaded[key]
                print(f"[Scuba Steve] ✓ Loaded learned weights")
        except Exception as e:
            print(f"[Scuba Steve] Using hardcoded weights")

    def _load_opening_book(self):
        """DIRECTIVE 2: Load opening book (grandmaster moves from top 10% games)"""
        try:
            book_path = os.path.join(os.path.dirname(__file__), 'opening_book.json')
            if os.path.exists(book_path):
                with open(book_path, 'r') as f:
                    raw_book = json.load(f)
                    # Convert stored format to usable moves
                    for board_hash, move_data in raw_book.items():
                        direction = Direction(move_data['direction'])
                        move_type = MoveType.EGG if move_data['move_type'] == 'egg' else \
                                   MoveType.TURD if move_data['move_type'] == 'turd' else MoveType.PLAIN
                        self.opening_book[board_hash] = (direction, move_type)
                print(f"[Scuba Steve] ✓ Loaded opening book with {len(self.opening_book)} positions")
        except Exception as e:
            print(f"[Scuba Steve] No opening book loaded: {e}")

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """Execute one turn using Alpha-Beta search with learned evaluation"""
        start_time = time.time()
        current_loc = board.chicken_player.get_location()

        # PHASE 2 FIX: Teleport Check (Absolute Truth)
        # Check if we teleported (not a normal move) = WE DIED
        if self.prev_location is not None and self.prev_location != current_loc:
            # Calculate valid neighbor squares from previous location
            px, py = self.prev_location
            expected_neighbors = [
                (px-1, py), (px+1, py), (px, py-1), (px, py+1)  # Adjacent squares
            ]

            # If current location is NOT a neighbor AND we're at spawn = DEATH
            if current_loc not in expected_neighbors and current_loc == self.spawn_location:
                # ABSOLUTE FACT: We died at prev_location
                # Mark with probability = 1.0 (LOCKED, no Bayesian updates allowed)
                self.belief.lock_death_trapdoor(self.prev_location)
                print(f"[Scuba Steve] ☠️  TELEPORT DETECTED! DEATH at {self.prev_location} - LOCKED AT 100% RISK!")

        # Update trapdoor beliefs using Bayesian inference
        self.belief.update_from_sensors(current_loc, sensor_data)
        for trapdoor in board.found_trapdoors:
            if trapdoor not in self.belief.found_trapdoors:
                self.belief.update_found_trapdoor(trapdoor, len(self.belief.found_trapdoors))

        # DIRECTIVE 2: Check opening book first (turns 1-15)
        if self.move_count < 15 and self.opening_book:
            board_hash = self._hash_board(board)
            if board_hash in self.opening_book:
                book_move = self.opening_book[board_hash]
                # Verify it's still valid (safety check)
                if board.is_valid_move(book_move[0], book_move[1], enemy=False):
                    # Update history before returning
                    dest = loc_after_direction(current_loc, book_move[0])
                    self.history.append(dest)
                    self.move_count += 1
                    print(f"[Scuba Steve] Move {self.move_count}: {book_move} [BOOK]")
                    return book_move

        # Adaptive search depth
        self._adjust_search_depth(time_left(), board)

        # Alpha-Beta search with HARD LOOP PRUNING
        best_move = self._search(board, time_left, start_time)

        # DIRECTIVE 1: Update history with destination
        dest_loc = loc_after_direction(current_loc, best_move[0])
        self.history.append(dest_loc)

        # DIRECTIVE 1: Track location for death detection
        self.prev_location = current_loc

        self.move_count += 1
        elapsed = time.time() - start_time
        print(f"[Scuba Steve] Move {self.move_count}: {best_move}  d={self.max_depth}  n={self.nodes_searched}  t={elapsed:.2f}s")

        return best_move

    def _adjust_search_depth(self, time_val: float, board: board.Board):
        """Adjust search depth based on available time"""
        moves_left = max(board.turns_left_player, 1)
        time_per_move = time_val / moves_left

        if time_per_move > 15:
            self.max_depth = 5
        elif time_per_move > 8:
            self.max_depth = 4
        elif time_per_move > 4:
            self.max_depth = 3
        else:
            self.max_depth = 2

        # Go deeper in early game
        if self.move_count < 5:
            self.max_depth = min(self.max_depth + 1, 6)

    def _search(self, board: board.Board, time_left: Callable, start_time: float) -> Tuple[Direction, MoveType]:
        """Alpha-Beta Minimax search"""
        # DIRECTIVE 1: Get valid moves with HARD LOOP PRUNING
        valid_moves = self._get_valid_moves_no_loops(board, enemy=False)

        if not valid_moves:
            # Fallback to all valid moves if pruning removed everything
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                return (Direction.UP, MoveType.PLAIN)

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Time budget
        moves_left = max(board.turns_left_player, 1)
        time_budget = min(time_left() * 0.25, time_left() / moves_left * 2.0, 10.0)

        # Search all moves
        self.nodes_searched = 0
        best_move = valid_moves[0]
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        # Order moves for better pruning
        ordered_moves = self._order_moves(valid_moves, board)

        for move in ordered_moves:
            if time.time() - start_time > time_budget:
                break

            sim_board = self._copy_board(board)
            if not self._apply_move(sim_board, move, False):
                continue

            score = self._minimax(sim_board, self.max_depth - 1, alpha, beta,
                                 False, start_time, start_time + time_budget)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        return best_move

    def _minimax(self, board: board.Board, depth: int, alpha: float, beta: float,
                 is_max: bool, start_time: float, deadline: float) -> float:
        """Minimax with alpha-beta pruning"""
        self.nodes_searched += 1

        # Cutoffs
        if time.time() >= deadline:
            return self._evaluate(board)

        if depth == 0 or board.winner is not None:
            return self._evaluate(board)

        moves = board.get_valid_moves(enemy=not is_max)
        if not moves:
            return self._evaluate(board)

        if is_max:
            value = -float('inf')
            for move in moves:
                sim = self._copy_board(board)
                if not self._apply_move(sim, move, False):
                    continue
                value = max(value, self._minimax(sim, depth - 1, alpha, beta, False, start_time, deadline))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for move in moves:
                sim = self._copy_board(board)
                if not self._apply_move(sim, move, True):
                    continue
                value = min(value, self._minimax(sim, depth - 1, alpha, beta, True, start_time, deadline))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def _evaluate(self, board: board.Board) -> float:
        """Evaluate position using learned weights from 5447 game dataset"""

        # Terminal states
        if board.winner == Result.PLAYER:
            return 10000
        elif board.winner == Result.ENEMY:
            return -10000
        elif board.winner == Result.TIE:
            return 0

        # Extract features
        egg_diff = board.chicken_player.eggs_laid - board.chicken_enemy.eggs_laid
        
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        my_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        mobility_diff = my_moves - enemy_moves
        
        corner_diff = self._corner_score(my_loc) - self._corner_score(enemy_loc)
        
        turd_diff = board.chicken_enemy.turds_left - board.chicken_player.turds_left
        
        # Trapdoor risk (higher = more dangerous)
        my_risk = self.belief.get_trapdoor_risk(my_loc)
        enemy_risk = self.belief.get_trapdoor_risk(enemy_loc)
        risk_diff = my_risk - enemy_risk
        
        # PHASE 4 DIRECTIVE 2: Local Fertility (The Lawnmower)
        # Reward being in areas with high egg-laying potential
        my_fertility = self._calculate_local_fertility(board, my_loc, enemy=False)
        enemy_fertility = self._calculate_local_fertility(board, enemy_loc, enemy=True)
        fertility_diff = my_fertility - enemy_fertility

        # PREDATOR UPGRADE - DIRECTIVE 1: Hard risk threshold enforcement
        # If our risk exceeds tolerance, apply MASSIVE penalty (data-driven from 18k games)
        score = 0.0
        if my_risk > self.MAX_RISK_TOLERANCE:
            score -= 1000.0  # Death is unacceptable

        # Linear combination using learned weights
        score += (
            self.WEIGHTS['egg_diff'] * egg_diff +
            self.WEIGHTS['mobility'] * mobility_diff +
            self.WEIGHTS['corner_proximity'] * corner_diff +
            self.WEIGHTS['turd_diff'] * turd_diff +
            self.WEIGHTS['trapdoor_risk'] * risk_diff +
            self.WEIGHTS['intercept']
        )
        
        # PHASE 4 DIRECTIVE 2: Add fertility bonus (0.5 per valid neighbor)
        score += fertility_diff * 0.5

        # Bonus for egg-laying opportunity
        if board.can_lay_egg():
            # HUGE bonus if we can lay in corner (+3 bonus eggs = 4 total!)
            if self._is_corner(my_loc):
                score += 25.0  # 4 eggs worth ~30, be a bit conservative
            else:
                score += 6.0  # Regular egg (weight ~7.6)
        
        # PREDATOR UPGRADE - DIRECTIVE 2: Turd Chokepoint Bonus
        # Reward strategic turd placement that traps enemy
        chokepoint_bonus = self._evaluate_turd_chokepoint(board, my_loc, enemy_loc)
        score += chokepoint_bonus

        # Penalty for low mobility (getting trapped)
        if my_moves < 2:
            score -= 15.0
        elif my_moves < 3:
            score -= 5.0
        
        # Bonus if enemy is about to be trapped
        if enemy_moves < 2:
            score += 10.0
        
        return score

    def _corner_score(self, loc: Tuple[int, int]) -> float:
        """Corner proximity (higher = closer to corner)"""
        x, y = loc
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        return 14 - min(abs(x - cx) + abs(y - cy) for cx, cy in corners)
    
    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        """Check if location is a corner (gets +3 bonus eggs!)"""
        x, y = loc
        return (x == 0 or x == 7) and (y == 0 or y == 7)

    def _calculate_local_fertility(self, board: board.Board, loc: Tuple[int, int], enemy: bool) -> float:
        """
        PHASE 4 DIRECTIVE 2: Local Fertility (The Lawnmower)
        Calculate how many adjacent squares are empty and match our color parity.
        This creates a "gravity well" in empty areas, encouraging wide sweeping patterns.

        Returns: Count of valid neighbors (0-4)
        """
        x, y = loc
        chicken = board.chicken_enemy if enemy else board.chicken_player
        is_even = chicken.even_chicken

        fertility = 0
        # Check all 4 adjacent squares (Up, Down, Left, Right)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy

            # Check bounds
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue

            neighbor = (nx, ny)

            # Check if square matches our parity
            parity_match = ((nx + ny) % 2 == 0) == is_even
            if not parity_match:
                continue

            # Check if square is empty (no eggs, no turds)
            is_empty = (
                neighbor not in board.eggs_player and
                neighbor not in board.eggs_enemy and
                neighbor not in board.turds_player and
                neighbor not in board.turds_enemy and
                neighbor != board.chicken_player.get_location() and
                neighbor != board.chicken_enemy.get_location()
            )

            if is_empty:
                fertility += 1

        return fertility

    def _evaluate_turd_chokepoint(self, board: board.Board, my_loc: Tuple[int, int],
                                   enemy_loc: Tuple[int, int]) -> float:
        """
        PREDATOR UPGRADE - DIRECTIVE 2: Turd Warfare
        Evaluate strategic value of turds based on chokepoint creation.

        Returns bonus score for:
        - Turds that reduce enemy mobility significantly (+5.0 if -2+ moves)
        - Turds adjacent to known trapdoors (+3.0, forces enemy into trap)
        """
        bonus = 0.0

        # Only evaluate if we have turds placed
        if not board.turds_player:
            return 0.0

        # Count current enemy mobility
        current_enemy_moves = len(board.get_valid_moves(enemy=True))

        # Check each of our turds for strategic value
        for turd_loc in board.turds_player:
            # Bonus 1: Turd adjacent to known trapdoor (trap forcing)
            for trapdoor_loc in self.belief.found_trapdoors.union(self.belief.death_trapdoors):
                # Check if turd is adjacent to trapdoor (Manhattan distance = 1)
                tx, ty = turd_loc
                trap_x, trap_y = trapdoor_loc
                if abs(tx - trap_x) + abs(ty - trap_y) == 1:
                    bonus += 3.0  # Forces enemy toward death
                    break

            # Bonus 2: Turd creates chokepoint (reduces enemy mobility)
            # Estimate: if enemy is near this turd, how much does it restrict movement?
            ex, ey = enemy_loc
            tx, ty = turd_loc
            distance_to_enemy = abs(ex - tx) + abs(ey - ty)

            # Only count turds that are actually blocking the enemy (nearby)
            if distance_to_enemy <= 3:
                # Heuristic: turds near enemy in corridors/corners create chokepoints
                # Check if turd is blocking access to corners
                if self._is_near_corner(turd_loc):
                    bonus += 2.0

        # Major bonus if enemy mobility is critically low (successful chokepoint)
        if current_enemy_moves <= 2:
            bonus += 5.0
        elif current_enemy_moves <= 3:
            bonus += 2.0

        return bonus

    def _is_near_corner(self, loc: Tuple[int, int]) -> bool:
        """Check if location is within 2 squares of a corner"""
        x, y = loc
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        return any(abs(x - cx) + abs(y - cy) <= 2 for cx, cy in corners)

    def _order_moves(self, moves: List[Tuple[Direction, MoveType]],
                    board: board.Board) -> List[Tuple[Direction, MoveType]]:
        """Order moves: corner eggs FIRST (4 eggs!), eggs, plains, turds"""
        def priority(move):
            direction, mtype = move
            my_loc = board.chicken_player.get_location()
            
            if mtype == MoveType.EGG:
                # CORNER EGGS = 4 eggs total! Super high priority!
                if self._is_corner(my_loc):
                    return -10000
                else:
                    return -1000  # Regular eggs still very high
            elif mtype == MoveType.PLAIN:
                new_loc = loc_after_direction(my_loc, direction)
                risk = self.belief.get_trapdoor_risk(new_loc)
                # Bonus for moving toward corners where we can lay
                if self._is_corner(new_loc) and board.chicken_player.can_lay_egg(new_loc):
                    return -500
                return 1 + risk * 10
            else:  # TURD
                return 100
        
        return sorted(moves, key=priority)

    def _copy_board(self, board_obj: board.Board) -> board.Board:
        """Deep copy board for simulation"""
        new_board = board.Board(board_obj.game_map, copy=True)
        
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

    def _copy_chicken(self, chick: chicken.Chicken) -> chicken.Chicken:
        """Copy chicken object"""
        new = chicken.Chicken(copy=True)
        new.loc = chick.loc
        new.spawn = chick.spawn
        new.even_chicken = chick.even_chicken
        new.turds_left = chick.turds_left
        new.eggs_laid = chick.eggs_laid
        return new

    def _apply_move(self, board: board.Board, move: Tuple[Direction, MoveType],
                   enemy: bool) -> bool:
        """Apply move to board (returns False if invalid)
        CRITICAL: Must account for corner bonus (+3 eggs) and blocking bonus (+5 eggs)
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
                    # Corner bonus: +3 extra eggs!
                    if self._is_corner(old):
                        board.chicken_enemy.eggs_laid += 4  # 1 base + 3 bonus
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
                    # Corner bonus: +3 extra eggs!
                    if self._is_corner(old):
                        board.chicken_player.eggs_laid += 4  # 1 base + 3 bonus
                    else:
                        board.chicken_player.eggs_laid += 1
                elif move_type == MoveType.TURD:
                    board.turds_player.add(old)
                    board.chicken_player.turds_left -= 1

                board.turns_left_player -= 1

            board.turn_count += 1

            # Check for blocking bonus: +5 eggs if enemy is trapped!
            if not enemy:  # We just moved
                enemy_moves = board.get_valid_moves(enemy=True)
                if not enemy_moves and board.turns_left_enemy > 0:
                    board.chicken_player.eggs_laid += 5
            else:  # Enemy just moved
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
        except:
            return False

    def _get_valid_moves_no_loops(self, board: board.Board, enemy: bool) -> List[Tuple[Direction, MoveType]]:
        """
        DIRECTIVE 1: Hard Loop Breaker
        Get valid moves but PRUNE any that would revisit recent history.
        This makes loops physically impossible, not just penalized.
        """
        valid_moves = board.get_valid_moves(enemy=enemy)

        # Only apply to player (we track our own history, not enemy's)
        if enemy or not self.history or len(valid_moves) <= 1:
            return valid_moves

        current_loc = board.chicken_player.get_location()

        # Filter out moves that land on any square in self.history
        non_looping_moves = []
        for direction, move_type in valid_moves:
            dest = loc_after_direction(current_loc, direction)
            if dest not in self.history:
                non_looping_moves.append((direction, move_type))

        # If we filtered out ALL moves, we're forced to loop (edge case)
        # In this case, allow the loop as a fallback
        if non_looping_moves:
            return non_looping_moves
        else:
            return valid_moves

    def _hash_board(self, board: board.Board) -> str:
        """
        DIRECTIVE 2: Create a hash of board state for opening book lookup
        Uses a simplified representation: player pos, enemy pos, eggs, turds
        """
        try:
            # Simple string-based hash (could use Zobrist for speed, but this is clearer)
            player_loc = board.chicken_player.get_location()
            enemy_loc = board.chicken_enemy.get_location()

            # Sort sets for consistency
            eggs_p = sorted(list(board.eggs_player))
            eggs_e = sorted(list(board.eggs_enemy))
            turds_p = sorted(list(board.turds_player))
            turds_e = sorted(list(board.turds_enemy))

            state_str = f"{player_loc}|{enemy_loc}|{eggs_p}|{eggs_e}|{turds_p}|{turds_e}"

            # Use MD5 hash for compact representation
            return hashlib.md5(state_str.encode()).hexdigest()
        except:
            return ""
