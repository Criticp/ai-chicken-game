"""
ScubaSteve V4 - Predator Upgrade Agent

A tactical AI agent for the Chicken Game that uses:
1. Linear evaluation model with distilled neural network weights
2. Trapdoor belief tracking with persistent Death Memory
3. Turd Warfare: Spatial chokepoint strategy for offensive turd placement
4. Data-driven risk tolerance threshold

Key Features:
- MAX_RISK_TOLERANCE: Derived from data analysis (default 0.18)
- Turd_Chokepoint_Score: +5.0 for reducing enemy mobility by >2
- Turd_Adjacent_Trapdoor: +3.0 for forcing enemy toward known traps
- Death Memory: Confirmed trapdoors are permanent No-Go Zones
"""

import json
import os
from collections.abc import Callable
from typing import Dict, List, Set, Tuple, Optional

from game import board, enums
from game.enums import Direction, MoveType, loc_after_direction

# Import belief module for trapdoor tracking
from .belief import TrapdoorBelief


class PlayerAgent:
    """
    ScubaSteve V4 - Predator Upgrade
    
    A deployment-ready agent using linear evaluation with distilled
    neural network intelligence for fast, tactical gameplay.
    """
    
    # Data-derived maximum risk tolerance (Death Line threshold)
    # Players who step on >18% risk squares lose 85% of the time
    MAX_RISK_TOLERANCE = 0.18
    
    def __init__(self, game_board: board.Board, time_left: Callable):
        """
        Initialize the ScubaSteve V4 agent.
        
        Args:
            game_board: Initial game board state
            time_left: Callable returning remaining time in seconds
        """
        self.map_size = game_board.game_map.MAP_SIZE
        
        # Initialize belief tracker for trapdoor probabilities
        self.belief = TrapdoorBelief(self.map_size)
        
        # Load learned weights from distillation
        self.weights = self._load_weights()
        
        # Track spawn position for death detection
        self.spawn_position: Optional[Tuple[int, int]] = None
        self.turn_count = 0
        
        # Cache for visited safe cells
        self.visited_cells: Set[Tuple[int, int]] = set()
    
    def _load_weights(self) -> Dict[str, float]:
        """
        Load learned weights from JSON file.
        
        Returns:
            Dictionary of feature weights
        """
        weights_path = os.path.join(os.path.dirname(__file__), "learned_weights.json")
        
        try:
            with open(weights_path, "r") as f:
                data = json.load(f)
                # Remove metadata if present
                if "metadata" in data:
                    del data["metadata"]
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to default weights
            return {
                "egg_diff": 10.0,
                "mobility": 0.5,
                "corner_proximity": 0.2,
                "turd_diff": 0.5,
                "trapdoor_risk": -15.0,
                "turd_chokepoint_mobility_reduction": 5.0,
                "turd_adjacent_trapdoor": 3.0,
                "corner_control": 0.5,
                "chokepoint_potential": 2.0,
                "enemy_mobility_penalty": -0.3,
                "safe_zone_bonus": 0.1,
                "max_risk_tolerance": 0.18,
            }
    
    def play(
        self,
        game_board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """
        Select the best move based on current game state.
        
        Args:
            game_board: Current game board state
            sensor_data: List of (heard, felt) tuples for each trapdoor
            time_left: Callable returning remaining time
            
        Returns:
            Tuple of (Direction, MoveType) representing the chosen move
        """
        self.turn_count += 1
        current_pos = game_board.chicken_player.get_location()
        
        # Initialize spawn position on first turn
        if self.spawn_position is None:
            self.spawn_position = game_board.chicken_player.get_spawn()
        
        # Sync found trapdoors from board state (Death Memory)
        for trap in game_board.found_trapdoors:
            self.belief.add_found_trapdoor(trap)
        
        # Check for death event and update Death Memory
        self.belief.update_death_memory(
            current_pos, 
            self.spawn_position,
            None  # Target will be set when we make our move
        )
        
        # Update beliefs from sensor data
        self.belief.update_from_sensors(current_pos, sensor_data)
        
        # Mark current position as safe (we're here and alive)
        if current_pos not in self.visited_cells:
            self.visited_cells.add(current_pos)
            self.belief.mark_safe(current_pos)
        
        # Get valid moves
        valid_moves = game_board.get_valid_moves()
        
        if not valid_moves:
            # Fallback: return any move (shouldn't happen in normal gameplay)
            return (Direction.UP, MoveType.PLAIN)
        
        # Evaluate all moves and select the best
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            score = self._evaluate_move(game_board, move)
            if score > best_score:
                best_score = score
                best_move = move
        
        # Record the target position for death detection on next turn
        if best_move:
            direction, _ = best_move
            target_pos = loc_after_direction(current_pos, direction)
            self.belief.set_move_target(target_pos)
        
        return best_move if best_move else valid_moves[0]
    
    def _evaluate_move(
        self,
        game_board: board.Board,
        move: Tuple[Direction, MoveType]
    ) -> float:
        """
        Evaluate a move using the linear evaluation model.
        
        Args:
            game_board: Current game board
            move: (Direction, MoveType) tuple to evaluate
            
        Returns:
            Evaluation score (higher is better)
        """
        direction, move_type = move
        
        # Forecast the resulting board state
        forecasted = game_board.forecast_move(direction, move_type, check_ok=False)
        if forecasted is None:
            return float('-inf')
        
        # Calculate target position
        current_pos = game_board.chicken_player.get_location()
        target_pos = loc_after_direction(current_pos, direction)
        
        # Feature extraction
        features = self._extract_features(game_board, forecasted, current_pos, target_pos, move_type)
        
        # Linear combination of features with learned weights
        score = 0.0
        
        # Core features
        score += self.weights.get("egg_diff", 10.0) * features["egg_diff"]
        score += self.weights.get("mobility", 0.5) * features["mobility"]
        score += self.weights.get("corner_proximity", 0.2) * features["corner_proximity"]
        score += self.weights.get("turd_diff", 0.5) * features["turd_diff"]
        score += self.weights.get("trapdoor_risk", -15.0) * features["trapdoor_risk"]
        
        # Turd Warfare features (chokepoint strategy)
        if move_type == MoveType.TURD:
            score += self.weights.get("turd_chokepoint_mobility_reduction", 5.0) * features["turd_chokepoint"]
            score += self.weights.get("turd_adjacent_trapdoor", 3.0) * features["turd_adjacent_trapdoor"]
        
        # Advanced features
        score += self.weights.get("corner_control", 0.5) * features["corner_control"]
        score += self.weights.get("chokepoint_potential", 2.0) * features["chokepoint_potential"]
        score += self.weights.get("enemy_mobility_penalty", -0.3) * features["enemy_mobility"]
        score += self.weights.get("safe_zone_bonus", 0.1) * features["safe_zone"]
        
        return score
    
    def _extract_features(
        self,
        current_board: board.Board,
        forecasted: board.Board,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        move_type: MoveType
    ) -> Dict[str, float]:
        """
        Extract features for move evaluation.
        
        Args:
            current_board: Board before move
            forecasted: Board after move
            current_pos: Current position
            target_pos: Target position after move
            move_type: Type of move being made
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Egg difference (our eggs - enemy eggs)
        our_eggs = forecasted.chicken_player.get_eggs_laid()
        enemy_eggs = forecasted.chicken_enemy.get_eggs_laid()
        features["egg_diff"] = our_eggs - enemy_eggs
        
        # Mobility (number of valid moves after this move)
        features["mobility"] = len(forecasted.get_valid_moves())
        
        # Corner proximity (distance to nearest corner - lower is better)
        features["corner_proximity"] = self._calculate_corner_proximity(target_pos)
        
        # Turd difference (our turds placed - enemy turds near us)
        our_turds = 5 - forecasted.chicken_player.get_turds_left()
        enemy_turds_nearby = self._count_nearby_enemy_turds(forecasted, target_pos)
        features["turd_diff"] = our_turds - enemy_turds_nearby
        
        # Trapdoor risk at target position
        risk = self.belief.get_risk_at(target_pos)
        
        # Check if exceeds MAX_RISK_TOLERANCE (Death Line)
        max_risk = self.weights.get("max_risk_tolerance", self.MAX_RISK_TOLERANCE)
        if risk > max_risk:
            features["trapdoor_risk"] = risk * 2  # Double penalty for exceeding threshold
        else:
            features["trapdoor_risk"] = risk
        
        # CONFIRMED TRAPDOOR: Absolute avoidance
        if self.belief.is_confirmed_trapdoor(target_pos):
            features["trapdoor_risk"] = 100.0  # Massive penalty
        
        # Turd Chokepoint Score (for TURD moves)
        features["turd_chokepoint"] = 0.0
        features["turd_adjacent_trapdoor"] = 0.0
        
        if move_type == MoveType.TURD:
            # Calculate enemy mobility reduction
            enemy_mobility_before = len(current_board.get_valid_moves(enemy=True))
            enemy_mobility_after = len(forecasted.get_valid_moves(enemy=True))
            mobility_reduction = enemy_mobility_before - enemy_mobility_after
            
            # +5.0 bonus for reducing enemy mobility by >2
            if mobility_reduction > 2:
                features["turd_chokepoint"] = 1.0
            elif mobility_reduction > 1:
                features["turd_chokepoint"] = 0.5
            
            # +3.0 bonus for placing turd adjacent to known trapdoor
            if self._is_adjacent_to_trapdoor(current_pos):
                features["turd_adjacent_trapdoor"] = 1.0
        
        # Corner control (can we lay egg in corner soon?)
        features["corner_control"] = self._calculate_corner_control(forecasted, target_pos)
        
        # Chokepoint potential (are we creating strategic chokepoints?)
        features["chokepoint_potential"] = self._calculate_chokepoint_potential(forecasted, target_pos)
        
        # Enemy mobility (lower is better for us)
        features["enemy_mobility"] = len(forecasted.get_valid_moves(enemy=True))
        
        # Safe zone bonus (being in areas with low trapdoor probability)
        features["safe_zone"] = 1.0 if risk < 0.05 else 0.0
        
        return features
    
    def _calculate_corner_proximity(self, pos: Tuple[int, int]) -> float:
        """
        Calculate inverse distance to nearest corner.
        Higher value = closer to corner.
        
        Args:
            pos: Position to evaluate
            
        Returns:
            Corner proximity score
        """
        x, y = pos
        max_coord = self.map_size - 1
        
        corners = [(0, 0), (0, max_coord), (max_coord, 0), (max_coord, max_coord)]
        
        min_dist = float('inf')
        for cx, cy in corners:
            dist = abs(x - cx) + abs(y - cy)  # Manhattan distance
            min_dist = min(min_dist, dist)
        
        # Invert: closer = higher score
        return max(0, (self.map_size * 2 - min_dist) / (self.map_size * 2))
    
    def _count_nearby_enemy_turds(
        self,
        game_board: board.Board,
        pos: Tuple[int, int]
    ) -> int:
        """
        Count enemy turds within blocking range of a position.
        
        Args:
            game_board: Current board state
            pos: Position to check
            
        Returns:
            Number of nearby enemy turds
        """
        count = 0
        for direction in Direction:
            adj = loc_after_direction(pos, direction)
            if adj in game_board.turds_enemy:
                count += 1
        if pos in game_board.turds_enemy:
            count += 1
        return count
    
    def _is_adjacent_to_trapdoor(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is adjacent to a confirmed trapdoor.
        
        Args:
            pos: Position to check
            
        Returns:
            True if adjacent to any confirmed trapdoor
        """
        confirmed = self.belief.get_confirmed_trapdoors()
        
        for trap in confirmed:
            tx, ty = trap
            px, py = pos
            # Adjacent = Manhattan distance of 1
            if abs(tx - px) + abs(ty - py) == 1:
                return True
        
        return False
    
    def _calculate_corner_control(
        self,
        game_board: board.Board,
        pos: Tuple[int, int]
    ) -> float:
        """
        Evaluate our control over corners for egg laying.
        
        Args:
            game_board: Current board state
            pos: Current position
            
        Returns:
            Corner control score
        """
        max_coord = self.map_size - 1
        corners = [(0, 0), (0, max_coord), (max_coord, 0), (max_coord, max_coord)]
        
        score = 0.0
        even_chicken = game_board.chicken_player.even_chicken
        
        for cx, cy in corners:
            # Can we lay an egg at this corner?
            if (cx + cy) % 2 == even_chicken:
                # Check if corner is blocked
                if (cx, cy) not in game_board.eggs_enemy and (cx, cy) not in game_board.turds_enemy:
                    dist = abs(pos[0] - cx) + abs(pos[1] - cy)
                    if dist <= 3:  # Close to corner
                        score += 1.0 / (dist + 1)
        
        return score
    
    def _calculate_chokepoint_potential(
        self,
        game_board: board.Board,
        pos: Tuple[int, int]
    ) -> float:
        """
        Evaluate the chokepoint potential of a position.
        
        Args:
            game_board: Current board state
            pos: Position to evaluate
            
        Returns:
            Chokepoint potential score
        """
        enemy_pos = game_board.chicken_enemy.get_location()
        
        # Check if we're between enemy and corner
        max_coord = self.map_size - 1
        corners = [(0, 0), (0, max_coord), (max_coord, 0), (max_coord, max_coord)]
        
        score = 0.0
        for cx, cy in corners:
            # Distance from enemy to corner
            enemy_to_corner = abs(enemy_pos[0] - cx) + abs(enemy_pos[1] - cy)
            # Distance from us to corner
            us_to_corner = abs(pos[0] - cx) + abs(pos[1] - cy)
            
            # We're creating a chokepoint if we're closer to the corner
            if us_to_corner < enemy_to_corner:
                score += 0.5
        
        return score
