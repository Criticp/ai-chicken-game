"""
Trapdoor Belief Management for ScubaSteve V4 - Predator Upgrade

Implements:
1. Bayesian probability tracking for trapdoor locations
2. Persistent "Death Memory" - confirmed trapdoors become permanent No-Go Zones
3. Event-driven updates based on sensor data (hear, feel)
"""

from typing import Dict, List, Set, Tuple, Optional
from game.game_map import prob_feel, prob_hear


class TrapdoorBelief:
    """
    Tracks beliefs about trapdoor locations using Bayesian inference.
    
    Key Features:
    - Maintains probability distribution over all possible trapdoor locations
    - Implements "Death Memory": confirmed trapdoors have probability 1.0 permanently
    - Updates beliefs based on sensor data (hear, feel signals)
    """
    
    def __init__(self, map_size: int = 8):
        """
        Initialize belief tracker for trapdoor probabilities.
        
        Args:
            map_size: Size of the game board (default 8x8)
        """
        self.map_size = map_size
        
        # Probability grid for each trapdoor (even and odd parity)
        # Even trapdoor can only be on cells where (x + y) % 2 == 0
        # Odd trapdoor can only be on cells where (x + y) % 2 == 1
        self.even_probs: Dict[Tuple[int, int], float] = {}
        self.odd_probs: Dict[Tuple[int, int], float] = {}
        
        # DEATH MEMORY: Confirmed trapdoor locations (permanent No-Go Zones)
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()
        
        # Track previous position to detect death events
        self.prev_position: Optional[Tuple[int, int]] = None
        self.prev_target_position: Optional[Tuple[int, int]] = None
        
        # Initialize uniform prior probabilities
        self._initialize_priors()
    
    def _initialize_priors(self) -> None:
        """Initialize uniform prior probabilities for trapdoor locations."""
        # Trapdoors spawn in the center region (2-5 range for 8x8 board)
        center_min = 2
        center_max = self.map_size - 2  # 6 for 8x8
        
        even_cells = []
        odd_cells = []
        
        for x in range(center_min, center_max):
            for y in range(center_min, center_max):
                if (x + y) % 2 == 0:
                    even_cells.append((x, y))
                else:
                    odd_cells.append((x, y))
        
        # Uniform prior
        even_prior = 1.0 / len(even_cells) if even_cells else 0.0
        odd_prior = 1.0 / len(odd_cells) if odd_cells else 0.0
        
        for cell in even_cells:
            self.even_probs[cell] = even_prior
        
        for cell in odd_cells:
            self.odd_probs[cell] = odd_prior
    
    def update_death_memory(
        self,
        current_position: Tuple[int, int],
        spawn_position: Tuple[int, int],
        attempted_target: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Check for and record trapdoor deaths (Death Memory).
        
        Logic: if prev_turn_action == MOVE and current_pos == START_POS:
               target_square_probability = 1.0 (PERMANENT)
        
        Args:
            current_position: Current chicken position
            spawn_position: Starting spawn position
            attempted_target: The position we tried to move to
            
        Returns:
            True if a death was detected and recorded
        """
        death_detected = False
        
        # Check if we died: we moved but ended up at spawn
        if self.prev_target_position is not None:
            if current_position == spawn_position and self.prev_position != spawn_position:
                # DEATH DETECTED - We tried to move but got sent back to spawn
                death_loc = self.prev_target_position
                
                # PERMANENT NO-GO ZONE: Mark as confirmed trapdoor
                self.confirmed_trapdoors.add(death_loc)
                
                # Set probability to 1.0 for the correct parity grid
                if (death_loc[0] + death_loc[1]) % 2 == 0:
                    self.even_probs[death_loc] = 1.0
                    # Normalize other cells to 0 for this trapdoor
                    for cell in self.even_probs:
                        if cell != death_loc:
                            self.even_probs[cell] = 0.0
                else:
                    self.odd_probs[death_loc] = 1.0
                    # Normalize other cells to 0 for this trapdoor
                    for cell in self.odd_probs:
                        if cell != death_loc:
                            self.odd_probs[cell] = 0.0
                
                death_detected = True
        
        # Update tracking for next turn
        self.prev_position = current_position
        if attempted_target is not None:
            self.prev_target_position = attempted_target
        
        return death_detected
    
    def set_move_target(self, target_position: Tuple[int, int]) -> None:
        """Record the target position for the current move."""
        self.prev_target_position = target_position
    
    def update_from_sensors(
        self,
        current_position: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]]
    ) -> None:
        """
        Update trapdoor beliefs based on sensor readings.
        
        Args:
            current_position: Current chicken position (x, y)
            sensor_data: List of (did_hear, did_feel) for each trapdoor
        """
        # sensor_data[0] is for even trapdoor, sensor_data[1] is for odd trapdoor
        even_hear, even_feel = sensor_data[0]
        odd_hear, odd_feel = sensor_data[1]
        
        # Update even trapdoor probabilities (skip if confirmed)
        if not any((cell[0] + cell[1]) % 2 == 0 for cell in self.confirmed_trapdoors):
            self._update_grid_from_sensor(
                self.even_probs, current_position, even_hear, even_feel
            )
        
        # Update odd trapdoor probabilities (skip if confirmed)
        if not any((cell[0] + cell[1]) % 2 == 1 for cell in self.confirmed_trapdoors):
            self._update_grid_from_sensor(
                self.odd_probs, current_position, odd_hear, odd_feel
            )
    
    def _update_grid_from_sensor(
        self,
        probs: Dict[Tuple[int, int], float],
        position: Tuple[int, int],
        did_hear: bool,
        did_feel: bool
    ) -> None:
        """
        Apply Bayesian update to probability grid based on sensor data.
        
        Args:
            probs: Probability dictionary to update
            position: Current position
            did_hear: Whether we heard the trapdoor
            did_feel: Whether we felt the trapdoor
        """
        px, py = position
        total = 0.0
        
        for cell, prob in probs.items():
            if prob <= 0:
                continue
                
            cx, cy = cell
            delta_x = abs(cx - px)
            delta_y = abs(cy - py)
            
            # Calculate likelihood of sensor readings given trapdoor at cell
            hear_p = prob_hear(delta_x, delta_y)
            feel_p = prob_feel(delta_x, delta_y)
            
            if did_hear:
                hear_likelihood = hear_p
            else:
                hear_likelihood = 1.0 - hear_p
            
            if did_feel:
                feel_likelihood = feel_p
            else:
                feel_likelihood = 1.0 - feel_p
            
            # Combined likelihood
            likelihood = hear_likelihood * feel_likelihood
            
            # Bayesian update: posterior = prior * likelihood
            probs[cell] = prob * likelihood
            total += probs[cell]
        
        # Normalize probabilities
        if total > 0:
            for cell in probs:
                probs[cell] /= total
    
    def mark_safe(self, position: Tuple[int, int]) -> None:
        """
        Mark a position as safe (visited without dying).
        
        Args:
            position: Position that was safely visited
        """
        # If we visited this cell and didn't die, it's not a trapdoor
        x, y = position
        parity = (x + y) % 2
        
        if parity == 0 and position in self.even_probs:
            self.even_probs[position] = 0.0
            self._normalize_grid(self.even_probs)
        elif parity == 1 and position in self.odd_probs:
            self.odd_probs[position] = 0.0
            self._normalize_grid(self.odd_probs)
    
    def _normalize_grid(self, probs: Dict[Tuple[int, int], float]) -> None:
        """Normalize probability grid to sum to 1."""
        total = sum(probs.values())
        if total > 0:
            for cell in probs:
                probs[cell] /= total
    
    def get_risk_at(self, position: Tuple[int, int]) -> float:
        """
        Get the risk (trapdoor probability) at a specific position.
        
        Args:
            position: Position to evaluate
            
        Returns:
            Combined probability of trapdoor at this position
        """
        # DEATH MEMORY: Confirmed trapdoors have risk 1.0
        if position in self.confirmed_trapdoors:
            return 1.0
        
        x, y = position
        parity = (x + y) % 2
        
        if parity == 0:
            return self.even_probs.get(position, 0.0)
        else:
            return self.odd_probs.get(position, 0.0)
    
    def is_confirmed_trapdoor(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is a confirmed trapdoor (Death Memory).
        
        Args:
            position: Position to check
            
        Returns:
            True if this is a confirmed No-Go Zone
        """
        return position in self.confirmed_trapdoors
    
    def get_confirmed_trapdoors(self) -> Set[Tuple[int, int]]:
        """
        Get all confirmed trapdoor locations.
        
        Returns:
            Set of confirmed trapdoor positions
        """
        return self.confirmed_trapdoors.copy()
    
    def get_high_risk_cells(self, threshold: float = 0.2) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get all cells with risk above a threshold.
        
        Args:
            threshold: Risk threshold (default 0.2 = 20%)
            
        Returns:
            List of (position, risk) tuples sorted by risk descending
        """
        high_risk = []
        
        # Add confirmed trapdoors first
        for pos in self.confirmed_trapdoors:
            high_risk.append((pos, 1.0))
        
        # Check even grid
        for pos, prob in self.even_probs.items():
            if prob >= threshold and pos not in self.confirmed_trapdoors:
                high_risk.append((pos, prob))
        
        # Check odd grid
        for pos, prob in self.odd_probs.items():
            if prob >= threshold and pos not in self.confirmed_trapdoors:
                high_risk.append((pos, prob))
        
        return sorted(high_risk, key=lambda x: x[1], reverse=True)
    
    def add_found_trapdoor(self, position: Tuple[int, int]) -> None:
        """
        Add a trapdoor that was found (e.g., from board.found_trapdoors).
        
        Args:
            position: Known trapdoor position
        """
        self.confirmed_trapdoors.add(position)
        
        # Update probability grid
        x, y = position
        if (x + y) % 2 == 0:
            for cell in self.even_probs:
                self.even_probs[cell] = 1.0 if cell == position else 0.0
        else:
            for cell in self.odd_probs:
                self.odd_probs[cell] = 1.0 if cell == position else 0.0
