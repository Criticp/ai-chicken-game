"""
Component 1: The Trapdoor Belief Engine (Bayesian Inference)

Maintains probability distributions over the board for trapdoor locations.
Uses sensor data (heard, felt) to perform Bayesian updates.
Outputs a "Risk Map" matrix where every square has P(Trapdoor) between 0.0 and 1.0.
"""

from typing import List, Tuple, Set
import numpy as np


class TrapdoorTracker:
    """
    Bayesian inference engine for trapdoor location tracking.
    Maintains separate probability distributions for even and odd trapdoors.
    """

    def __init__(self, map_size: int = 8):
        """
        Initialize tracker with prior probabilities weighted by distance from edge.
        Center squares are more likely to contain trapdoors.
        """
        self.map_size = map_size

        # Two 8x8 arrays for the two trapdoors
        self.prob_even = np.ones((map_size, map_size), dtype=np.float32)
        self.prob_odd = np.ones((map_size, map_size), dtype=np.float32)

        # Initialize with edge-weighted priors (center more likely)
        self._initialize_edge_weighted_priors()

        # Track discovered information
        self.safe_cells: Set[Tuple[int, int]] = set()
        self.found_trapdoors: Set[Tuple[int, int]] = set()
        self.death_trapdoors: Set[Tuple[int, int]] = set()  # Squares where we died (100% certain)

        # For identifying which trapdoor is which
        self.trapdoor_even_location = None
        self.trapdoor_odd_location = None

        # Pre-computed likelihood lookup table (64x64) for sensor probabilities
        # Directive 1.2: Pre-compute for all (my_pos, trap_pos) combinations
        self.likelihood_table = self._build_likelihood_table()

    def _initialize_edge_weighted_priors(self):
        """
        Initialize prior probabilities using EXACT spawn weights from trapdoor_manager.py.

        Source Truth (trapdoor_manager.py lines 62-64):
        - Outer 2 rings: weight 0.0
        - Middle ring (2:6, 2:6): weight 1.0
        - Center 2x2 (3:5, 3:5): weight 2.0

        Then apply parity constraints (even squares for even trap, odd for odd).
        """
        dim = self.map_size

        # Replicate EXACT source code from trapdoor_manager.py
        unnormalized = np.zeros((dim, dim), dtype=np.float32)
        unnormalized[2:dim-2, 2:dim-2] = 1.0  # Middle ring
        unnormalized[3:dim-3, 3:dim-3] = 2.0  # Center 2x2

        # Apply to both even and odd
        self.prob_even = unnormalized.copy()
        self.prob_odd = unnormalized.copy()

        # Zero out impossible squares (parity constraint)
        for y in range(dim):
            for x in range(dim):
                is_even_square = (x + y) % 2 == 0
                if is_even_square:
                    self.prob_odd[y, x] = 0.0  # Odd trap can't be on even square
                else:
                    self.prob_even[y, x] = 0.0  # Even trap can't be on odd square

        # Normalize to proper probability distributions
        self._normalize()

    def _build_likelihood_table(self) -> np.ndarray:
        """
        Pre-compute 64x64 likelihood lookup table for sensor probabilities.

        Directive 1.2: Exact sensor model from game_map.py
        Table[my_pos_idx][trap_pos_idx] = P(heard, felt | trap at trap_pos)

        Returns 8x8x8x8x2x2 array: [my_y][my_x][trap_y][trap_x][heard][felt]
        """
        # Store probabilities for all combinations
        # Shape: (8, 8, 8, 8, 2, 2) - last two dims are [heard][felt] boolean outcomes
        table = np.zeros((8, 8, 8, 8, 2, 2), dtype=np.float32)

        for my_y in range(8):
            for my_x in range(8):
                for trap_y in range(8):
                    for trap_x in range(8):
                        # Calculate distance (Manhattan components)
                        delta_x = abs(my_x - trap_x)
                        delta_y = abs(my_y - trap_y)

                        # Exact probabilities from game_map.py
                        p_hear = self._prob_hear(delta_x, delta_y)
                        p_feel = self._prob_feel(delta_x, delta_y)

                        # P(heard=True, felt=True)
                        table[my_y, my_x, trap_y, trap_x, 1, 1] = p_hear * p_feel

                        # P(heard=True, felt=False)
                        table[my_y, my_x, trap_y, trap_x, 1, 0] = p_hear * (1 - p_feel)

                        # P(heard=False, felt=True)
                        table[my_y, my_x, trap_y, trap_x, 0, 1] = (1 - p_hear) * p_feel

                        # P(heard=False, felt=False) - CRUCIAL for negative information
                        table[my_y, my_x, trap_y, trap_x, 0, 0] = (1 - p_hear) * (1 - p_feel)

        return table

    def _prob_hear(self, delta_x: int, delta_y: int) -> float:
        """Exact copy of prob_hear from game_map.py"""
        if delta_x > 2 or delta_y > 2:
            return 0.0
        if delta_x == 2 and delta_y == 2:
            return 0.0
        if delta_x == 2 or delta_y == 2:
            return 0.1
        if delta_x == 1 and delta_y == 1:
            return 0.25
        if delta_x == 1 or delta_y == 1:
            return 0.5
        return 0.0

    def _prob_feel(self, delta_x: int, delta_y: int) -> float:
        """Exact copy of prob_feel from game_map.py"""
        if delta_x > 1 or delta_y > 1:
            return 0.0
        if delta_x == 1 and delta_y == 1:
            return 0.15
        if delta_x == 1 or delta_y == 1:
            return 0.3
        return 0.0

    def update_from_sensors(self, current_loc: Tuple[int, int],
                           sensor_data: List[Tuple[bool, bool]]):
        """
        Perform Bayesian update using sensor data.

        Args:
            current_loc: (x, y) current chicken location
            sensor_data: [(heard_even, felt_even), (heard_odd, felt_odd)]
        """
        heard_even, felt_even = sensor_data[0]
        heard_odd, felt_odd = sensor_data[1]

        # Update even trapdoor probability
        self._bayesian_update(self.prob_even, current_loc, heard_even, felt_even)

        # Update odd trapdoor probability
        self._bayesian_update(self.prob_odd, current_loc, heard_odd, felt_odd)

        # Mark current location as safe (we didn't die)
        self.mark_safe(current_loc)

    def _bayesian_update(self, prob_map: np.ndarray, current_loc: Tuple[int, int],
                        heard: bool, felt: bool):
        """
        Apply Bayes' rule using PRE-COMPUTED likelihood table.

        Directive 1.2: Posterior[x,y] = Prior[x,y] * P(sensors | Trap at x,y)

        Critical: Handles negative information automatically:
        - If sensors = (False, False), likelihood from table includes (1-p_hear)*(1-p_feel)
        - This drives neighbor probabilities down when we hear nothing
        """
        cx, cy = current_loc

        # Convert boolean sensors to indices (0 or 1)
        heard_idx = 1 if heard else 0
        felt_idx = 1 if felt else 0

        for y in range(self.map_size):
            for x in range(self.map_size):
                loc = (x, y)

                # Known information overrides probabilistic updates
                if loc in self.safe_cells:
                    prob_map[y, x] = 0.0
                    continue
                if loc in self.found_trapdoors or loc in self.death_trapdoors:
                    prob_map[y, x] = 1.0
                    continue

                # Lookup pre-computed likelihood from table
                # P(heard, felt | Trap at (x,y))
                likelihood = self.likelihood_table[cy, cx, y, x, heard_idx, felt_idx]

                # Bayesian update: Posterior ∝ Likelihood × Prior
                prob_map[y, x] *= likelihood

        # Normalize to maintain probability distribution
        self._normalize_array(prob_map)

    def _prob_hear(self, delta_x: int, delta_y: int) -> float:
        """
        Probability of hearing trapdoor based on Manhattan distance components.
        From tournament rules probability table.
        """
        if delta_x > 2 or delta_y > 2:
            return 0.0
        if delta_x == 2 and delta_y == 2:
            return 0.0
        if delta_x == 2 or delta_y == 2:
            return 0.1
        if delta_x == 1 and delta_y == 1:
            return 0.25
        if delta_x == 1 or delta_y == 1:
            return 0.5
        return 0.0  # Same cell

    def _prob_feel(self, delta_x: int, delta_y: int) -> float:
        """
        Probability of feeling trapdoor based on Manhattan distance components.
        From tournament rules probability table.
        """
        if delta_x > 1 or delta_y > 1:
            return 0.0
        if delta_x == 1 and delta_y == 1:
            return 0.15
        if delta_x == 1 or delta_y == 1:
            return 0.3
        return 0.0  # Same cell

    def _normalize_array(self, arr: np.ndarray):
        """Normalize array to sum to 1.0"""
        total = np.sum(arr)
        if total > 1e-10:
            arr[:] = arr / total
        else:
            # If all zeros, reset to uniform
            arr[:] = 1.0 / (self.map_size * self.map_size)

    def _normalize(self):
        """Normalize both probability maps"""
        self._normalize_array(self.prob_even)
        self._normalize_array(self.prob_odd)

    def mark_safe(self, loc: Tuple[int, int]):
        """Mark a cell as safe (visited without dying)"""
        if loc not in self.safe_cells and loc not in self.found_trapdoors:
            self.safe_cells.add(loc)
            x, y = loc
            self.prob_even[y, x] = 0.0
            self.prob_odd[y, x] = 0.0
            self._normalize()

    def mark_found_trapdoor(self, loc: Tuple[int, int], is_even: bool):
        """
        Mark that a trapdoor was found at location.

        Args:
            loc: (x, y) location
            is_even: True if this is the even trapdoor
        """
        self.found_trapdoors.add(loc)
        x, y = loc

        if is_even:
            # This is the even trapdoor
            self.prob_even = np.zeros((self.map_size, self.map_size), dtype=np.float32)
            self.prob_even[y, x] = 1.0
            self.trapdoor_even_location = loc
        else:
            # This is the odd trapdoor
            self.prob_odd = np.zeros((self.map_size, self.map_size), dtype=np.float32)
            self.prob_odd[y, x] = 1.0
            self.trapdoor_odd_location = loc

        # If both found, mark rest of board as safe
        if len(self.found_trapdoors) >= 2:
            self._mark_all_remaining_safe()

    def mark_death_location(self, loc: Tuple[int, int]):
        """
        Mark a location where we died (teleported from).
        This is 100% certain trapdoor - LOCKED, no further updates.
        """
        self.death_trapdoors.add(loc)
        self.found_trapdoors.add(loc)

        x, y = loc
        # We don't know if it's even or odd, but mark both as possible
        # The risk map will show 100% risk regardless
        self.prob_even[y, x] = 1.0
        self.prob_odd[y, x] = 1.0

        print(f"[TrapdoorTracker] ☠️ DEATH CONFIRMED at {loc} - LOCKED AT 100% RISK")

    def mark_safe_square(self, loc: Tuple[int, int]):
        """
        Mark a location as 0% trapdoor (Zero Knowledge).
        Called when we step on a square and DON'T die.
        """
        x, y = loc
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return

        if loc in self.death_trapdoors or loc in self.found_trapdoors:
            return  # Already confirmed as trap, don't override

        self.safe_cells.add(loc)

        # Set probability to 0 for the appropriate parity
        is_even = (x + y) % 2 == 0

        if is_even:
            self.prob_even[y, x] = 0.0
        else:
            self.prob_odd[y, x] = 0.0

    def _mark_all_remaining_safe(self):
        """Once both trapdoors found, mark all other cells as safe"""
        self.prob_even = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.prob_odd = np.zeros((self.map_size, self.map_size), dtype=np.float32)

        # Restore found trapdoor locations
        for loc in self.found_trapdoors:
            x, y = loc
            # Don't know which is which if we have death trapdoors
            self.prob_even[y, x] = 0.5
            self.prob_odd[y, x] = 0.5

    def get_risk_map(self) -> np.ndarray:
        """
        Generate the Risk Map: an 8x8 matrix with P(Trapdoor) for each square.

        Returns:
            8x8 numpy array with values between 0.0 and 1.0
        """
        risk_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)

        for y in range(self.map_size):
            for x in range(self.map_size):
                loc = (x, y)

                # Death trapdoors: 100% risk (absolute certainty)
                if loc in self.death_trapdoors:
                    risk_map[y, x] = 1.0
                # Found trapdoors: 100% risk
                elif loc in self.found_trapdoors:
                    risk_map[y, x] = 1.0
                # Safe cells: 0% risk
                elif loc in self.safe_cells:
                    risk_map[y, x] = 0.0
                # Probabilistic estimate: P(at least one trapdoor here)
                else:
                    p_even = float(self.prob_even[y, x])
                    p_odd = float(self.prob_odd[y, x])
                    # P(A or B) = P(A) + P(B) - P(A and B)
                    # Assuming independence: P(A and B) = P(A) * P(B)
                    risk_map[y, x] = p_even + p_odd - (p_even * p_odd)

        return risk_map

    def get_trapdoor_risk(self, loc: Tuple[int, int]) -> float:
        """
        Get the trapdoor risk for a specific location.

        Returns:
            Float between 0.0 (safe) and 1.0 (certain trapdoor)
        """
        x, y = loc

        # Death trapdoors: absolute certainty
        if loc in self.death_trapdoors:
            return 1.0

        # Found trapdoors: confirmed
        if loc in self.found_trapdoors:
            return 1.0

        # Safe cells: no risk
        if loc in self.safe_cells:
            return 0.0

        # Probabilistic estimate
        p_even = float(self.prob_even[y, x])
        p_odd = float(self.prob_odd[y, x])

        # Combined probability: at least one trapdoor here
        combined_risk = p_even + p_odd - (p_even * p_odd)

        return float(combined_risk)

