"""
Belief State Manager for Scuba Steve
Bayesian inference for trapdoor detection using sensor data
"""

from typing import List, Tuple
import numpy as np


class TrapdoorBelief:
    """
    Manages belief probabilities for trapdoor locations using Bayesian updates
    """

    def __init__(self, map_size=8):
        self.map_size = map_size
        # Initialize uniform prior for both trapdoors
        # Each cell has equal initial probability
        self.belief_a = np.ones((map_size, map_size)) / (map_size * map_size)
        self.belief_b = np.ones((map_size, map_size)) / (map_size * map_size)

        # Track visited cells (no trapdoor there)
        self.safe_cells = set()
        self.found_trapdoors = set()

        # PREDATOR UPGRADE: Persistent trauma - squares where we died
        # Once we die on a square, it's PERMANENTLY marked as 100% deadly
        self.death_trapdoors = set()

    def update_safe_cell(self, loc: Tuple[int, int]):
        """Mark a cell as safe (visited without triggering trapdoor)"""
        if loc not in self.safe_cells:
            self.safe_cells.add(loc)
            x, y = loc
            # Set belief to 0 for safe cells
            self.belief_a[y, x] = 0.0
            self.belief_b[y, x] = 0.0
            # Renormalize
            self._normalize()

    def update_found_trapdoor(self, loc: Tuple[int, int], trapdoor_id: int):
        """Mark that we found a trapdoor at location"""
        self.found_trapdoors.add(loc)
        x, y = loc
        if trapdoor_id == 0:  # Trapdoor A
            self.belief_a = np.zeros((self.map_size, self.map_size))
            self.belief_a[y, x] = 1.0
        else:  # Trapdoor B
            self.belief_b = np.zeros((self.map_size, self.map_size))
            self.belief_b[y, x] = 1.0

    def update_death_trapdoor(self, loc: Tuple[int, int]):
        """
        Legacy method - redirects to lock_death_trapdoor
        """
        self.lock_death_trapdoor(loc)

    def lock_death_trapdoor(self, location: Tuple[int, int]):
        """
        PHASE 2 FIX: Lock a square as 100% trapdoor (PERMANENT)
        This square caused actual death (teleport detected)
        NO Bayesian updates allowed - this is absolute truth
        """
        self.death_trapdoors.add(location)
        self.found_trapdoors.add(location)

        # Set belief maps to certainty for this location
        x, y = location
        # Mark this specific location as 100% trapdoor
        self.belief_a[y, x] = 1.0  # Could be trapdoor A
        self.belief_b[y, x] = 1.0  # Could be trapdoor B

        # Check if we've found both trapdoors
        if len(self.found_trapdoors) >= 2:
            print(f"[Belief] Both trapdoors found! Marking rest of board as SAFE")
            self._mark_remaining_safe()

    def _mark_remaining_safe(self):
        """
        PHASE 2 FIX: Once 2 trapdoors found, all other squares are safe
        Set their probability to 0.0 (no more trapdoors exist)
        """
        # Zero out all probabilities except the found trapdoors
        self.belief_a = np.zeros((self.map_size, self.map_size))
        self.belief_b = np.zeros((self.map_size, self.map_size))

        # Restore the found trapdoor locations
        for loc in self.found_trapdoors:
            x, y = loc
            self.belief_a[y, x] = 0.5  # Each could be either trapdoor
            self.belief_b[y, x] = 0.5

    def update_from_sensors(self, current_loc: Tuple[int, int],
                           sensor_data: List[Tuple[bool, bool]]):
        """
        Update beliefs based on sensor readings
        sensor_data[0] = (heard_a, felt_a)
        sensor_data[1] = (heard_b, felt_b)
        """
        heard_a, felt_a = sensor_data[0]
        heard_b, felt_b = sensor_data[1]

        # Update for trapdoor A
        self._bayesian_update(self.belief_a, current_loc, heard_a, felt_a)

        # Update for trapdoor B
        self._bayesian_update(self.belief_b, current_loc, heard_b, felt_b)

        # Mark current location as safe
        self.update_safe_cell(current_loc)

    def _bayesian_update(self, belief_map, current_loc: Tuple[int, int],
                        heard: bool, felt: bool):
        """Apply Bayesian update based on sensor data"""
        cx, cy = current_loc

        for y in range(self.map_size):
            for x in range(self.map_size):
                if (x, y) in self.safe_cells:
                    belief_map[y, x] = 0.0
                    continue

                # Calculate distance
                dx = abs(x - cx)
                dy = abs(y - cy)

                # Probability of hearing from this distance
                p_hear = self._prob_hear(dx, dy)
                # Probability of feeling from this distance
                p_feel = self._prob_feel(dx, dy)

                # Likelihood of observation given trapdoor at (x,y)
                if heard and felt:
                    likelihood = p_hear * p_feel
                elif heard and not felt:
                    likelihood = p_hear * (1 - p_feel)
                elif not heard and felt:
                    likelihood = (1 - p_hear) * p_feel
                else:  # not heard and not felt
                    likelihood = (1 - p_hear) * (1 - p_feel)

                # Bayesian update: P(trap|obs) ∝ P(obs|trap) * P(trap)
                belief_map[y, x] *= likelihood

        self._normalize_array(belief_map)

    def _prob_hear(self, delta_x: int, delta_y: int) -> float:
        """Probability of hearing based on distance"""
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
        """Probability of feeling based on distance"""
        if delta_x > 1 or delta_y > 1:
            return 0.0
        if delta_x == 1 and delta_y == 1:
            return 0.15
        if delta_x == 1 or delta_y == 1:
            return 0.3
        return 0.0

    def _normalize_array(self, arr):
        """Normalize array to sum to 1"""
        total = np.sum(arr)
        if total > 0:
            arr[:] = arr / total

    def _normalize(self):
        """Normalize both belief maps"""
        self._normalize_array(self.belief_a)
        self._normalize_array(self.belief_b)

    def get_trapdoor_risk(self, loc: Tuple[int, int]) -> float:
        """Get combined trapdoor risk for a location (0-1)

        PREDATOR UPGRADE: Death squares ALWAYS return 1.0 (100% risk)
        """
        # DIRECTIVE 1: Persistent trauma - never step on death squares
        if loc in self.death_trapdoors:
            return 1.0

        if loc in self.safe_cells:
            return 0.0
        if loc in self.found_trapdoors:
            return 1.0

        x, y = loc
        # Combined risk from both trapdoors
        risk_a = float(self.belief_a[y, x])
        risk_b = float(self.belief_b[y, x])

        # Probability that at least one trapdoor is here
        combined_risk = risk_a + risk_b - (risk_a * risk_b)
        return float(combined_risk)

    def get_most_likely_trapdoor_locations(self) -> List[Tuple[Tuple[int, int], float]]:
        """Return most likely locations for trapdoors"""
        locations = []

        # For trapdoor A
        max_prob_a = np.max(self.belief_a)
        if max_prob_a > 0:
            y_a, x_a = np.unravel_index(np.argmax(self.belief_a), self.belief_a.shape)
            locations.append(((x_a, y_a), max_prob_a, 'A'))

        # For trapdoor B
        max_prob_b = np.max(self.belief_b)
        if max_prob_b > 0:
            y_b, x_b = np.unravel_index(np.argmax(self.belief_b), self.belief_b.shape)
            locations.append(((x_b, y_b), max_prob_b, 'B'))

        return locations
