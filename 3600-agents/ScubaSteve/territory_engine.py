"""
TERRITORY ENGINE - Flood Fill Based Control Evaluation

Strategic Directive: "The Architect"
Replace simple mobility counting with territorial control via BFS flood fill.

Key Insight: 4 moves into open field >> 4 moves in trapped room
"""

from typing import Tuple, Set
from collections import deque
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board


class TerritoryEvaluator:
    """
    Evaluate board control using BFS flood fill.

    DEFENSE & ZONING SPECIALIST:
    Weaponized Geography - Treat probable traps as natural walls

    Replaces simple move counting with reachable area analysis.
    Fast: 8×8 BFS takes <0.5ms
    """

    def __init__(self, trapdoor_tracker=None):
        self.board_size = 8
        self.tracker = trapdoor_tracker  # For weaponized geography

    def evaluate_control(self, board: "game_board.Board") -> Tuple[int, int, float]:
        """
        Calculate territorial control score.

        WEAPONIZED GEOGRAPHY:
        Treats squares with risk > 10% as walls, creating natural barriers.

        Returns:
            (my_area, enemy_area, territory_score)

        Territory_Score = (Area_Self - Area_Enemy) × 10.0
        """
        # Get positions
        my_pos = board.chicken_player.get_location()
        enemy_pos = board.chicken_enemy.get_location()

        # Get walls (eggs and turds) - FIXED: Use board attributes, not chicken attributes
        my_eggs = set(board.eggs_player)
        my_turds = set(board.turds_player)
        enemy_eggs = set(board.eggs_enemy)
        enemy_turds = set(board.turds_enemy)

        # WEAPONIZED GEOGRAPHY: Add probable traps as walls
        trap_walls = self._get_trap_walls() if self.tracker else set()

        # Flood fill for player (can walk on own eggs)
        # Walls: Enemy assets + probable traps
        my_walls = enemy_eggs | enemy_turds | trap_walls
        my_area = self._flood_fill_area(my_pos, my_walls, board)

        # Flood fill for enemy (cannot pass player's eggs/turds)
        # Walls: My assets + probable traps
        enemy_walls = my_eggs | my_turds | trap_walls
        enemy_area = self._flood_fill_area(enemy_pos, enemy_walls, board)

        # Calculate territory score
        territory_score = (my_area - enemy_area) * 10.0

        return my_area, enemy_area, territory_score

    def _get_trap_walls(self) -> Set[Tuple[int, int]]:
        """
        Weaponized Geography: Get squares that should be treated as walls.

        Returns all squares with risk > WALL_THRESHOLD (10%).
        Strategic effect: Traps become terrain features that split the board.
        """
        trap_walls = set()
        if not self.tracker:
            return trap_walls

        for y in range(self.board_size):
            for x in range(self.board_size):
                loc = (x, y)
                if self.tracker.is_wall(loc):  # Risk > 10%
                    trap_walls.add(loc)

        return trap_walls

    def _flood_fill_area(self, start_pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        board: "game_board.Board") -> int:
        """
        BFS flood fill to count reachable squares.

        Args:
            start_pos: Starting position
            walls: Set of positions that block movement
            board: Board for trap checking

        Returns:
            Number of reachable squares
        """
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)

        while queue:
            x, y = queue.popleft()

            # Check all 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Bounds check
                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                    continue

                # Already visited
                if (nx, ny) in visited:
                    continue

                # Wall check
                if (nx, ny) in walls:
                    continue

                # Valid square - add to reachable area
                visited.add((nx, ny))
                queue.append((nx, ny))

        return len(visited)

    def evaluate_choke_point(self, board: "game_board.Board",
                           candidate_move: Tuple[int, int]) -> float:
        """
        Evaluate if placing an egg/turd at candidate_move creates a choke point.

        The "Branch and Block" behavior:
        If placing here reduces Enemy_Reachable from 40 → 10, massive bonus!

        Returns:
            Choke point bonus (0.0 if not a choke point)
        """
        enemy_pos = board.chicken_enemy.get_location()

        # Current enemy walls - FIXED: Use board attributes
        my_eggs = set(board.eggs_player)
        my_turds = set(board.turds_player)
        enemy_eggs = set(board.eggs_enemy)
        enemy_turds = set(board.turds_enemy)

        enemy_walls = my_eggs | my_turds

        # Enemy area before placing
        area_before = self._flood_fill_area(enemy_pos, enemy_walls, board)

        # Simulate placing egg/turd at candidate
        enemy_walls_after = enemy_walls | {candidate_move}
        area_after = self._flood_fill_area(enemy_pos, enemy_walls_after, board)

        # Calculate choke point value
        area_reduction = area_before - area_after

        # Massive bonus for creating choke points
        # If reducing enemy area by 30 squares → bonus = 300 points!
        choke_bonus = area_reduction * 10.0

        return choke_bonus


class TacticalSuicideEvaluator:
    """
    Dynamic trapdoor cost calculation.

    Strategic Directive: "The Eject Button"
    Trapdoor is a fast-travel that costs 4 eggs.
    Profitable if Moves_Saved > 4.
    """

    def __init__(self):
        self.base_trap_cost = -4.0  # Enemy gets +4 eggs
        self.avg_points_per_turn = 0.8  # Estimated egg gain per turn

    def calculate_dynamic_trap_penalty(self, board: "game_board.Board",
                                      current_pos: Tuple[int, int],
                                      trap_pos: Tuple[int, int]) -> float:
        """
        Calculate dynamic trapdoor penalty.

        Directive Formula:
        - Base_Cost = -4.0
        - Travel_Savings = (dist_to_spawn - 1) × Avg_Points_Per_Turn
        - If Travel_Savings > 4.0: Effective_Cost = Base_Cost + Travel_Savings

        Returns:
            Effective trap penalty (negative = bad, positive = good!)
        """
        # Get spawn position
        spawn_pos = self._get_spawn_position(board)

        # Calculate distance to spawn
        dist_to_spawn = abs(current_pos[0] - spawn_pos[0]) + abs(current_pos[1] - spawn_pos[1])

        # Calculate spawn potential (empty squares around spawn)
        spawn_potential = self._count_spawn_potential(board, spawn_pos)

        # Calculate travel savings
        # If I'm 10 squares away, I save (10-1) × 0.8 = 7.2 points by teleporting
        travel_savings = (dist_to_spawn - 1) * self.avg_points_per_turn

        # Spawn fertility bonus
        # If spawn area has 8 empty squares, it's very fertile
        fertility_bonus = spawn_potential * 0.5

        # Calculate effective cost
        effective_cost = self.base_trap_cost + travel_savings + fertility_bonus

        # Additional check: Am I out of ammo (turds)?
        if board.chicken_player.turds_left == 0:
            # No turds left, teleporting gives me 5 fresh turds
            effective_cost += 2.0

        return effective_cost

    def _get_spawn_position(self, board: "game_board.Board") -> Tuple[int, int]:
        """Get player's spawn position from board"""
        # Spawn is stored in chicken object
        # For now, use common spawns or extract from board
        # Typical spawns: (0, 2), (0, 3), (7, 2), (7, 3), etc.
        # We'll use a heuristic: closest corner-ish position

        # Check board for spawn info
        if hasattr(board.chicken_player, 'spawn'):
            return board.chicken_player.spawn

        # Fallback: estimate based on starting positions
        # Most spawns are on edges
        current = board.chicken_player.get_location()
        x, y = current

        # Assume spawn is on edge closest to current position
        if x < 4:
            spawn_x = 0
        else:
            spawn_x = 7

        spawn_y = y  # Usually same y or close

        return (spawn_x, spawn_y)

    def _count_spawn_potential(self, board: "game_board.Board",
                               spawn_pos: Tuple[int, int]) -> int:
        """
        Count empty squares in 3×3 area around spawn.

        Returns:
            Number of empty squares (0-9)
        """
        sx, sy = spawn_pos
        empty_count = 0

        # Check 3×3 area
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x, y = sx + dx, sy + dy

                # Bounds check
                if not (0 <= x < 8 and 0 <= y < 8):
                    continue

                pos = (x, y)

                # Check if empty (not in any eggs/turds) - FIXED: Use board attributes
                if (pos not in board.eggs_player and
                    pos not in board.turds_player and
                    pos not in board.eggs_enemy and
                    pos not in board.turds_enemy):
                    empty_count += 1

        return empty_count


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def evaluate_with_territory(board: "game_board.Board") -> dict:
    """
    Convenience function for quick territory evaluation.

    Returns:
        {
            'my_area': int,
            'enemy_area': int,
            'territory_score': float,
            'control_ratio': float
        }
    """
    evaluator = TerritoryEvaluator()
    my_area, enemy_area, territory_score = evaluator.evaluate_control(board)

    # Calculate control ratio (avoid division by zero)
    total_area = my_area + enemy_area
    if total_area > 0:
        control_ratio = my_area / total_area
    else:
        control_ratio = 0.5

    return {
        'my_area': my_area,
        'enemy_area': enemy_area,
        'territory_score': territory_score,
        'control_ratio': control_ratio
    }

