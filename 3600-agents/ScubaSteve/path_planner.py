"""
Path Planner - Intelligent Navigation

Routes to strategic targets while laying eggs opportunistically.
Uses A* pathfinding with exploration awareness and smart backtracking.

Features:
- A* pathfinding to separator wall targets
- Opportunistic egg placement during traversal
- Smart backtracking through unexplored areas
- Move sequence generation for lookahead
"""

from typing import Tuple, List, Optional, Set
from collections import deque
import heapq
import sys
import os

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Direction, MoveType
from game.board import manhattan_distance


class PathPlanner:
    """
    Intelligent pathfinding with opportunistic egg placement.

    Key features:
    - A* search to strategic targets
    - Opportunistic egg laying on first pass
    - Smart backtracking through fresh territory
    - Integration with exploration tracker
    """

    # Path preferences
    FRESH_CELL_BONUS = 5.0          # Prefer unexplored cells
    EGG_OPPORTUNITY_BONUS = 10.0    # Bonus for cells where we can lay eggs
    DIRECT_PATH_BONUS = 2.0         # Slight preference for direct routes

    # Thresholds
    MIN_PATH_LENGTH_FOR_EGGS = 3    # Lay eggs if path > 3 cells
    MAX_PATH_DEPTH = 15             # Limit A* search depth

    def __init__(self, exploration_tracker=None, trapdoor_tracker=None):
        """
        Initialize path planner.

        Args:
            exploration_tracker: ExplorationTracker instance
            trapdoor_tracker: TrapdoorTracker instance for risk assessment
        """
        self.exploration = exploration_tracker
        self.tracker = trapdoor_tracker

        # Cached paths
        self._cached_path: Optional[List[Tuple[int, int]]] = None
        self._cached_target: Optional[Tuple[int, int]] = None
        self._cache_valid = False

    def plan_route(self, start: Tuple[int, int], target: Tuple[int, int],
                   board: "game_board.Board",
                   exploration_tracker=None,
                   force_egg_on_route: bool = False) -> List[Tuple[int, int]]:
        """
        Plan optimal route from start to target.

        Uses A* pathfinding with heuristics:
        - Distance to goal (Manhattan)
        - Exploration freshness
        - Trapdoor risk
        - Egg laying opportunities

        Args:
            start: Starting position
            target: Target position
            board: Current board state
            exploration_tracker: Optional tracker (overrides self.exploration)

        Returns:
            List of positions forming path (including start and target)
        """
        # Use provided tracker or default
        tracker = exploration_tracker or self.exploration

        # Store force_egg flag for use in should_lay_egg_on_route
        self._force_egg_on_route = force_egg_on_route

        # Check cache
        if self._cache_valid and self._cached_target == target:
            if self._cached_path and self._cached_path[0] == start:
                return self._cached_path

        # A* search
        path = self._astar_search(start, target, board, tracker)

        # Cache result
        self._cached_path = path
        self._cached_target = target
        self._cache_valid = True

        return path

    def _astar_search(self, start: Tuple[int, int], goal: Tuple[int, int],
                     board: "game_board.Board",
                     exploration_tracker) -> List[Tuple[int, int]]:
        """
        A* pathfinding with exploration awareness.

        Returns:
            Path as list of positions, or empty list if no path
        """
        # Priority queue: (f_score, counter, position, path)
        counter = 0
        heap = [(0, counter, start, [start])]
        visited = set()

        # Get exploration heatmap for bonuses
        heatmap = {}
        if exploration_tracker:
            heatmap = exploration_tracker.get_heatmap(board)

        while heap:
            f_score, _, pos, path = heapq.heappop(heap)

            # Check depth limit
            if len(path) > self.MAX_PATH_DEPTH:
                continue

            # Goal check
            if pos == goal:
                return path

            # Skip if visited
            if pos in visited:
                continue
            visited.add(pos)

            # Expand neighbors
            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Bounds check
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue

                neighbor = (nx, ny)

                # Skip visited
                if neighbor in visited:
                    continue

                # Skip impassable (turds)
                if neighbor in board.turds_player or neighbor in board.turds_enemy:
                    continue

                # Calculate costs
                g_cost = len(path)  # Distance from start
                h_cost = manhattan_distance(neighbor, goal)  # Heuristic to goal

                # Exploration bonus (prefer fresh cells)
                exploration_bonus = heatmap.get(neighbor, 0) * 0.1

                # Risk penalty
                risk_penalty = 0
                if self.tracker:
                    risk = self.tracker.get_trapdoor_risk(neighbor)
                    risk_penalty = risk * 10

                # Egg opportunity bonus
                egg_bonus = 0
                if exploration_tracker and not exploration_tracker.has_egg(neighbor):
                    if neighbor not in board.eggs_player and neighbor not in board.eggs_enemy:
                        egg_bonus = 1.0  # Small bonus for egg opportunities

                # NEW: Heavy penalty for already-egg'd cells (avoid revisiting)
                already_egged_penalty = 0
                if neighbor in board.eggs_player:
                    already_egged_penalty = 500  # DEATH PENALTY to avoid backtracking (was 50)
                elif exploration_tracker and exploration_tracker.has_egg(neighbor):
                    already_egged_penalty = 300  # Severe penalty for tracker-known eggs (was 30)

                # NEW: Penalty for recently visited cells (reduce oscillation)
                recent_visit_penalty = 0
                if exploration_tracker and neighbor in exploration_tracker.visited_turn:
                    freshness = exploration_tracker.get_freshness_score(neighbor, board.turn_count)
                    if freshness < 0.3:  # Recently visited
                        recent_visit_penalty = 100  # Increased from 20

                # Total f-score
                f = (g_cost + h_cost - exploration_bonus + risk_penalty - egg_bonus +
                     already_egged_penalty + recent_visit_penalty)

                # Add to heap
                counter += 1
                heapq.heappush(heap, (f, counter, neighbor, path + [neighbor]))

        # No path found - return direct path as fallback
        return [start, goal]

    def should_lay_egg_on_route(self, current_pos: Tuple[int, int],
                               target_pos: Tuple[int, int],
                               board: "game_board.Board") -> bool:
        """
        Decide if we should lay egg at current position while routing to target.

        Decision criteria:
        - If force_egg_on_route is True, lay eggs aggressively (always if possible)
        - Path distance to target > MIN_PATH_LENGTH_FOR_EGGS
        - Cell is fresh (not previously egged)
        - Can lay egg (have egg available)
        - Still can reach target within reasonable moves

        Args:
            current_pos: Current position
            target_pos: Destination position
            board: Board state

        Returns:
            True if should lay egg here
        """
        # Check if we can lay egg
        if not board.can_lay_egg():
            return False

        # Check if cell already has egg
        if current_pos in board.eggs_player:
            return False

        # Check exploration status
        if self.exploration and self.exploration.has_egg(current_pos):
            return False

        # NEW: If force_egg_on_route is enabled, lay eggs aggressively
        if hasattr(self, '_force_egg_on_route') and self._force_egg_on_route:
            # Still check basic safety (not too close to target to avoid blocking)
            dist_to_target = manhattan_distance(current_pos, target_pos)
            if dist_to_target >= 2:  # Leave at least 2 moves to reach target
                return True

        # Calculate distance to target
        dist_to_target = manhattan_distance(current_pos, target_pos)

        # If path is short, go direct (no egg laying)
        if dist_to_target < self.MIN_PATH_LENGTH_FOR_EGGS:
            return False

        # Check if cell is fresh
        if self.exploration:
            freshness = self.exploration.get_freshness_score(current_pos, board.turn_count)
            if freshness > 0.5:  # Fresh enough
                return True

        # Default: lay egg if we haven't been here
        return current_pos not in board.eggs_player

    def generate_move_sequence(self, start: Tuple[int, int],
                              target: Tuple[int, int],
                              board: "game_board.Board",
                              max_moves: int = 5,
                              force_egg_on_route: bool = False) -> List[Tuple[Direction, MoveType]]:
        """
        Generate sequence of moves to reach target.

        Returns list of (Direction, MoveType) tuples with opportunistic egg placement.

        Args:
            start: Starting position
            target: Target position
            board: Board state
            max_moves: Maximum moves to generate
            force_egg_on_route: If True, lay eggs aggressively during traversal

        Returns:
            List of (Direction, MoveType) moves
        """
        # Store flag for use in should_lay_egg_on_route
        self._force_egg_on_route = force_egg_on_route

        # Get path
        path = self.plan_route(start, target, board, force_egg_on_route=force_egg_on_route)

        if len(path) < 2:
            return []

        # Convert path to moves
        moves = []
        for i in range(min(len(path) - 1, max_moves)):
            current = path[i]
            next_pos = path[i + 1]

            # Determine direction
            direction = self._get_direction(current, next_pos)
            if direction is None:
                continue

            # Determine move type (egg or plain)
            move_type = MoveType.PLAIN

            # Check if should lay egg
            if self.should_lay_egg_on_route(current, target, board):
                move_type = MoveType.EGG

            moves.append((direction, move_type))

        return moves

    def _get_direction(self, from_pos: Tuple[int, int],
                      to_pos: Tuple[int, int]) -> Optional[Direction]:
        """
        Get direction from one position to adjacent position.

        Returns:
            Direction enum or None if not adjacent
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if dx == 0 and dy == 1:
            return Direction.DOWN
        elif dx == 0 and dy == -1:
            return Direction.UP
        elif dx == 1 and dy == 0:
            return Direction.RIGHT
        elif dx == -1 and dy == 0:
            return Direction.LEFT
        else:
            return None

    def get_next_move_toward_target(self, current_pos: Tuple[int, int],
                                   target_pos: Tuple[int, int],
                                   board: "game_board.Board") -> Optional[Tuple[Direction, MoveType]]:
        """
        Get single best move toward target.

        Simplified interface for immediate move selection.

        Args:
            current_pos: Current position
            target_pos: Target position
            board: Board state

        Returns:
            (Direction, MoveType) or None if no valid move
        """
        moves = self.generate_move_sequence(current_pos, target_pos, board, max_moves=1)

        if moves:
            return moves[0]

        return None

    def find_efficient_backtrack_route(self, current_pos: Tuple[int, int],
                                      previous_target: Tuple[int, int],
                                      board: "game_board.Board") -> List[Tuple[int, int]]:
        """
        Find efficient route when backtracking from a completed objective.

        Prioritizes unexplored cells and egg-laying opportunities.

        Args:
            current_pos: Current position
            previous_target: Where we just came from
            board: Board state

        Returns:
            Path through fresh territory
        """
        if not self.exploration:
            # No exploration tracker - use standard pathfinding
            return []

        # Find nearest unexplored region
        nearest_unexplored = self.exploration.find_nearest_unexplored(current_pos, board)

        if nearest_unexplored:
            # Route to fresh territory
            return self.exploration.get_backtrack_route(current_pos, nearest_unexplored, board)

        # All territory explored - route to corners (high value)
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        # Find nearest unoccupied corner
        best_corner = None
        best_dist = float('inf')

        for corner in corners:
            if corner not in board.eggs_player and corner not in board.eggs_enemy:
                dist = manhattan_distance(current_pos, corner)
                if dist < best_dist:
                    best_dist = dist
                    best_corner = corner

        if best_corner:
            return self.plan_route(current_pos, best_corner, board)

        return []

    def invalidate_cache(self):
        """Invalidate cached path (call when board state changes significantly)."""
        self._cache_valid = False
        self._cached_path = None
        self._cached_target = None

    def get_path_efficiency_score(self, path: List[Tuple[int, int]],
                                  board: "game_board.Board") -> float:
        """
        Score a path for efficiency.

        Higher score = better path
        Factors: length, freshness, egg opportunities, risk

        Args:
            path: List of positions
            board: Board state

        Returns:
            Efficiency score
        """
        if not path:
            return 0.0

        score = 0.0

        # Length penalty (shorter = better)
        score -= len(path) * 0.5

        # Freshness bonus
        if self.exploration:
            for pos in path:
                freshness = self.exploration.get_freshness_score(pos, board.turn_count)
                score += freshness * 2.0

        # Egg opportunity bonus
        egg_opportunities = 0
        for pos in path:
            if (pos not in board.eggs_player and
                pos not in board.eggs_enemy and
                (not self.exploration or not self.exploration.has_egg(pos))):
                egg_opportunities += 1

        score += egg_opportunities * 3.0

        # Risk penalty
        if self.tracker:
            for pos in path:
                risk = self.tracker.get_trapdoor_risk(pos)
                score -= risk * 5.0

        return score
