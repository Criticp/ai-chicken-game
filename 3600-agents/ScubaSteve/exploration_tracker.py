"""
Exploration Tracker for ScubaSteve

Keeps lightweight memory about visited squares, eggs placed and turds placed.
Provides freshness scores and simple helper routes used by the PathPlanner.
"""
from typing import Tuple, Dict, Set, Optional, List
from collections import deque
from game.board import manhattan_distance


class ExplorationTracker:
    """Simple exploration memory used by PathPlanner and higher-level logic.

    Responsibilities:
    - Track visited squares and the turn they were last visited
    - Track where we've placed eggs and turds
    - Provide freshness/heatmap information
    - Provide a nearest-unexplored helper and a simple backtrack route
    """

    def __init__(self):
        # Maps position -> last turn visited
        self.visited_turn: Dict[Tuple[int, int], int] = {}

        # Cells where we have placed eggs/turds
        self.egged: Set[Tuple[int, int]] = set()
        self.turded: Set[Tuple[int, int]] = set()

        # Simple cache of last stats
        self._last_stats = None

    def update(self, current_loc: Tuple[int, int], turn_count: int):
        """Mark current location as visited on this turn."""
        self.visited_turn[current_loc] = turn_count
        self._last_stats = None

    def mark_egged(self, pos: Tuple[int, int]):
        """Record that we placed an egg at pos."""
        if pos:
            self.egged.add(pos)
            self._last_stats = None

    def mark_turd(self, pos: Tuple[int, int]):
        """Record that we placed a turd at pos."""
        if pos:
            self.turded.add(pos)
            self._last_stats = None

    def has_egg(self, pos: Tuple[int, int]) -> bool:
        return pos in self.egged

    def has_turd(self, pos: Tuple[int, int]) -> bool:
        return pos in self.turded

    def get_freshness_score(self, pos: Tuple[int, int], current_turn: Optional[int] = None) -> float:
        """Return freshness in range [0,1].

        Freshness is 1.0 for never-visited cells, and decays towards 0 as time since visit increases.
        """
        if pos not in self.visited_turn:
            return 1.0

        if current_turn is None:
            # If caller didn't provide current turn, assume fairly fresh
            return 0.2

        age = current_turn - self.visited_turn.get(pos, current_turn)
        # Decay function: freshness = max(0, 1 - age / 20)
        return max(0.0, 1.0 - (age / 20.0))

    def get_heatmap(self, board) -> Dict[Tuple[int, int], float]:
        """Return a dict of position -> heat (higher means more attractive / fresh).

        The PathPlanner expects small numeric bonuses; keep values modest (0..1).
        """
        heatmap: Dict[Tuple[int, int], float] = {}
        for x in range(8):
            for y in range(8):
                pos = (x, y)

                # Skip if occupied by any turd
                if pos in board.turds_player or pos in board.turds_enemy:
                    heatmap[pos] = 0.0
                    continue

                # Prefer cells without eggs
                egg_bonus = 0.0
                if pos not in board.eggs_player and pos not in board.eggs_enemy and pos not in self.egged:
                    egg_bonus = 0.8

                # Freshness based on visit age
                freshness = self.get_freshness_score(pos, board.turn_count)

                heatmap[pos] = freshness * 0.6 + egg_bonus * 0.4

        return heatmap

    def find_nearest_unexplored(self, current_pos: Tuple[int, int], board) -> Optional[Tuple[int, int]]:
        """Return nearest cell that does not yet have our egg (or never visited).

        Prefers cells that are reachable and not occupied.
        """
        best = None
        best_dist = float('inf')

        for x in range(8):
            for y in range(8):
                pos = (x, y)

                # Skip occupied cells
                if pos in board.turds_player or pos in board.turds_enemy:
                    continue
                if pos in board.eggs_player or pos in board.eggs_enemy:
                    continue

                # Prefer never-visited or not-our-egged cells
                if pos in self.egged:
                    continue

                # Compute distance
                d = manhattan_distance(current_pos, pos)
                if d < best_dist:
                    best_dist = d
                    best = pos

        return best

    def get_backtrack_route(self, start: Tuple[int, int], target: Tuple[int, int], board) -> List[Tuple[int, int]]:
        """Return a simple BFS shortest path from start to target avoiding turds.

        This is intentionally lightweight to avoid importing PathPlanner (no circular deps).
        """
        from collections import deque

        if start == target:
            return [start]

        queue = deque()
        queue.append((start, [start]))
        visited = {start}

        while queue:
            pos, path = queue.popleft()
            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                npos = (nx, ny)

                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                if npos in visited:
                    continue
                if npos in board.turds_player or npos in board.turds_enemy:
                    continue

                new_path = path + [npos]
                if npos == target:
                    return new_path

                visited.add(npos)
                queue.append((npos, new_path))

        # Fallback: direct two-step path (may be invalid)
        return [start, target]

    def get_exploration_stats(self) -> Dict[str, int]:
        """Return counts for status printing."""
        if self._last_stats is None:
            self._last_stats = {
                'visited_cells': len(self.visited_turn),
                'egged_cells': len(self.egged),
                'turd_cells': len(self.turded)
            }
        return self._last_stats

