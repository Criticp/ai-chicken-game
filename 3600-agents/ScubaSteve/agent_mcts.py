"""
Scuba Steve MCTS - Agent with Monte Carlo Tree Search

Hybrid approach:
- MCTS for strategic decisions (simulates to turn 80)
- Neural policy guidance for faster convergence
- Bayesian trapdoor tracking for probabilistic sampling

Key improvements over Negamax:
1. Full-game simulations reveal long-term turd conservation value
2. Statistical learning eliminates need for hand-coded penalties
3. Natural handling of trapdoor uncertainty through many rollouts
"""

from collections.abc import Callable
from collections import deque
from typing import List, Tuple, Dict
import json
import os
import sys

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Direction, MoveType, Result, loc_after_direction

from .trapdoor_tracker import TrapdoorTracker
from .mcts_engine import MCTSEngine
from .evaluator import HybridEvaluator
from .turd_advisor import TurdAdvisor

# ScubaSteve V6 components
from .separator_planner import SeparatorPlanner
from .exploration_tracker import ExplorationTracker
from .path_planner import PathPlanner
from .strategic_mode_manager import StrategyManager, StrategyMode


class PlayerAgent:
    """
    Scuba Steve MCTS: Monte Carlo Tree Search Agent

    Core Components:
    - Monte Carlo Tree Search (MCTS) with UCB1 selection
    - Bayesian trapdoor inference
    - Neural policy for simulation guidance
    - Full-game rollouts to turn 80

    Advantages:
    - No depth horizon limitation
    - Learns resource conservation through win statistics
    - Handles uncertainty through probabilistic sampling
    """

    def __init__(self, board: "game_board.Board", time_left: Callable):
        """Initialize MCTS agent components"""

        # Component 1: Trapdoor Belief Engine
        self.tracker = TrapdoorTracker(map_size=8)

        # Component 3: Hybrid Evaluator (for terminal evaluation)
        model_path = os.path.join(os.path.dirname(__file__), 'chicken_eval_model_numpy.json')
        use_neural = False  # Fast heuristic mode

        if use_neural:
            self.evaluator = HybridEvaluator(self.tracker, model_path)
        else:
            self.evaluator = HybridEvaluator(self.tracker, None)

        # Neural policy for MCTS simulation guidance
        self.neural_policy = None
        try:
            from .neural_policy import NeuralPolicy
            self.neural_policy = NeuralPolicy()
            policy_path = os.path.join(os.path.dirname(__file__), 'chickennet_hybrid_policy.pth')
            if os.path.exists(policy_path):
                self.neural_policy.load(policy_path)
                print("[MCTS Agent] Neural policy loaded for simulation guidance")
        except Exception as e:
            print(f"[MCTS Agent] Neural policy not available: {e}")

        # Component 2: MCTS Search Engine
        self.mcts_engine = MCTSEngine(
            trapdoor_tracker=self.tracker,
            neural_policy=self.neural_policy,
            max_time_per_move=5.0
        )

        # V6 Components (optional, for advanced features)
        try:
            self.separator_planner = SeparatorPlanner(trapdoor_tracker=self.tracker)
        except Exception as e:
            print(f"[SeparatorPlanner] Failed: {e}")
            self.separator_planner = None

        try:
            self.exploration_tracker = ExplorationTracker()
        except Exception as e:
            print(f"[ExplorationTracker] Failed: {e}")
            self.exploration_tracker = None

        try:
            self.path_planner = PathPlanner(
                exploration_tracker=self.exploration_tracker,
                trapdoor_tracker=self.tracker
            )
        except Exception as e:
            print(f"[PathPlanner] Failed: {e}")
            self.path_planner = None

        try:
            self.strategy_manager = StrategyManager(separator_planner=self.separator_planner)
        except Exception as e:
            print(f"[StrategyManager] Failed: {e}")
            self.strategy_manager = None

        # Tracking
        self.move_count = 0
        self.pos_history = []
        self.move_history: deque = deque(maxlen=4)

        # Death detection
        self.prev_location = None
        self.spawn_location = None
        self._initialized = False

        print("[Scuba Steve MCTS] All systems initialized")
        print("  - MCTS Engine: Full-game simulations to turn 80")
        print("  - Trapdoor Tracker: Bayesian inference")
        print(f"  - Neural Policy: {'Loaded' if self.neural_policy else 'Heuristic fallback'}")

    def play(self, board: "game_board.Board", trapdoor_samples=None, time_left: Callable = None) -> Tuple[Direction, MoveType]:
        """
        Select the best move using MCTS.

        Args:
            board: Current game state
            trapdoor_samples: Sensor data (ignored, we use board.get_sensor_data())
            time_left: Callable returning remaining time
            
        Returns:
            Best move as (Direction, MoveType)
        """
        # Handle old signature compatibility
        if callable(trapdoor_samples) and time_left is None:
            time_left = trapdoor_samples
            trapdoor_samples = None

        self.move_count += 1
        current_loc = board.chicken_player.get_location()

        # First-turn initialization
        if not self._initialized:
            self.spawn_location = current_loc
            self.prev_location = current_loc
            self._initialized = True
            print(f"[MCTS Agent] Initialized at spawn {self.spawn_location}")

        # Death detection (teleported back to spawn)
        if self.prev_location and current_loc == self.spawn_location and self.prev_location != self.spawn_location:
            if board.turn_count > 0:  # Not first turn
                # We died at prev_location
                print(f"[MCTS Agent] DEATH DETECTED at {self.prev_location}")
                self.tracker.mark_death_location(self.prev_location)

        # Update trapdoor tracker with sensor data
        if trapdoor_samples:
            self.tracker.update_from_sensors(current_loc, trapdoor_samples)

        # Update exploration tracker
        if self.exploration_tracker:
            self.exploration_tracker.update(current_loc, board.turn_count)

        # Breadcrumb trail (anti-looping)
        self.pos_history.append(current_loc)
        if len(self.pos_history) > 8:
            self.pos_history.pop(0)

        # Run MCTS search
        try:
            move = self.mcts_engine.search(
                board=board,
                time_left=time_left,
                move_history=None,  # MCTS doesn't use move history
                pos_history=self.pos_history
            )
        except Exception as e:
            print(f"[MCTS Agent] Search failed: {e}")
            # Fallback to first valid move
            valid_moves = board.get_valid_moves(enemy=False)
            if valid_moves:
                # Prioritize eggs
                egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                move = egg_moves[0] if egg_moves else valid_moves[0]
            else:
                move = (Direction.UP, MoveType.PLAIN)

        # Update tracking
        self.prev_location = current_loc
        self.move_history.append(move)

        # Track egg placements
        if move[1] == MoveType.EGG and self.exploration_tracker:
            self.exploration_tracker.mark_egged(current_loc)

        direction, move_type = move
        print(f"[Turn {board.turn_count}] MCTS selected: {direction.name} {move_type.name} "
              f"(iterations: {self.mcts_engine.iterations})")

        return move

