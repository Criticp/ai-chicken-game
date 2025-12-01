"""
Component 3: The Evaluation Function (Hybrid Neural Net)

Combines:
- Safety Mask (hard-coded safety checks)
- Residual CNN for position evaluation
- Input: C × 8 × 8 tensor with channels for:
  [My Chicken, Enemy Chicken, My Eggs, Enemy Eggs, My Turds, Enemy Turds, Trapdoor Risk Map]
- Output: Expected Egg Differential (scalar)
"""

import numpy as np
from typing import Tuple, Optional, List
import os
import json
import sys

# Add engine to path
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from game import board as game_board
from game.enums import Result

# Defense & Zoning Specialist - Territory control only
try:
    from .territory_engine import TerritoryEvaluator
    TERRITORY_AVAILABLE = True
except ImportError:
    TERRITORY_AVAILABLE = False
    print("[WARNING] Territory engine not available")


class ResidualBlock:
    """
    Single Residual Block for CNN.
    Structure: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv layers (3x3 kernels)
        self.conv1_weight = None
        self.conv1_bias = None
        self.bn1_weight = None
        self.bn1_bias = None
        self.bn1_mean = None
        self.bn1_var = None

        self.conv2_weight = None
        self.conv2_bias = None
        self.bn2_weight = None
        self.bn2_bias = None
        self.bn2_mean = None
        self.bn2_var = None

        # Skip connection (if dimensions change)
        self.skip_conv_weight = None
        self.skip_conv_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through residual block"""
        identity = x

        # Conv1 -> BN1 -> ReLU
        out = self._conv2d(x, self.conv1_weight, bias=None)  # No bias
        out = self._batch_norm(out, self.bn1_weight, self.bn1_bias,
                              self.bn1_mean, self.bn1_var)
        out = np.maximum(0, out)

        # Conv2 -> BN2
        out = self._conv2d(out, self.conv2_weight, bias=None)  # No bias
        out = self._batch_norm(out, self.bn2_weight, self.bn2_bias,
                              self.bn2_mean, self.bn2_var)

        # Skip connection
        if self.skip_conv_weight is not None:
            identity = self._conv2d(identity, self.skip_conv_weight, bias=None)

        # Add and ReLU
        out = out + identity
        out = np.maximum(0, out)

        return out

    def _conv2d(self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimized 3x3 convolution with padding=1 using vectorization"""
        # x shape: (C_in, H, W)
        # weight shape: (C_out, C_in, 3, 3)
        # Output: (C_out, H, W)

        C_in, H, W = x.shape
        C_out = weight.shape[0]

        # Pad input
        x_padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode='constant')

        # Output
        out = np.zeros((C_out, H, W), dtype=np.float32)

        # Optimized: Use einsum for batched convolution
        for c_out in range(C_out):
            for h in range(H):
                for w in range(W):
                    # Extract 3x3 patch and compute dot product
                    patch = x_padded[:, h:h+3, w:w+3]
                    out[c_out, h, w] = np.sum(patch * weight[c_out])

        # Add bias if present
        if bias is not None:
            out += bias[:, np.newaxis, np.newaxis]

        return out

    def _batch_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Batch normalization (inference mode)"""
        # x shape: (C, H, W)
        # Normalize each channel
        normalized = np.zeros_like(x)
        for c in range(x.shape[0]):
            normalized[c] = gamma[c] * (x[c] - mean[c]) / np.sqrt(var[c] + eps) + beta[c]
        return normalized


class NeuralEvaluator:
    """
    Hybrid evaluator combining safety checks with neural network.

    Architecture:
    - Input: 7 channels × 8 × 8
    - ResBlock1: 7 -> 32 channels
    - ResBlock2: 32 -> 32 channels
    - ResBlock3: 32 -> 16 channels
    - Flatten + Dense: (16*8*8=1024) -> 128 -> 1
    """

    # Safety thresholds
    MAX_RISK_TOLERANCE = 0.60  # Don't step on squares with >60% trapdoor risk
    BLOCKED_PENALTY = -500  # Penalty for having no valid moves (5 egg penalty)

    def __init__(self, trapdoor_tracker):
        """
        Args:
            trapdoor_tracker: TrapdoorTracker instance for risk assessment
        """
        self.tracker = trapdoor_tracker

        # Model components
        self.res_block1 = None
        self.res_block2 = None
        self.res_block3 = None

        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None

        # DEFENSE & ZONING SPECIALIST
        # Territory evaluator with weaponized geography (traps as walls)
        if TERRITORY_AVAILABLE:
            self.territory_evaluator = TerritoryEvaluator(trapdoor_tracker=trapdoor_tracker)
        else:
            self.territory_evaluator = None

        # REMOVED: Tactical suicide (incompatible with zero-tolerance safety)

        # V7 FACELIFT: Rebalanced weights for egg maximization + efficiency
        # Philosophy: EGG FIRST, safety always, strategic turds only
        # Target: 25+ eggs/game, 95%+ movement efficiency, 80%+ win rate
        self.fallback_weights = {
            # CORE GAME METRICS - Egg maximization is KING
            'egg_diff': 100.0,  # MASSIVE BOOST - egg differential is everything
            'egg_placement_bonus': 150.0,  # CRITICAL: Always prefer laying eggs (was 50)

            # MOVEMENT & EFFICIENCY - Explore efficiently
            'mobility': 3.0,  # Increased - having options matters
            'corner_proximity': 2.0,  # Moderate - corners are strategic
            'turd_diff': 1.0,  # Low - turds less important than eggs

            # SAFETY METRICS - Never die
            'trapdoor_risk': -100.0,  # HIGH PENALTY - safety is critical
            'territory_control': 5.0,  # Increased - controlling board matters
            'safe_mobility_press': 4.0,  # Increased - restrict opponent movement

            # ANTI-LOOPING - High efficiency required
            'repetition_penalty': -50.0,  # STRONG - don't revisit positions (was -15)
            'recent_visit_penalty': -100.0,  # STRONG - recent revisits very bad (was -50)
            'egged_cell_penalty': -200.0,  # SEVERE - never walk on eggs (was -100)
            'backtrack_penalty': -30.0,  # Moderate - discourage backtracking (was -10)
            'egg_walk_penalty': -5.0,  # Increased - walking on eggs wastes turn

            # EXPLORATION BONUSES - Maximum coverage
            'exploration_freshness': 80.0,  # HUGE BOOST - unvisited cells top priority (was 30)
            'density_bonus': 3.0,  # Increased - fill in gaps
            'invasion_weight': 2.0,  # Moderate - push forward
            'egg_hunter_weight': 3.0,  # Increased - target fresh territory
            'corner_gravity': 4.0,  # Increased - corners are valuable

            # TURD STRATEGY - Conservative but strategic
            'turd_usage_penalty': -300.0,  # SEVERE - save turds (was -200)
            'early_turd_penalty': -1000.0,  # MASSIVE - NO early turds! (was -500)
            'turd_aggression': 5.0,  # Moderate - blocking is good when needed
            'turd_cut_weight': 10.0,  # Increased - cutting territory valuable
            'turd_killer': 15.0,  # Increased - removing enemy options powerful
            'parity_turd_bonus': 20.0,  # Increased - parity blocking excellent

            # SIMPLIFIED - Removed complex mode bonuses (invasion, frenzy, walls)
            # Focus on core gameplay instead

            'intercept': 0.0,  # Neutral baseline
        }

        # INVASION MODE: Context passed from agent
        self.invasion_mode = False
        self.exploration_frenzy = False  # NEW: Exploration frenzy mode
        self.territory_tracker = None
        self.exploration_tracker = None
        self.spawn_location = None  # For distance calculations in frenzy mode

        # WALL STRATEGY: Separator planner reference
        self.separator_planner = None  # Will be set by agent


        self.model_loaded = False

    def load_model(self, model_path: str):
        """Load pre-trained ResNet model weights from JSON"""
        try:
            print(f"[NeuralEvaluator] Loading model from {model_path}...")

            with open(model_path, 'r') as f:
                weights = json.load(f)

            # Initialize ResBlocks
            self.res_block1 = ResidualBlock(7, 32)
            self._load_resblock_weights(self.res_block1, weights, 'res1')

            self.res_block2 = ResidualBlock(32, 32)
            self._load_resblock_weights(self.res_block2, weights, 'res2')

            self.res_block3 = ResidualBlock(32, 16)
            self._load_resblock_weights(self.res_block3, weights, 'res3')

            # Load FC layers
            self.fc1_weight = np.array(weights['fc1_weight'], dtype=np.float32)
            self.fc1_bias = np.array(weights['fc1_bias'], dtype=np.float32)
            self.fc2_weight = np.array(weights['fc2_weight'], dtype=np.float32)
            self.fc2_bias = np.array(weights['fc2_bias'], dtype=np.float32)

            self.model_loaded = True
            print(f"[NeuralEvaluator] ✓ ResNet model loaded successfully!")

        except Exception as e:
            print(f"[NeuralEvaluator] Failed to load model: {e}")
            print(f"[NeuralEvaluator] Using fallback heuristic evaluation")
            self.model_loaded = False

    def _load_resblock_weights(self, block: ResidualBlock, weights: dict, prefix: str):
        """Load weights into a ResidualBlock"""
        block.conv1_weight = np.array(weights[f'{prefix}_conv1_weight'], dtype=np.float32)
        block.bn1_weight = np.array(weights[f'{prefix}_bn1_weight'], dtype=np.float32)
        block.bn1_bias = np.array(weights[f'{prefix}_bn1_bias'], dtype=np.float32)
        block.bn1_mean = np.array(weights[f'{prefix}_bn1_mean'], dtype=np.float32)
        block.bn1_var = np.array(weights[f'{prefix}_bn1_var'], dtype=np.float32)

        block.conv2_weight = np.array(weights[f'{prefix}_conv2_weight'], dtype=np.float32)
        block.bn2_weight = np.array(weights[f'{prefix}_bn2_weight'], dtype=np.float32)
        block.bn2_bias = np.array(weights[f'{prefix}_bn2_bias'], dtype=np.float32)
        block.bn2_mean = np.array(weights[f'{prefix}_bn2_mean'], dtype=np.float32)
        block.bn2_var = np.array(weights[f'{prefix}_bn2_var'], dtype=np.float32)

        # Skip connection (if exists)
        skip_key = f'{prefix}_skip_weight'
        if skip_key in weights:
            block.skip_conv_weight = np.array(weights[skip_key], dtype=np.float32)


    def evaluate(self, board: "game_board.Board", depth: int = 0) -> float:
        """
        Evaluate board position using 50-50 Neural-Heuristic Hybrid.

        Args:
            board: Board state to evaluate
            depth: Search depth (for depth-based adjustments)

        Returns:
            Score from player's perspective (higher = better for player)
        """
        # SAFETY MASK: Check terminal states first
        if board.winner == Result.PLAYER:
            return 10000 - depth
        elif board.winner == Result.ENEMY:
            return -10000 + depth
        elif board.winner == Result.TIE:
            return 0

        # SAFETY MASK: Check if blocked (no valid moves)
        my_moves = board.get_valid_moves(enemy=False)
        if not my_moves:
            return self.BLOCKED_PENALTY  # -500 (5 egg penalty × weight)

        # V7 FACELIFT: Expected Value (EV) Risk Management
        # Instead of a hard penalty, calculate the expected score based on trap probability.
        my_loc = board.chicken_player.get_location()
        trap_prob = self.tracker.get_trapdoor_risk(my_loc)

        # 1. Calculate the raw score of the position assuming it's safe
        if self.model_loaded:
            # Get both evaluations
            neural_score = self._neural_evaluate(board)
            heuristic_score = self._heuristic_evaluate(board)

            # Blend 50-50
            base_score = 0.5 * neural_score + 0.5 * heuristic_score

            # Track usage for debugging
            if not hasattr(self, '_eval_count'): self._eval_count = 0
            self._eval_count += 1
            if self._eval_count % 100 == 0:
                print(f"[V7 Hybrid] Neural: {neural_score:.1f}, Heuristic: {heuristic_score:.1f}, Blended: {base_score:.1f}")
        else:
            # Fallback to heuristic only
            base_score = self._heuristic_evaluate(board)

        # 2. Define cost of hitting a trap (Teleport + Enemy gets 4 eggs + Lost tempo)
        # A conservative estimate is -600 points.
        TRAP_PENALTY = -600.0

        # 3. Calculate Expected Value
        expected_value = ((1.0 - trap_prob) * base_score) + (trap_prob * TRAP_PENALTY)

        return expected_value

    def _neural_evaluate(self, board: "game_board.Board") -> float:
        """
        Evaluate using ResNet CNN.

        Input tensor: 7 × 8 × 8
        Channels:
        0: My Chicken (1 at position, 0 elsewhere)
        1: Enemy Chicken (1 at position, 0 elsewhere)
        2: My Eggs (1 where eggs, 0 elsewhere)
        3: Enemy Eggs
        4: My Turds
        5: Enemy Turds
        6: Trapdoor Risk Map (0.0 to 1.0)
        """
        # Create input tensor
        input_tensor = self._create_input_tensor(board)

        # Forward pass through ResBlocks
        x = self.res_block1.forward(input_tensor)
        x = self.res_block2.forward(x)
        x = self.res_block3.forward(x)

        # Flatten: (16, 8, 8) -> (1024,)
        x = x.flatten()

        # FC1: 1024 -> 128
        x = np.dot(x, self.fc1_weight.T) + self.fc1_bias
        x = np.maximum(0, x)  # ReLU

        # FC2: 128 -> 1
        output = np.dot(x, self.fc2_weight.T) + self.fc2_bias

        return float(output[0])

    def _create_input_tensor(self, board: "game_board.Board") -> np.ndarray:
        """
        Create 7 × 8 × 8 input tensor from board state.

        Returns:
            numpy array of shape (7, 8, 8)
        """
        tensor = np.zeros((7, 8, 8), dtype=np.float32)

        # Channel 0: My Chicken
        px, py = board.chicken_player.get_location()
        tensor[0, py, px] = 1.0

        # Channel 1: Enemy Chicken
        ex, ey = board.chicken_enemy.get_location()
        tensor[1, ey, ex] = 1.0

        # Channel 2: My Eggs
        for (x, y) in board.eggs_player:
            tensor[2, y, x] = 1.0

        # Channel 3: Enemy Eggs
        for (x, y) in board.eggs_enemy:
            tensor[3, y, x] = 1.0

        # Channel 4: My Turds
        for (x, y) in board.turds_player:
            tensor[4, y, x] = 1.0

        # Channel 5: Enemy Turds
        for (x, y) in board.turds_enemy:
            tensor[5, y, x] = 1.0

        # Channel 6: Trapdoor Risk Map
        risk_map = self.tracker.get_risk_map()
        tensor[6] = risk_map

        return tensor

    def _heuristic_evaluate(self, board: "game_board.Board") -> float:
        """
        PROJECT FERTILITY: Simplified fertility-based evaluation.
        Prevents safety loops and parity dead-ends.
        """
        from .fertility_eval import heuristic_evaluate_fertility
        return heuristic_evaluate_fertility(self, board)
        # Initialize score
        score = 0.0

        # Extract features
        egg_diff = board.chicken_player.eggs_laid - board.chicken_enemy.eggs_laid
        score += self.fallback_weights['egg_diff'] * egg_diff

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        # OLD: Simple mobility counting
        # my_moves = len(board.get_valid_moves(enemy=False))
        # enemy_moves = len(board.get_valid_moves(enemy=True))
        # NEW: Territory control via flood fill with weaponized geography
        # Treats probable traps (>10% risk) as walls
        if self.territory_evaluator:
            my_area, enemy_area, territory_score = self.territory_evaluator.evaluate_control(board)
            score += self.fallback_weights['territory_control'] * territory_score

        # Mobility (immediate moves)
        my_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        mobility_diff = my_moves - enemy_moves
        score += self.fallback_weights['mobility'] * mobility_diff

        corner_diff = self._corner_score(my_loc) - self._corner_score(enemy_loc)
        score += self.fallback_weights['corner_proximity'] * corner_diff

        turd_diff = board.chicken_enemy.turds_left - board.chicken_player.turds_left
        score += self.fallback_weights['turd_diff'] * turd_diff

        # ═══════════════════════════════════════════════════════════════
        # LAVA FLOOR PROTOCOL: Zero-tolerance trap avoidance
        # ═══════════════════════════════════════════════════════════════

        my_risk = self.tracker.get_trapdoor_risk(my_loc)
        enemy_risk = self.tracker.get_trapdoor_risk(enemy_loc)

        # MASSIVE penalty for being on risky square
        score += self.fallback_weights['trapdoor_risk'] * my_risk  # -100 per 1.0 risk

        # Bonus if enemy is on risky square
        score -= self.fallback_weights['trapdoor_risk'] * enemy_risk  # Double negative = positive

        # ═══════════════════════════════════════════════════════════════
        # THE PRESS: Safe Mobility Differential
        # ═══════════════════════════════════════════════════════════════

        # Count SAFE moves (risk <= 5%) for both players
        my_safe_moves = self._count_safe_moves(board, enemy=False)
        enemy_safe_moves = self._count_safe_moves(board, enemy=True)
        safe_mobility_diff = my_safe_moves - enemy_safe_moves

        # Reward constricting enemy's SAFE options
        score += self.fallback_weights['safe_mobility_press'] * safe_mobility_diff

        # ═══════════════════════════════════════════════════════════════
        # FINAL POLISH: The Breadcrumb Trail (Anti-Looping)
        # ═══════════════════════════════════════════════════════════════

        # Get position history from search engine (if available)
        pos_history = getattr(self, '_current_pos_history', [])
        if pos_history and my_loc in pos_history:
            # Apply repetition penalty: -50 per occurrence
            repetition_count = pos_history.count(my_loc)
            repetition_penalty = self.fallback_weights['repetition_penalty'] * repetition_count
            score += repetition_penalty

            if repetition_count >= 2:
                # Severe looping detected!
                score += repetition_penalty  # Double penalty

        # ═══════════════════════════════════════════════════════════════
        # FINAL POLISH: Egg Walk Tax
        # ══════════════════════════════════���════════════════════════════

        # Penalize walking on own eggs (wastes a turn unless high territory gain)
        if my_loc in board.eggs_player:
            score += self.fallback_weights['egg_walk_penalty']

        # ═══════════════════════════════════════════════════════════════
        # ANTI-LOOPING: Strengthen loop prevention
        # ═══════════════════════════════════════════════════════════════

        # Penalty for cells we've recently visited (last 8 turns)
        if self.exploration_tracker:
            if my_loc in self.exploration_tracker.visited_turn:
                last_visit = self.exploration_tracker.visited_turn[my_loc]
                turns_since_visit = board.turn_count - last_visit

                # Heavy penalty if visited in last 8 turns
                if turns_since_visit < 8:
                    score += self.fallback_weights['recent_visit_penalty']

        # Massive penalty for moving to cells with our eggs (complete waste)
        if my_loc in board.eggs_player:
            score += self.fallback_weights['egged_cell_penalty']  # -300

            # EXTRA PENALTY during normal mode (non-invasion): Make it almost impossible
            if not self.invasion_mode:
                score -= 200.0  # Total: -500 penalty for egg-backtracking in normal mode
                print(f"[ANTI-BACKTRACK] Severe penalty for revisiting egg at {my_loc}")

        # ═══════════════════════════════════════════════════════════════
        # INVASION MODE: Aggressive opponent territory exploration
        # ═══════════════════════════════════════════════════════════════

        if self.invasion_mode and self.territory_tracker:
            # Massive bonus for being on opponent's side
            if self.territory_tracker.is_opponent_side(my_loc):
                score += self.fallback_weights['invasion_opponent_side_bonus']

                # Even bigger bonus if this is an unexplored opponent cell
                if self.exploration_tracker:
                    if my_loc not in self.exploration_tracker.visited_turn:
                        score += self.fallback_weights['invasion_unexplored_bonus']
                        print(f"[INVASION] Unexplored opponent cell {my_loc} - bonus: +{self.fallback_weights['invasion_unexplored_bonus']}")

            # Penalty for staying on our side during invasion
            elif self.territory_tracker.is_our_side(my_loc):
                score -= 50.0  # Push agent to cross over

        # NORMAL MODE: Bonus for exploring OUR expanded territory
        # Encourage filling our 60% of the board before invading
        elif not self.invasion_mode and self.territory_tracker:
            if self.territory_tracker.is_our_side(my_loc):
                # Bonus for being on our side
                score += 25.0

                # Extra bonus for unexplored cells on our side
                if self.exploration_tracker:
                    if my_loc not in self.exploration_tracker.visited_turn:
                        score += 40.0  # Encourage exploring new cells on our side

                # Extra bonus if not egged yet
                if my_loc not in board.eggs_player:
                    score += 15.0  # Prefer fresh cells

        # ═══════════════════════════════════════════════════════════════
        # EXPLORATION FRENZY MODE: Pure exploration focus
        # ═══════════════════════════════════════════════════════════════

        if self.exploration_frenzy and self.exploration_tracker:
            # MASSIVE bonus for visiting completely unvisited cells
            if my_loc not in self.exploration_tracker.visited_turn:
                score += self.fallback_weights['frenzy_unvisited_bonus']  # +200

                # Extra bonus if it's far from spawn (encourage wide exploration)
                spawn_loc = getattr(self, 'spawn_location', None)
                if spawn_loc:
                    distance_from_spawn = abs(my_loc[0] - spawn_loc[0]) + abs(my_loc[1] - spawn_loc[1])
                    score += distance_from_spawn * 15.0  # Reward distance from spawn

            # Multiply normal exploration bonuses
            exploration_bonus = self.fallback_weights.get('exploration_bonus', 0)
            if exploration_bonus > 0:
                score += exploration_bonus * (self.fallback_weights['frenzy_exploration_multiplier'] - 1)  # Add 2x more

            # Heavily penalize staying still or moving to already-visited cells
            if my_loc in self.exploration_tracker.visited_turn:
                visits = self.exploration_tracker.visited_turn[my_loc]
                score -= 100.0 * (1 + visits)  # Escalating penalty for multiple visits

            # Still respect trapdoor safety (don't override safety penalties)
            # The trapdoor penalties are already applied above, so frenzy won't override them

        # ═══════════════════════════════════════════════════════════════
        # TURD WALL STRATEGY: Strategic wall building bonuses
        # ═══════════════════════════════════════════════════════════════

        # EARLY GAME (turns < 30): Massive bonus for being at wall positions
        if board.turn_count < 30 and self.separator_planner:
            if self.separator_planner.is_wall_position(my_loc):
                wall_bonus = self.fallback_weights['wall_division_bonus']

                # Multiplier if walls already partially built (synergy)
                if self.separator_planner.both_walls_active():
                    wall_bonus *= self.fallback_weights['wall_completion_multiplier']

                score += wall_bonus

                if board.turn_count % 10 == 0:  # Log occasionally
                    print(f"[WALL BONUS] Position {my_loc} on wall line: +{wall_bonus:.0f}")

        # MID/LATE GAME (turns >= 30): Avoid saturated territory
        if board.turn_count >= 30 and self.territory_tracker:
            saturation = self.territory_tracker.calculate_saturation(board)

            # If our side is saturated (>70%), penalize being there
            if saturation >= 70.0:
                if self.territory_tracker.is_our_side(my_loc):
                    # Heavy penalty for staying on saturated side
                    saturation_penalty = self.fallback_weights['saturated_area_penalty']
                    score += saturation_penalty

                    if board.turn_count % 10 == 0:
                        print(f"[SATURATION] Our side {saturation:.0f}% full, penalty for staying: {saturation_penalty:.0f}")

                # Bonus for moving to opponent side (fresh territory)
                elif self.territory_tracker.is_opponent_side(my_loc):
                    opponent_bonus = self.fallback_weights['fresh_opponent_bonus']
                    score += opponent_bonus

                    if board.turn_count % 10 == 0:
                        print(f"[FRESH TERRITORY] Opponent side bonus: +{opponent_bonus:.0f}")

        # ═══════════════════════════════════════════════════════════════
        # PROJECT VANGUARD: Aggression & Corner Hunting
        # ═══════════════════════════════════════════════════════════════

        # Phase 1: The "Invasion" Bias (Crossing the Rubicon)
        invasion_score = self._calculate_invasion_score(board, my_loc)
        score += invasion_score * self.fallback_weights['invasion_weight']

        # Phase 1.2: The "Egg Hunter" Logic
        hunter_score = self._calculate_egg_hunter_score(board, my_loc, enemy_loc)
        score += hunter_score * self.fallback_weights['egg_hunter_weight']

        # Phase 2: The "Corner King" Module (Enhanced gravity well)
        corner_score = self._calculate_corner_gravity(board, my_loc)
        score += corner_score * self.fallback_weights['corner_gravity']

        # Intercept
        score += self.fallback_weights['intercept']

        # ═══════════════════════════════════════════════════════════════
        # BONUSES & PENALTIES
        # ═══════════════════════════════════════════════════════════════

        if board.can_lay_egg():
            egg_bonus = 6.0

            # Double bonus during invasion mode
            if self.invasion_mode:
                egg_bonus *= self.fallback_weights['invasion_egg_placement_multiplier']

            if self._is_corner(my_loc):
                score += 25.0  # Corner egg = 4 eggs (3x bonus)
            else:
                score += egg_bonus

        # Critical mobility
        if my_moves < 2:
            score -= 15.0
        if enemy_moves < 2:
            score += 10.0

        return score

    def _count_safe_moves(self, board: "game_board.Board", enemy: bool = False) -> int:
        """
        THE PRESS: Count moves that are SAFE (risk <= 5%).

        This is different from total legal moves. We only count moves
        to squares we KNOW are safe, not just legal.
        """
        valid_moves = board.get_valid_moves(enemy=enemy)

        # Get current position
        if enemy:
            current_loc = board.chicken_enemy.get_location()
        else:
            current_loc = board.chicken_player.get_location()

        safe_count = 0
        for direction, move_type in valid_moves:
            # Calculate destination
            from game.enums import loc_after_direction
            new_loc = loc_after_direction(current_loc, direction)

            # Check if destination is safe (Lava Floor Protocol)
            if self.tracker.is_safe(new_loc):
                safe_count += 1

        return safe_count

    def _evaluate_turd_cut(self, board: "game_board.Board", turd_loc: Tuple[int, int]) -> float:
        """
        FINAL POLISH: The "Great Wall" - Calculate turd cut value.

        Returns the reduction in enemy's reachable area if turd placed at turd_loc.
        This identifies choke points where 1 turd can slice off large territory sections.
        """
        if not self.territory_evaluator:
            return 0.0

        try:
            # Calculate enemy area BEFORE turd
            _, enemy_area_before, _ = self.territory_evaluator.evaluate_control(board)

            # Simulate turd placement
            simulated_turds = board.turds_player.copy()
            simulated_turds.add(turd_loc)

            # Temporarily modify board (hacky but works for evaluation)
            original_turds = board.turds_player
            board.turds_player = simulated_turds

            # Calculate enemy area AFTER turd
            _, enemy_area_after, _ = self.territory_evaluator.evaluate_control(board)

            # Restore original state
            board.turds_player = original_turds

            # Cut value = how much territory we denied
            cut_value = enemy_area_before - enemy_area_after

            return cut_value

        except Exception:
            # Fallback if simulation fails
            return 0.0

    def _calculate_invasion_score(self, board: "game_board.Board", my_loc: Tuple[int, int]) -> float:
        """
        PROJECT VANGUARD - Phase 1: The "Invasion" Bias

        Directive 1.1: Forward progress into enemy territory
        Rewards moving away from spawn toward enemy concentration

        Returns:
            Invasion score based on distance from spawn axis
        """
        # Get spawn position (edge of board)
        my_spawn = self._get_spawn_position(board)

        # Calculate distance from spawn (Manhattan distance)
        spawn_dist = abs(my_loc[0] - my_spawn[0]) + abs(my_loc[1] - my_spawn[1])

        # Normalize to 0-10 range
        invasion_score = min(spawn_dist, 10)

        return float(invasion_score)

    def _calculate_egg_hunter_score(self, board: "game_board.Board",
                                    my_loc: Tuple[int, int],
                                    enemy_loc: Tuple[int, int]) -> float:
        """
        PROJECT VANGUARD - Phase 1.2: The "Egg Hunter" Logic

        Directive 1.2: Attraction to enemy egg clusters
        Calculates centroid of enemy eggs and rewards proximity

        Returns:
            Hunter score based on distance to enemy egg cluster
        """
        enemy_eggs = list(board.eggs_enemy)

        if not enemy_eggs:
            # No enemy eggs yet, track enemy chicken instead
            centroid_x, centroid_y = enemy_loc
        else:
            # Calculate centroid of enemy eggs
            centroid_x = sum(x for x, y in enemy_eggs) / len(enemy_eggs)
            centroid_y = sum(y for x, y in enemy_eggs) / len(enemy_eggs)

        # Manhattan distance to centroid
        dist_to_cluster = abs(my_loc[0] - centroid_x) + abs(my_loc[1] - centroid_y)

        # Invert: closer = higher score (max 10)
        hunter_score = max(0, 10 - dist_to_cluster)

        return float(hunter_score)

    def _calculate_corner_gravity(self, board: "game_board.Board", my_loc: Tuple[int, int]) -> float:
        """
        PROJECT VANGUARD - Phase 2: The "Corner King" Module

        Directive 2.1: Corner Proximity Gradient
        Creates gravity well around corners (worth 3x eggs)
        Guides agent toward corners even from distance

        Returns:
            Corner gravity score (higher when closer to valid corner)
        """
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        # Find nearest corner that doesn't have an egg yet
        min_dist = float('inf')
        for corner in corners:
            # Skip if corner already has an egg
            if corner in board.eggs_player or corner in board.eggs_enemy:
                continue

            # Calculate distance
            dist = abs(my_loc[0] - corner[0]) + abs(my_loc[1] - corner[1])
            min_dist = min(min_dist, dist)

        # If all corners occupied, use closest anyway
        if min_dist == float('inf'):
            min_dist = min(abs(my_loc[0] - c[0]) + abs(my_loc[1] - c[1]) for c in corners)

        # Gravity score: 8 at corner, decreases with distance
        gravity_score = max(0, 8 - min_dist)

        return float(gravity_score)

    def _get_spawn_position(self, board: "game_board.Board") -> Tuple[int, int]:
        """Get estimated spawn position (usually on edge)"""
        # Try to get from board/chicken
        if hasattr(board.chicken_player, 'spawn'):
            return board.chicken_player.spawn

        # Fallback: estimate based on current position
        # Spawns are typically on edges (x=0, x=7, y=0, y=7)
        current = board.chicken_player.get_location()
        x, y = current

        # Estimate closest edge as spawn
        if x < 4:
            spawn_x = 0
        else:
            spawn_x = 7

        spawn_y = y  # Usually similar y coordinate

        return (spawn_x, spawn_y)

    def _corner_score(self, loc: Tuple[int, int]) -> float:
        """Corner proximity score"""
        x, y = loc
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        return 14 - min(abs(x - cx) + abs(y - cy) for cx, cy in corners)

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        """Check if location is a corner"""
        x, y = loc
        return (x == 0 or x == 7) and (y == 0 or y == 7)


class HybridEvaluator:
    """
    Wrapper that combines safety checks with neural evaluation.
    This is the main evaluation function used by the search engine.
    """

    def __init__(self, trapdoor_tracker, model_path: Optional[str] = None):
        """
        Args:
            trapdoor_tracker: TrapdoorTracker instance
            model_path: Optional path to pre-trained model weights
        """
        self.neural_eval = NeuralEvaluator(trapdoor_tracker)

        # Try to load model
        if model_path and os.path.exists(model_path):
            self.neural_eval.load_model(model_path)

    def __call__(self, board: "game_board.Board", depth: int = 0) -> float:
        """
        Evaluate board position.
        Callable interface for SearchEngine.
        """
        return self.neural_eval.evaluate(board, depth)

    def evaluate_move_safety(self, board: "game_board.Board",
                            move: Tuple) -> Tuple[bool, float]:
        """
        Check if a move is safe before considering it.

        Returns:
            (is_safe, penalty) tuple
        """
        from game.enums import loc_after_direction

        direction, move_type = move
        current_loc = board.chicken_player.get_location()
        dest_loc = loc_after_direction(current_loc, direction)

        # Check risk at destination
        risk = self.neural_eval.tracker.get_trapdoor_risk(dest_loc)

        if risk > NeuralEvaluator.MAX_RISK_TOLERANCE:
            return (False, -500.0)  # Unsafe

        return (True, 0.0)  # Safe

    def apply_position_penalty(self, score: float, position: Tuple[int, int],
                               history: List[Tuple[int, int]]) -> float:
        """
        Apply anti-looping penalty based on position history.

        Lead Architect Directive:
        repetition_count = history[-4:].count(position)
        score -= repetition_count * 20.0

        Args:
            score: Base score for the position
            position: The position being evaluated
            history: Recent position history (last 4+ positions)

        Returns:
            Adjusted score with repetition penalty
        """
        if not history:
            return score

        # Count how many times this position appears in recent history
        recent_history = history[-4:] if len(history) >= 4 else history
        repetition_count = recent_history.count(position)

        # Apply penalty: -20 points per repetition
        penalty = repetition_count * 20.0

        if penalty > 0:
            # Debug output for significant penalties
            if repetition_count >= 2:
                print(f"[AntiLoop] Position {position} visited {repetition_count}x recently - penalty: {penalty}")

        return score - penalty

    def evaluate_turd_move(self, board: "game_board.Board", turd_loc: Tuple[int, int]) -> float:
        """
        DEEPCHICKEN: Evaluate turd move with conservation logic.

        Directive 1.2 & 1.3: Conservation Threshold + Turd Killer Weight
        - Calculates connectivity collapse impact
        - Applies conservation threshold (< 8 = waste)
        - Returns weighted score

        Args:
            board: Current board state
            turd_loc: Proposed turd location

        Returns:
            Score adjustment for turd placement
        """
        # Check if territory evaluator available
        if not hasattr(self.neural_eval, 'territory_evaluator') or not self.neural_eval.territory_evaluator:
            # Fallback: use basic turd aggression
            return 0.0

        try:
            # Calculate connectivity collapse using Zone of Denial
            impact, weighted_score = self.neural_eval.territory_evaluator.evaluate_turd_with_conservation(
                board, turd_loc
            )

            return weighted_score

        except Exception:
            # Fallback on error
            return 0.0
