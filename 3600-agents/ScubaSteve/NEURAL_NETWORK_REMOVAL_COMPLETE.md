# Neural Network Removal & Configuration Centralization - COMPLETE

## Overview
Complete removal of neural network components from the search engine, increased search depth, and centralized all weights into a single configuration file for easy tuning.

## Changes Made

### 1. Created Centralized Weights Configuration
**File: `weights_config.py`**

All heuristic weights and search parameters are now in ONE file:

#### Search Parameters
- `MAX_SEARCH_DEPTH = 8` (increased from 6)
- `MAX_TIME_PER_MOVE = 5.0` seconds
- `TIME_FRACTION_OF_REMAINING = 0.3` (use max 30% of remaining time)
- `TIME_MULTIPLIER_PER_MOVE = 2.0` (or 2x average time per move)

#### Move Ordering Weights (for Alpha-Beta)
- `CORNER_EGG_PRIORITY = -10000.0` (highest)
- `REGULAR_EGG_PRIORITY = -1000.0`
- `SEPARATOR_EGG_BONUS = -100.0`
- `PLAIN_MOVE_PRIORITY = 0.0`
- `TURD_ABSOLUTE_BAN_PRIORITY = 10000.0` (lowest - turns 0-3)
- `TURD_EARLY_GAME_PRIORITY = 1000.0`
- `TURD_SEPARATOR_PRIORITY = 500.0`
- `TURD_NON_SEPARATOR_PRIORITY = 800.0`

#### Move Scoring Weights (in search tree)
- `EGG_CORNER_BONUS = 400.0`
- `EGG_REGULAR_BONUS = 250.0`
- `PLAIN_MOVE_PENALTY = 150.0`
- `TURD_ABSOLUTE_BAN_PENALTY = 10000.0`
- `TURD_EARLY_HEAVY_PENALTY = 1000.0`
- `TURD_MID_SAVE_LAST_PENALTY = 500.0`
- `TURD_MID_GENERAL_PENALTY = 200.0`
- `TURD_NON_SEPARATOR_PENALTY = 300.0`
- `TURD_EARLY_SEPARATOR_BONUS = 50.0`
- `TURD_MID_SEPARATOR_BONUS = 150.0`
- `TURD_LATE_SEPARATOR_BONUS = 300.0`
- `TURD_LATE_NON_SEPARATOR_BONUS = 50.0`

#### Game Phase Thresholds
- `ABSOLUTE_BAN_TURD_TURNS = 3`
- `EARLY_GAME_THRESHOLD = 10`
- `MID_GAME_THRESHOLD = 30`
- `EARLY_GAME_MIN_TURDS = 4`
- `MID_GAME_MIN_TURDS = 2`

#### Terminal State Scores
- `WIN_SCORE = 10000`
- `BLOCKED_PENALTY_MULTIPLIER = 5.0`

#### Safety Protocol
- `LAVA_FLOOR_ENABLED = True`
- `SAFETY_THRESHOLD = 0.05` (5% max trap risk)

### 2. Updated `search_engine.py`

#### Removed Neural Network Code
- ✅ Removed `NeuralPolicy` import
- ✅ Removed neural policy initialization in `__init__`
- ✅ Removed `_create_board_state_for_nn()` helper function
- ✅ Removed hybrid neural+heuristic move ordering
- ✅ Removed neural_eval position history injection
- ✅ Simplified blocked penalty calculation (no neural_eval dependency)

#### Increased Search Depth
- Changed from hardcoded `range(1, 7)` to `range(1, MAX_SEARCH_DEPTH + 1)`
- Now searches up to depth 8 (configurable in `weights_config.py`)

#### Centralized All Weights
- All weight constants now imported from `weights_config.py`
- All hardcoded values replaced with named constants
- Time management constants from config
- Game phase thresholds from config

#### Pure Heuristic Move Ordering
The `_order_moves()` function now uses 100% heuristic ordering:
1. Corner eggs (highest priority)
2. Separator eggs
3. Regular eggs
4. Plain moves
5. Late-game turds on separators
6. Early-game turds (lowest priority)

## How the System Works Together

### Iterative Deepening + Transposition Table + Alpha-Beta
```
For depth = 1 to MAX_SEARCH_DEPTH:
    For each move (ordered by heuristics):
        Negamax(depth - 1):
            - Check transposition table (cache hit?)
            - If depth == 0: call evaluator
            - Else: recurse with alpha-beta pruning
            - Store result in transposition table
```

**Key Points:**
1. **Transposition Table**: Caches board positions using Zobrist hashing
   - Avoids re-evaluating the same position reached via different move orders
   - Stores (score, depth, best_move) for each position
   - Only uses cached result if cached depth ≥ current depth

2. **Iterative Deepening**: Searches depth 1, 2, 3, ... until time runs out
   - Always has a move available (from depth 1)
   - Deeper searches override shallower results
   - Time management prevents timeout

3. **Alpha-Beta Pruning**: Eliminates branches that can't affect result
   - Move ordering critical: better moves first = more cutoffs
   - Heuristic ordering prioritizes eggs over turds

4. **Evaluator at Depth 0**: Base case for recursion
   - Called when depth reaches 0 (leaf nodes)
   - Returns static evaluation of board position
   - Score propagates back up the tree

## Testing & Tuning

### To Change Search Depth
Edit `weights_config.py`:
```python
MAX_SEARCH_DEPTH = 10  # Try deeper search
```

### To Adjust Turd Strategy
Edit `weights_config.py`:
```python
TURD_EARLY_HEAVY_PENALTY = 500.0  # Make less conservative
EARLY_GAME_THRESHOLD = 15  # Extend early game phase
```

### To Modify Egg Bonuses
Edit `weights_config.py`:
```python
EGG_CORNER_BONUS = 500.0  # Prioritize corners more
SEPARATOR_EGG_BONUS = -200.0  # Prioritize separators more
```

### To Adjust Time Management
Edit `weights_config.py`:
```python
MAX_TIME_PER_MOVE = 3.0  # Faster moves
TIME_FRACTION_OF_REMAINING = 0.5  # Use more time early
```

## Architecture Benefits

### ✅ No Neural Network Dependencies
- Pure Python (no PyTorch required)
- Faster startup (no model loading)
- Simpler debugging
- Deterministic behavior

### ✅ Easy Tuning
- All weights in ONE file
- Change and test instantly
- Document different strategies
- Version control friendly

### ✅ Optimal Search Performance
- Transposition table reduces nodes searched
- Iterative deepening ensures move availability
- Alpha-beta pruning eliminates bad branches
- Move ordering maximizes cutoffs

### ✅ Proper Minimax Structure
- Evaluator only called at depth 0 (leaf nodes)
- Negamax simplifies implementation
- Terminal states handled correctly
- Score propagation via recursion

## Performance Expectations

With depth 8 and transposition table:
- **Early game** (many moves): Reaches depth 4-6
- **Mid game** (moderate moves): Reaches depth 6-7
- **Late game** (few moves): Reaches full depth 8
- **Nodes searched**: 1000-10000 per move (with TT hits)
- **Transposition hits**: 20-50% of nodes (big time saver)

## Status: ✅ COMPLETE

All neural network code removed. All weights centralized. Search depth increased to 8. System properly integrated with transposition table + iterative deepening + alpha-beta + depth-0 evaluation.

Ready for testing and tuning!

