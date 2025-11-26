# Scuba Steve - Tournament AI Agent

## Overview
Scuba Steve is a data-driven AI agent for the CS3600 Chicken Game Tournament, trained on 18,000 game replays. The agent combines:
- **Alpha-Beta Minimax Search** with adaptive depth
- **Bayesian Trapdoor Tracking** for risk assessment
- **Machine-Learned Evaluation** from 5,447 high-quality games
- **Efficient time management** for 6-minute games

## Training Results

### Dataset Statistics
- **Total games analyzed**: 18,108
- **High-quality games used**: 5,447 (winner ≥12 eggs, margin ≥2)
- **Board states analyzed**: 50,000
- **Opening book positions**: 133

### Learned Evaluation Weights (R² = 0.514)
Trained via Linear Regression on 50k board positions:

| Feature | Weight | Interpretation |
|---------|--------|----------------|
| **Egg Differential** | 7.65 | Most critical - maximize egg advantage |
| **Mobility** | 0.17 | Slight advantage for more valid moves |
| **Corner Proximity** | -0.04 | Corners slightly less important than expected |
| **Turd Usage Diff** | 0.34 | Enemy using turds is beneficial |
| **Trapdoor Risk** | -20.0 | High penalty for dangerous positions |

## Architecture

### Module A: Oracle (Belief State Manager)
**File**: `belief.py`
- Bayesian inference for trapdoor location estimation
- Tracks safe cells (visited without triggering)
- Updates probabilities based on sensor data (hear/feel)
- Provides risk scores for path planning

### Module B: Calculator (Alpha-Beta Search)
**File**: `agent.py` - `_search()`, `_minimax()`
- **Adaptive depth**: 2-6 ply based on time remaining
- **Alpha-beta pruning**: Efficient tree exploration
- **Move ordering**: Eggs → Plains → Turds for better cutoffs
- **Time management**: Allocates ~20% of time per move, max 10s

### Module C: Evaluator (Data-Tuned Heuristic)
**File**: `agent.py` - `_evaluate()`
- **Learned weights** from regression on historical games
- **Features**: Egg diff, mobility, corners, turds, trapdoor risk
- **Bonuses**: Egg-laying opportunities (+1.0)
- **Penalties**: Low mobility/trapped (-10.0)

### Module D: Opening Book (Simplified)
**File**: `opening_book.json`
- 133 early-game positions mapped to best moves
- Extracted from high-scoring game sequences
- Currently not actively used (hash matching needs refinement)
- Future enhancement opportunity

## Performance Characteristics

### Search Efficiency
- **Nodes per move**: ~3,000-4,000 at depth 5
- **Time per move**: 0.15-0.20 seconds (early game)
- **Branching factor**: ~8-12 moves typical
- **Pruning effectiveness**: Alpha-beta cuts ~40-60% of tree

### Time Management
- **Early game** (moves 1-5): Depth 5-6, thorough search
- **Mid game** (moves 6-30): Depth 4-5, balanced
- **Late game** (moves 31-40): Depth 2-3, fast decisions
- **Safety margin**: Reserves time for final moves

## Strategy Insights from Data

### Winning Patterns (from 5,447 games)
1. **Egg priority is paramount**: Weight of 7.65 >> all other features
2. **Mobility matters**: Avoid getting trapped (weight: 0.17)
3. **Corners are overrated**: Slight negative weight (-0.04)
4. **Enemy turds help**: Positive weight (0.34) - they block themselves
5. **Trapdoor avoidance critical**: Heavy penalty (-20.0)

### Tactical Guidelines
- **Lay eggs aggressively** when on correct parity squares
- **Maintain mobility** - always have escape routes
- **Use turds sparingly** - they limit your own movement
- **Track trapdoors** - update beliefs every turn
- **Manage time** - don't over-search in early game

## File Structure

```
agents/ScubaSteve/
├── __init__.py              # Package initialization
├── agent.py                 # Main PlayerAgent class (280 lines)
├── belief.py                # TrapdoorBelief manager (160 lines)
├── learned_weights.json     # Regression coefficients
└── opening_book.json        # Early game positions (133 entries)
```

## Usage

### In Tournament
The agent is ready to submit as-is. Simply include the `ScubaSteve/` folder in your submission.

### Local Testing
```bash
cd engine
python run_local_agents.py ScubaSteve Steve
```

### Training Pipeline
To retrain with updated data:
```bash
cd training_ground
python data_processor.py
```

This regenerates:
- `learned_weights.json` with updated regression coefficients
- `opening_book.json` with refined opening sequences

## Key Advantages

1. **Data-Driven**: Evaluation function learned from real tournament games
2. **Adaptive**: Adjusts search depth based on time pressure
3. **Safe**: Bayesian trapdoor tracking minimizes risk
4. **Efficient**: Alpha-beta pruning + move ordering
5. **Robust**: Handles edge cases, time limits, trapped positions

## Future Enhancements

1. **Improved Opening Book**: Better state hashing for position matching
2. **Transposition Tables**: Cache evaluated positions
3. **Quiescence Search**: Extend search in tactical positions
4. **Neural Policy Network**: For move ordering (if <5ms inference)
5. **Endgame Tablebases**: Pre-computed optimal play in simple endings

## Credits

**Agent Name**: Scuba Steve  
**Architecture**: Minimax + Bayesian Inference + Regression Learning  
**Training Data**: 18,108 tournament games (5,447 filtered)  
**Evaluation Model**: Linear Regression (R² = 0.514)  
**Search Algorithm**: Alpha-Beta with adaptive depth  

---

*"Stay calm and lay eggs"* - Scuba Steve

