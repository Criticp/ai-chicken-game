"""
Scuba Steve - Tournament Agent
Deployed: V6 Numba Edition (High-Performance JIT-Compiled Search)
"""

# Try to use Numba-accelerated agent (10-100x faster)
try:
    from .agent_numba import PlayerAgent
    print("[Agent] Using Numba-accelerated search engine")
except ImportError as e:
    print(f"[Agent] Numba not available: {e}")
    print("[Agent] Falling back to V5 (Python implementation)")
    from .agent_v5 import PlayerAgent

# V6 Numba Edition includes:
# - Iterative Deepening Negamax (JIT-compiled)
# - Alpha-Beta Pruning (JIT-compiled)
# - Transposition Table with Zobrist Hashing (fixed-size array)
# - Pure heuristic evaluation (JIT-compiled, no neural network)
# - All weights centralized in weights_config.py
# - Expected depth: 8-15 plies (vs 4-6 in Python version)
