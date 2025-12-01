"""
Fixed-size transposition table optimized for Numba JIT compilation.

Replaces dict-based table with NumPy array for 10x faster lookups.
"""

import numpy as np
from numba import njit

from .numba_types import TT_EXACT, TT_LOWERBOUND, TT_UPPERBOUND
from .weights_config import *

# Default transposition table size (can be overridden)
DEFAULT_TT_SIZE = 1_000_000


# ═══════════════════════════════════════════════════════════════
# FIXED-SIZE TRANSPOSITION TABLE (GLOBAL)
# ═══════════════════════════════════════════════════════════════

# Structure: [hash_key, depth, flag, score]
# - hash_key: Zobrist hash of board position (int64)
# - depth: Search depth when this entry was stored (int32)
# - flag: TT_EXACT, TT_LOWERBOUND, or TT_UPPERBOUND (int8)
# - score: Evaluation score (float32 stored as int32 for uniformity)

# Initialize global transposition table
_TT_SIZE = DEFAULT_TT_SIZE
TRANSPOSITION_TABLE = np.zeros((DEFAULT_TT_SIZE, 4), dtype=np.int64)


def init_transposition_table(size: int = DEFAULT_TT_SIZE):
    """
    Initialize or resize the global transposition table.

    Args:
        size: Number of entries in table
    """
    global TRANSPOSITION_TABLE, _TT_SIZE
    _TT_SIZE = size
    TRANSPOSITION_TABLE = np.zeros((size, 4), dtype=np.int64)
    print(f"[TT] Initialized transposition table with {size:,} entries")


def clear_transposition_table():
    """Clear all entries in transposition table"""
    global TRANSPOSITION_TABLE
    TRANSPOSITION_TABLE.fill(0)


def get_tt_stats():
    """
    Get statistics about transposition table usage.

    Returns:
        (entries_used, utilization_percent)
    """
    global TRANSPOSITION_TABLE
    entries_used = np.count_nonzero(TRANSPOSITION_TABLE[:, 0])
    utilization = (entries_used / _TT_SIZE) * 100
    return entries_used, utilization


# ═══════════════════════════════════════════════════════════════
# JIT-COMPILED TRANSPOSITION TABLE OPERATIONS
# ═══════════════════════════════════════════════════════════════

@njit
def tt_store(tt_array: np.ndarray,
            hash_key: np.int64,
            depth: np.int32,
            flag: np.int8,
            score: np.float32):
    """
    Store an entry in the transposition table.

    Uses hash % size for indexing (simple replacement scheme).

    Args:
        tt_array: The transposition table array
        hash_key: Zobrist hash of position
        depth: Search depth
        flag: TT_EXACT, TT_LOWERBOUND, or TT_UPPERBOUND
        score: Evaluation score
    """
    tt_size = tt_array.shape[0]
    idx = hash_key % tt_size

    # Convert float32 score to int64 for storage (preserve bits)
    score_bits = np.int64(np.float32(score).view(np.int32))

    tt_array[idx, 0] = hash_key
    tt_array[idx, 1] = depth
    tt_array[idx, 2] = flag
    tt_array[idx, 3] = score_bits


@njit
def tt_lookup(tt_array: np.ndarray,
             hash_key: np.int64,
             depth: np.int32,
             alpha: np.float32,
             beta: np.float32) -> tuple:
    """
    Look up position in transposition table.

    Returns cached score if:
    1. Hash matches (same position)
    2. Stored depth >= current depth (searched deeper before)
    3. Bound type allows cutoff

    Args:
        tt_array: The transposition table array
        hash_key: Zobrist hash to look up
        depth: Current search depth
        alpha: Current alpha value
        beta: Current beta value

    Returns:
        (found, score, new_alpha, new_beta)
        - found: True if usable entry exists
        - score: Cached score (only valid if found=True)
        - new_alpha: Updated alpha (may be same)
        - new_beta: Updated beta (may be same)
    """
    tt_size = tt_array.shape[0]
    idx = hash_key % tt_size

    stored_hash = tt_array[idx, 0]

    # Check if entry exists and matches
    if stored_hash != hash_key:
        return (False, np.float32(0.0), alpha, beta)

    stored_depth = tt_array[idx, 1]

    # Only use if searched to sufficient depth
    if stored_depth < depth:
        return (False, np.float32(0.0), alpha, beta)

    stored_flag = tt_array[idx, 2]
    score_bits = tt_array[idx, 3]

    # Convert int64 back to float32
    score = np.int32(score_bits).view(np.float32)

    # Apply transposition table cutoffs based on bound type
    if stored_flag == TT_EXACT:
        # Exact score - can return immediately
        return (True, score, alpha, beta)

    elif stored_flag == TT_LOWERBOUND:
        # Lower bound - update alpha
        if score > alpha:
            alpha = score
        if alpha >= beta:
            # Beta cutoff
            return (True, score, alpha, beta)

    elif stored_flag == TT_UPPERBOUND:
        # Upper bound - update beta
        if score < beta:
            beta = score
        if alpha >= beta:
            # Alpha cutoff
            return (True, score, alpha, beta)

    # Entry found but no cutoff
    return (False, score, alpha, beta)


@njit
def tt_probe(tt_array: np.ndarray, hash_key: np.int64) -> tuple:
    """
    Simple probe to check if position exists in table.

    Returns:
        (exists, depth, flag, score)
    """
    tt_size = tt_array.shape[0]
    idx = hash_key % tt_size

    stored_hash = tt_array[idx, 0]

    if stored_hash != hash_key:
        return (False, 0, 0, np.float32(0.0))

    stored_depth = tt_array[idx, 1]
    stored_flag = tt_array[idx, 2]
    score_bits = tt_array[idx, 3]
    score = np.int32(score_bits).view(np.float32)

    return (True, stored_depth, stored_flag, score)


# ═══════════════════════════════════════════════════════════════
# WARMUP FUNCTION
# ═══════════════════════════════════════════════════════════════

def warmup_tt_jit():
    """Warm up transposition table JIT compilation"""
    print("[TT] Warming up transposition table JIT...")

    dummy_tt = np.zeros((1000, 4), dtype=np.int64)

    # Test store
    tt_store(dummy_tt, np.int64(12345), np.int32(5), TT_EXACT, np.float32(100.0))

    # Test lookup
    _ = tt_lookup(dummy_tt, np.int64(12345), np.int32(3), np.float32(-1000.0), np.float32(1000.0))

    # Test probe
    _ = tt_probe(dummy_tt, np.int64(12345))

    print("[TT] Transposition table JIT ready!")

