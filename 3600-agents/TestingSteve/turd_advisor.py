"""
Turd Advisor - Sidecar module

Non-invasive advisory system that evaluates board cells for potential turd
placements. Designed to run alongside the existing agent without changing
decision logic. If the agent chooses a TURD move, we log/return the advisor's
recommendation for diagnostics and future tuning.

API:
- TurdAdvisor(neural_eval): create with NeuralEvaluator (or any object with
  .territory_evaluator and .tracker and .fallback_weights)
- recommend(board) -> (best_cell, score, reasons)
- score_all_cells(board) -> list of (cell,score,reasons)

This module intentionally does NOT alter the agent's moves. It only computes
and returns scores and a human-readable breakdown.
"""

from typing import Tuple, List, Dict, Optional
from collections import namedtuple

CellScore = namedtuple('CellScore', ['cell', 'score', 'reasons'])


class TurdAdvisor:
    """Evaluate board cells for turd placement value.

    Uses the existing TerritoryEvaluator when available so weights and logic
    are consistent with the rest of the agent. Falls back to simple
    heuristics if territory engine is not present.
    """

    def __init__(self, neural_eval):
        """neural_eval: instance of NeuralEvaluator (or HybridEvaluator.neural_eval)
        Must provide: .territory_evaluator (optional), .tracker (optional),
        .fallback_weights (dict)
        """
        self.neural_eval = neural_eval
        self.territory = getattr(neural_eval, 'territory_evaluator', None)
        self.tracker = getattr(neural_eval, 'tracker', None)
        self.weights = getattr(neural_eval, 'fallback_weights', {}) if neural_eval else {}

        # Friendly defaults if weights missing
        self.default_weights = {
            'turd_cut_weight': 8.0,
            'turd_killer': 10.0,
            'turd_aggression': 5.0,
            'proximity_bonus': 50.0,
            'conservation_penalty_per_turd': -30.0,
        }

    def _get_weight(self, key: str) -> float:
        return float(self.weights.get(key, self.default_weights.get(key, 0.0)))

    def score_cell(self, board, cell: Tuple[int, int]) -> CellScore:
        """Score a single board cell as a candidate turd placement.

        Returns CellScore(cell, score, reasons_dict)
        """
        x, y = cell

        reasons: Dict[str, float] = {}

        # If cell is occupied by any asset, give heavy negative score
        occupied = False
        try:
            if cell in board.eggs_player or cell in board.eggs_enemy or cell in board.turds_player or cell in board.turds_enemy:
                occupied = True
        except Exception:
            occupied = False

        if occupied:
            reasons['occupied'] = -10000.0
            return CellScore(cell, -10000.0, reasons)

        # Base score components
        score = 0.0

        # 1) Conservation & Impact via TerritoryEvaluator if available
        if self.territory:
            try:
                impact, weighted = self.territory.evaluate_turd_with_conservation(board, cell)
                reasons['impact'] = float(impact)
                reasons['conservation_weighted_score'] = float(weighted)

                # Use the weighted score directly (it already applies conservation rules)
                score += float(weighted)
            except Exception:
                reasons['impact'] = 0.0
                reasons['conservation_weighted_score'] = 0.0
        else:
            # Fallback: small positive for empty center-ish squares
            cx = abs(3.5 - x) + abs(3.5 - y)
            fallback = max(0.0, 8.0 - cx)
            reasons['fallback_area'] = fallback
            score += fallback

        # 2) Choke point bonus
        if self.territory:
            try:
                choke_bonus = self.territory.evaluate_choke_point(board, cell)
                reasons['choke_bonus_raw'] = float(choke_bonus)

                # weigh by configured cut weight
                cut_w = self._get_weight('turd_cut_weight')
                score += choke_bonus * cut_w / max(1.0, cut_w)  # normalize a bit
            except Exception:
                reasons['choke_bonus_raw'] = 0.0
        else:
            reasons['choke_bonus_raw'] = 0.0

        # 3) Proximity to enemy (drive-off): prefer cells that are nearer to enemy
        try:
            enemy_loc = board.chicken_enemy.get_location()
            dist = abs(enemy_loc[0] - x) + abs(enemy_loc[1] - y)
            # closer -> higher bonus; linear ramp with cap
            proximity_weight = self._get_weight('turd_aggression')
            proximity_bonus = max(0.0, (6 - dist)) * (proximity_weight)
            reasons['proximity_dist'] = float(dist)
            reasons['proximity_bonus'] = float(proximity_bonus)
            score += proximity_bonus
        except Exception:
            reasons['proximity_bonus'] = 0.0

        # 4) Conservation penalty: discourage immediate dumping if many turds left
        try:
            turds_left = board.chicken_player.get_turds_left()
            pen_per_turd = self._get_weight('conservation_penalty_per_turd')
            # If many turds available, small penalty per use; if few left, larger penalty
            conserve_penalty = pen_per_turd if turds_left > 2 else pen_per_turd * 1.5
            reasons['conservation_penalty'] = float(-conserve_penalty)
            score -= conserve_penalty
        except Exception:
            reasons['conservation_penalty'] = 0.0

        # 5) Risk: avoid high trapdoor risk squares
        if self.tracker:
            try:
                risk = self.tracker.get_trapdoor_risk(cell)
                # scale risk to penalty
                risk_pen = -risk * 100.0
                reasons['trapdoor_risk'] = float(risk)
                reasons['trapdoor_penalty'] = float(risk_pen)
                score += risk_pen
            except Exception:
                reasons['trapdoor_penalty'] = 0.0

        # Final assembly
        reasons['final_score'] = float(score)
        return CellScore(cell, float(score), reasons)

    def score_all_cells(self, board) -> List[CellScore]:
        """Score all empty board cells for turd placement.

        Returns list sorted by descending score.
        """
        candidates: List[CellScore] = []
        board_size = 8

        for y in range(board_size):
            for x in range(board_size):
                cell = (x, y)
                # skip if cell is the enemy or our current chicken position (advisor considers future spots too)
                try:
                    if cell == board.chicken_enemy.get_location():
                        continue
                except Exception:
                    pass

                cs = self.score_cell(board, cell)
                candidates.append(cs)

        candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)
        return candidates_sorted

    def recommend(self, board) -> Optional[CellScore]:
        """Return the highest scoring cell or None if none suitable."""
        scored = self.score_all_cells(board)
        if not scored:
            return None
        return scored[0]

