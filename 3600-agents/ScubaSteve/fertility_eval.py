"""
PROJECT FERTILITY - Comprehensive Fix for ScubaSteve V7

This module implements the fertility-based evaluation to replace the
old heuristic_evaluate function that causes safety loops and parity dead-ends.
"""

from typing import Tuple
from game.enums import loc_after_direction


def heuristic_evaluate_fertility(evaluator, board) -> float:
    """
    PROJECT FERTILITY: Heuristic evaluation with fertility sensing and anti-looping.
    
    Key upgrades:
    - Fertility Score: Values moves by accessible empty squares of our parity
    - Anti-Egg Walking: Massive penalty for stepping on own eggs (-300)
    - Dead-End Detection: Look 2 steps ahead for parity cul-de-sacs
    - Corner Gravity: Strong pull towards high-value corners (+100 at corner, -5 per step away)
    """
    score = 0.0
    
    # 1. CORE SCORING
    egg_diff = board.chicken_player.eggs_laid - board.chicken_enemy.eggs_laid
    score += 100.0 * egg_diff # Massive weight on egg differential

    my_loc = board.chicken_player.get_location()
    enemy_loc = board.chicken_enemy.get_location()

    # 2. FERTILITY SENSING (NEW!)
    # Don't just count moves. Count "Future Egg Spots".
    # Determine our parity from the chicken's even_chicken attribute
    is_even = board.chicken_player.even_chicken

    # Check accessible neighbors (Fertility)
    fertility_score = 0
    valid_moves = board.get_valid_moves(enemy=False)
    
    for direction, move_type in valid_moves:
        next_loc = loc_after_direction(my_loc, direction)
        
        # Check immediate layability
        dest_parity = (next_loc[0] + next_loc[1]) % 2 == 0
        if dest_parity == is_even and next_loc not in board.eggs_player and next_loc not in board.eggs_enemy:
            fertility_score += 10.0 # High value for immediate egg spots
        
        # Check secondary fertility (neighbors of destination - look ahead 2 steps)
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            n2_x, n2_y = next_loc[0]+dx, next_loc[1]+dy
            if 0 <= n2_x < 8 and 0 <= n2_y < 8:
                n2_parity = (n2_x + n2_y) % 2 == 0
                if n2_parity == is_even and (n2_x, n2_y) not in board.eggs_player and (n2_x, n2_y) not in board.eggs_enemy:
                     fertility_score += 2.0 # Value for future potential
                     
    score += fertility_score

    # 3. ANTI-EGG WALKING (CRITICAL FIX)
    # Severely penalize being on your own egg - THIS IS THE KEY FIX
    if my_loc in board.eggs_player:
        score -= 300.0 # TERRIBLE - this is wasted movement
        
    # 4. STANDARD METRICS
    my_moves = len(valid_moves)
    enemy_moves = len(board.get_valid_moves(enemy=True))
    score += 3.0 * (my_moves - enemy_moves)
    
    turd_diff = board.chicken_enemy.turds_left - board.chicken_player.turds_left
    score += 1.0 * turd_diff

    # 5. SPITE PROTOCOL (AGGRESSION TUNING)
    # Calculate score differential
    my_score = board.chicken_player.eggs_laid
    enemy_score = board.chicken_enemy.eggs_laid
    score_diff = my_score - enemy_score

    turds_used = 5 - board.chicken_player.turds_left

    if score_diff <= 1:
        # WE ARE LOSING, TIED, OR BARELY AHEAD -> WEAPONS FREE
        # Reward using turds to disrupt the board state.
        # +40 points per turd used encourages dropping them to save the game.
        score += turds_used * 40.0
    else:
        # WE ARE WINNING COMFORTABLY -> CONSERVE AMMO
        # Small penalty to prevent wasting them when we have the lead.
        score += turds_used * -20.0

    # Early game inhibition (Don't panic in the first 15 turns)
    if board.turn_count < 15:
        score += turds_used * -200.0

    # 6. SAFETY
    my_risk = evaluator.tracker.get_trapdoor_risk(my_loc)
    score -= 800.0 * my_risk 

    # 7. CORNER GRAVITY (Boosted for guaranteed high scores)
    if evaluator._is_corner(my_loc):
        score += 100.0 # Massive reward for securing a corner (worth 3 eggs!)
    
    # Proximity bonus - strong pull towards corners
    corners = [(0,0), (0,7), (7,0), (7,7)]
    dist_to_corner = min(abs(my_loc[0]-cx) + abs(my_loc[1]-cy) for cx, cy in corners)
    score -= dist_to_corner * 5.0 # Strong pull towards corners (-5 per step away)

    # 8. ANTI-LOOPING (From position history)
    pos_history = getattr(evaluator, '_current_pos_history', [])
    if pos_history and my_loc in pos_history:
        repetition_count = pos_history.count(my_loc)
        score -= repetition_count * 50.0 # Escalating penalty
        
        if repetition_count >= 2:
            score -= 100.0 # Severe loop penalty

    # 9. EGG PLACEMENT INCENTIVE
    if board.can_lay_egg():
        score += 20.0

    return score

