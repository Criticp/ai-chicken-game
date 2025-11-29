"""
Turd Analyzer - Minimax-based Turd Evaluation Module

This module provides a dedicated engine for evaluating the strategic value of
placing a turd. It uses a limited-depth minimax search to simulate the
consequences of a turd placement and determine its impact on the game,
comparing it against other move options like laying an egg or moving plainly.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .search_engine import SearchEngine
    from game.board import Board
    from game.enums import MoveType, Direction


class TurdAnalyzer:
    """
    Analyzes the value of placing a turd by simulating future outcomes.
    """

    def __init__(self, search_engine: SearchEngine, evaluator: callable):
        """
        Initializes the TurdAnalyzer.

        Args:
            search_engine: The main search engine instance, used for its utilities.
            evaluator: The hybrid board evaluation function.
        """
        self.search_engine = search_engine
        self.evaluator = evaluator
        self.simulation_depth = 3  # How many moves ahead to look (3 moves = 1.5 turns for each player)

    def evaluate_turd_placement(
        self,
        board: Board,
        turd_move: Tuple[Direction, MoveType],
        deadline: float
    ) -> float:
        """
        Calculates the strategic value of a specific turd move.

        It simulates the turd placement and then runs a minimax search to see
        how the game state evolves.

        Args:
            board: The current board state.
            turd_move: The turd move to evaluate.
            deadline: The time limit for the evaluation.

        Returns:
            The evaluated score of the board state after the turd placement
            and subsequent simulated moves.
        """
        # Create a copy of the board to simulate the turd move
        sim_board = self.search_engine._copy_board(board)
        self.search_engine._apply_move(sim_board, turd_move, enemy=False)

        # Now, run a minimax search from this new board state
        # We start with the opponent's turn (maximizing=False)
        score = self.search_engine._negamax(
            board=sim_board,
            depth=self.simulation_depth,
            alpha=-float('inf'),
            beta=float('inf'),
            is_player_turn=False,  # It's enemy's turn after our move
            deadline=deadline
        )

        # The score from negamax is from the perspective of the player whose turn it is.
        # Since we are evaluating the position for our agent after the enemy has moved,
        # the returned score is what we want.
        return score

    def compare_actions(
        self,
        board: Board,
        deadline: float
    ) -> dict[str, float]:
        """
        Compares the expected outcomes of laying a turd, laying an egg, or making a plain move.

        This helps the agent decide on the best type of action in a given turn.

        Returns:
            A dictionary with the best scores for 'turd', 'egg', and 'plain' moves.
        """
        valid_moves = board.get_valid_moves(enemy=False)
        action_scores = {
            'turd': -float('inf'),
            'egg': -float('inf'),
            'plain': -float('inf'),
        }

        for move in valid_moves:
            _, move_type = move
            sim_board = self.search_engine._copy_board(board)
            self.search_engine._apply_move(sim_board, move, enemy=False)

            score = -self.search_engine._negamax(
                board=sim_board,
                depth=self.simulation_depth,
                alpha=-float('inf'),
                beta=float('inf'),
                is_player_turn=False,
                deadline=deadline
            )

            if move_type == MoveType.TURD:
                if score > action_scores['turd']:
                    action_scores['turd'] = score
            elif move_type == MoveType.EGG:
                if score > action_scores['egg']:
                    action_scores['egg'] = score
            elif move_type == MoveType.PLAIN:
                if score > action_scores['plain']:
                    action_scores['plain'] = score

        return action_scores

