"""Neural policy loader for move ordering"""
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ScubaSteve.model import ChickenNet

class NeuralPolicy:
    def __init__(self):
        self.model = None
        self.device = 'cpu'

    def load(self, model_path):
        """Load trained policy model"""
        try:
            self.model = ChickenNet()
            self.model.load(model_path)
            self.model.eval()
            print(f"[NeuralPolicy] ✓ Loaded from {model_path}")
            return True
        except Exception as e:
            print(f"[NeuralPolicy] Failed to load: {e}")
            return False

    def get_move_probs(self, board_state):
        """Get direction probabilities [UP, RIGHT, DOWN, LEFT]"""
        if self.model is None:
            return [0.25, 0.25, 0.25, 0.25]  # Uniform fallback

        try:
            state_tensor = torch.from_numpy(board_state).unsqueeze(0)
            with torch.no_grad():
                policy_probs, _ = self.model(state_tensor)
                probs = policy_probs[0].cpu().numpy()
            return probs.tolist()
        except:
            return [0.25, 0.25, 0.25, 0.25]

