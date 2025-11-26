"""
LEAD ARCHITECT DIRECTIVE: Curriculum Learning Training Pipeline

Phase 1: Train with TrapdoorProb = 0 (teach egg maximization and mobility)
Phase 2: Train with TrapdoorProb = 1.0 (teach trap avoidance penalty)
Phase 3: Self-Play (force loop breaking and strategic depth)

Key Innovation: Pass trapdoor probability maps as input channels to the NN
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Dict
import time
from datetime import datetime
import random


# ============================================================================
# NEURAL NETWORK WITH TRAP PROBABILITY CHANNELS
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for CNN"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.relu(out)

        return out


class EnhancedChickenEvaluationNet(nn.Module):
    """
    Enhanced ResNet with trap probability channels.

    Input: 9 channels (was 7):
    0: My Chicken
    1: Enemy Chicken
    2: My Eggs
    3: Enemy Eggs
    4: My Turds
    5: Enemy Turds
    6: Even Trapdoor Probabilities (NEW - from TrapdoorTracker)
    7: Odd Trapdoor Probabilities (NEW - from TrapdoorTracker)
    8: Safe Squares (1 where we've stepped and survived)
    """

    def __init__(self):
        super().__init__()

        # Input: 9 channels
        self.res_block1 = ResidualBlock(9, 32)
        self.res_block2 = ResidualBlock(32, 32)
        self.res_block3 = ResidualBlock(32, 16)

        # Flatten: 16 × 8 × 8 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ============================================================================
# DATASET
# ============================================================================

class CurriculumDataset(Dataset):
    """Dataset with trap probability channels"""

    def __init__(self, data_file: str):
        self.data = []

        print(f"[Dataset] Loading from {data_file}...")
        with open(data_file, 'r') as f:
            for line in f:
                sample = json.loads(line)

                # Convert to tensor (expecting 9 channels now)
                board_tensor = torch.tensor(sample['board'], dtype=torch.float32)

                # Ensure we have 9 channels
                if board_tensor.shape[0] == 7:
                    # Old format - add two zero channels for trap probabilities
                    zeros = torch.zeros(2, 8, 8, dtype=torch.float32)
                    board_tensor = torch.cat([board_tensor, zeros], dim=0)

                egg_diff = float(sample['egg_diff'])

                self.data.append((board_tensor, egg_diff))

        print(f"[Dataset] Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# CURRICULUM TRAINING
# ============================================================================

def train_phase1_no_traps(output_dir: str, epochs: int = 30):
    """
    Phase 1: Train with TrapdoorProb = 0
    Teaches: Egg maximization, mobility, corner control
    """
    print("=" * 70)
    print("PHASE 1: TRAINING WITHOUT TRAPS")
    print("Goal: Learn egg maximization and mobility")
    print("=" * 70)

    # Generate data without traps
    print("\n[Phase 1] Generating training data (no traps)...")
    # TODO: Call data generation with trap_prob=0.0

    print("[Phase 1] Training not implemented yet - needs data generation")
    print("[Phase 1] This would teach basic egg-laying strategy")


def train_phase2_deadly_traps(output_dir: str, epochs: int = 30):
    """
    Phase 2: Train with TrapdoorProb = 1.0
    Teaches: Trap avoidance, risk assessment
    """
    print("=" * 70)
    print("PHASE 2: TRAINING WITH DEADLY TRAPS")
    print("Goal: Learn trap avoidance penalty")
    print("=" * 70)

    print("\n[Phase 2] This teaches the agent that high-risk squares = death")
    print("[Phase 2] Training not implemented yet - needs data generation")


def train_phase3_selfplay(model_path: str, output_dir: str, games: int = 1000):
    """
    Phase 3: Self-Play Training
    Teaches: Loop breaking, strategic asymmetry

    Key: When both agents loop, game is a draw (low reward).
    Forces agent to learn to break symmetry and create winning positions.
    """
    print("=" * 70)
    print("PHASE 3: SELF-PLAY TRAINING")
    print("Goal: Break loops and develop strategic depth")
    print("=" * 70)

    print("\n[Phase 3] Self-play forces loop-breaking behavior")
    print(f"[Phase 3] Will play {games} games of Current_Best vs Current_Best")
    print("[Phase 3] Implementation needed")

    # Load current best model
    # Play games against itself
    # Retrain on new data where loops = draws (low reward)
    # This creates evolutionary pressure to avoid loops


# ============================================================================
# MAIN TRAINING ORCHESTRATOR
# ============================================================================

def run_curriculum_training(output_dir: str = '3600-agents/ScubaSteve'):
    """
    Run the complete curriculum training pipeline
    """
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CURRICULUM LEARNING PIPELINE" + " " * 25 + "║")
    print("║" + " " * 15 + "Lead Architect Directive" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")

    print("\nTraining Strategy:")
    print("  Phase 1: No traps → Learn egg maximization")
    print("  Phase 2: Deadly traps → Learn risk avoidance")
    print("  Phase 3: Self-play → Learn loop breaking")

    print("\n" + "=" * 70)
    print("CRITICAL REQUIREMENT:")
    print("  ✓ Input channels MUST include trap probability maps")
    print("  ✓ Network cannot deduce probability from raw board state")
    print("  ✓ Must feed calculated risk maps as input")
    print("=" * 70)

    # Phase 1
    print("\n[PHASE 1] Starting no-trap training...")
    train_phase1_no_traps(output_dir, epochs=30)

    # Phase 2
    print("\n[PHASE 2] Starting deadly-trap training...")
    train_phase2_deadly_traps(output_dir, epochs=30)

    # Phase 3
    print("\n[PHASE 3] Starting self-play training...")
    train_phase3_selfplay(
        model_path=os.path.join(output_dir, 'phase2_model.pth'),
        output_dir=output_dir,
        games=1000
    )

    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("LEAD ARCHITECT TRAINING DIRECTIVE")
    print("=" * 70)
    print("\nThis implements:")
    print("  1. Curriculum Learning (Phase 1 → 2 → 3)")
    print("  2. Trap probability channels as input")
    print("  3. Self-play for loop breaking")
    print("\nNOTE: Full implementation requires data generation integration")
    print("=" * 70 + "\n")

    run_curriculum_training()

