"""
Component 4: PACE Training Pipeline (Offline RL)

Generates training data and trains the Neural Network evaluation function.
Implements an AlphaZero-style self-play loop:
1. Generate data from RandomAgent vs GreedyAgent
2. Train network on final egg differentials
3. Use NetworkAgent vs NetworkAgent for better data
4. Retrain (iterate)

Usage:
    python pace_training.py --phase 1 --games 10000
    python pace_training.py --phase 2 --epochs 50
    python pace_training.py --phase 3 --games 5000
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from typing import List, Tuple, Dict
from collections import defaultdict
import time


# ============================================================================
# NEURAL NETWORK ARCHITECTURE (PyTorch)
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for CNN"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

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


class ChickenEvaluationNet(nn.Module):
    """
    Residual CNN for board evaluation.

    Input: 7 × 8 × 8 tensor
    Output: Single scalar (expected egg differential)

    Architecture optimized for CPU inference (<0.01s per evaluation)
    """

    def __init__(self):
        super().__init__()

        # Input: 7 channels (chicken positions, eggs, turds, risk map)
        self.res_block1 = ResidualBlock(7, 32)
        self.res_block2 = ResidualBlock(32, 32)
        self.res_block3 = ResidualBlock(32, 16)

        # Flatten: 16 × 8 × 8 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 7, 8, 8)

        Returns:
            Tensor of shape (batch, 1)
        """
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

class ChickenGameDataset(Dataset):
    """Dataset of (board_state, egg_differential) pairs"""

    def __init__(self, data_file: str):
        """
        Load training data from file.

        Format: Each line is a JSON object:
        {
            "board": 7×8×8 tensor (as list),
            "egg_diff": final egg differential,
            "winner": "player" | "enemy" | "tie"
        }
        """
        self.data = []

        with open(data_file, 'r') as f:
            for line in f:
                sample = json.loads(line)

                # Convert board to tensor
                board_tensor = torch.tensor(sample['board'], dtype=torch.float32)

                # Target: egg differential
                egg_diff = float(sample['egg_diff'])

                self.data.append((board_tensor, egg_diff))

        print(f"[Dataset] Loaded {len(self.data)} training samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_training_data(num_games: int, output_file: str, agent1_type: str = 'random',
                          agent2_type: str = 'greedy'):
    """
    Generate training data by playing games.

    Args:
        num_games: Number of games to play
        output_file: File to save data to
        agent1_type: 'random', 'greedy', or 'network'
        agent2_type: 'random', 'greedy', or 'network'
    """
    print(f"[DataGen] Generating {num_games} games: {agent1_type} vs {agent2_type}")
    print(f"[DataGen] This is a placeholder - you need to implement actual game simulation")
    print(f"[DataGen] Use the engine to run games and extract board states")

    # Placeholder: In actual implementation, you would:
    # 1. Import your game engine
    # 2. Create agent instances
    # 3. Play games and record (state, action, outcome) tuples
    # 4. Save to output_file in JSON format

    # Example structure:
    samples = []

    # TODO: Actual game loop here
    # for game_idx in range(num_games):
    #     game_result = play_game(agent1, agent2)
    #     for state in game_result.states:
    #         sample = {
    #             'board': state.to_tensor().tolist(),
    #             'egg_diff': game_result.final_egg_diff,
    #             'winner': game_result.winner
    #         }
    #         samples.append(sample)

    # Save samples
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"[DataGen] Saved {len(samples)} samples to {output_file}")


# ============================================================================
# TRAINING
# ============================================================================

def train_network(data_file: str, output_model: str, epochs: int = 50,
                 batch_size: int = 256, learning_rate: float = 0.001):
    """
    Train the ResNet evaluation network.

    Args:
        data_file: Path to training data (JSONL format)
        output_model: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    print(f"[Training] Loading dataset from {data_file}")

    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"[Training] ERROR: Data file not found: {data_file}")
        print(f"[Training] Run Phase 1 (data generation) first!")
        return

    # Load dataset
    dataset = ChickenGameDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChickenEvaluationNet().to(device)

    print(f"[Training] Using device: {device}")
    print(f"[Training] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5)

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (boards, targets) in enumerate(dataloader):
            boards = boards.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"[Training] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_model)
            print(f"[Training] ✓ Saved best model (loss={best_loss:.4f})")

    print(f"[Training] Training complete! Best loss: {best_loss:.4f}")

    # Convert to NumPy-compatible format
    export_model_to_numpy(model, output_model.replace('.pth', '_numpy.json'))


def export_model_to_numpy(model: nn.Module, output_path: str):
    """
    Export PyTorch model to NumPy-compatible JSON format.
    """
    model.eval()

    weights = {}

    # Export ResBlock1
    weights['res1_conv1_weight'] = model.res_block1.conv1.weight.detach().cpu().numpy().tolist()
    weights['res1_conv1_bias'] = model.res_block1.conv1.bias.detach().cpu().numpy().tolist()
    weights['res1_bn1_weight'] = model.res_block1.bn1.weight.detach().cpu().numpy().tolist()
    weights['res1_bn1_bias'] = model.res_block1.bn1.bias.detach().cpu().numpy().tolist()
    weights['res1_bn1_mean'] = model.res_block1.bn1.running_mean.detach().cpu().numpy().tolist()
    weights['res1_bn1_var'] = model.res_block1.bn1.running_var.detach().cpu().numpy().tolist()
    weights['res1_conv2_weight'] = model.res_block1.conv2.weight.detach().cpu().numpy().tolist()
    weights['res1_conv2_bias'] = model.res_block1.conv2.bias.detach().cpu().numpy().tolist()
    weights['res1_bn2_weight'] = model.res_block1.bn2.weight.detach().cpu().numpy().tolist()
    weights['res1_bn2_bias'] = model.res_block1.bn2.bias.detach().cpu().numpy().tolist()
    weights['res1_bn2_mean'] = model.res_block1.bn2.running_mean.detach().cpu().numpy().tolist()
    weights['res1_bn2_var'] = model.res_block1.bn2.running_var.detach().cpu().numpy().tolist()
    if model.res_block1.skip is not None:
        weights['res1_skip_weight'] = model.res_block1.skip.weight.detach().cpu().numpy().tolist()
        weights['res1_skip_bias'] = model.res_block1.skip.bias.detach().cpu().numpy().tolist()

    # Export ResBlock2
    weights['res2_conv1_weight'] = model.res_block2.conv1.weight.detach().cpu().numpy().tolist()
    weights['res2_conv1_bias'] = model.res_block2.conv1.bias.detach().cpu().numpy().tolist()
    weights['res2_bn1_weight'] = model.res_block2.bn1.weight.detach().cpu().numpy().tolist()
    weights['res2_bn1_bias'] = model.res_block2.bn1.bias.detach().cpu().numpy().tolist()
    weights['res2_bn1_mean'] = model.res_block2.bn1.running_mean.detach().cpu().numpy().tolist()
    weights['res2_bn1_var'] = model.res_block2.bn1.running_var.detach().cpu().numpy().tolist()
    weights['res2_conv2_weight'] = model.res_block2.conv2.weight.detach().cpu().numpy().tolist()
    weights['res2_conv2_bias'] = model.res_block2.conv2.bias.detach().cpu().numpy().tolist()
    weights['res2_bn2_weight'] = model.res_block2.bn2.weight.detach().cpu().numpy().tolist()
    weights['res2_bn2_bias'] = model.res_block2.bn2.bias.detach().cpu().numpy().tolist()
    weights['res2_bn2_mean'] = model.res_block2.bn2.running_mean.detach().cpu().numpy().tolist()
    weights['res2_bn2_var'] = model.res_block2.bn2.running_var.detach().cpu().numpy().tolist()

    # Export ResBlock3
    weights['res3_conv1_weight'] = model.res_block3.conv1.weight.detach().cpu().numpy().tolist()
    weights['res3_conv1_bias'] = model.res_block3.conv1.bias.detach().cpu().numpy().tolist()
    weights['res3_bn1_weight'] = model.res_block3.bn1.weight.detach().cpu().numpy().tolist()
    weights['res3_bn1_bias'] = model.res_block3.bn1.bias.detach().cpu().numpy().tolist()
    weights['res3_bn1_mean'] = model.res_block3.bn1.running_mean.detach().cpu().numpy().tolist()
    weights['res3_bn1_var'] = model.res_block3.bn1.running_var.detach().cpu().numpy().tolist()
    weights['res3_conv2_weight'] = model.res_block3.conv2.weight.detach().cpu().numpy().tolist()
    weights['res3_conv2_bias'] = model.res_block3.conv2.bias.detach().cpu().numpy().tolist()
    weights['res3_bn2_weight'] = model.res_block3.bn2.weight.detach().cpu().numpy().tolist()
    weights['res3_bn2_bias'] = model.res_block3.bn2.bias.detach().cpu().numpy().tolist()
    weights['res3_bn2_mean'] = model.res_block3.bn2.running_mean.detach().cpu().numpy().tolist()
    weights['res3_bn2_var'] = model.res_block3.bn2.running_var.detach().cpu().numpy().tolist()
    if model.res_block3.skip is not None:
        weights['res3_skip_weight'] = model.res_block3.skip.weight.detach().cpu().numpy().tolist()
        weights['res3_skip_bias'] = model.res_block3.skip.bias.detach().cpu().numpy().tolist()

    # Export FC layers
    weights['fc1_weight'] = model.fc1.weight.detach().cpu().numpy().tolist()
    weights['fc1_bias'] = model.fc1.bias.detach().cpu().numpy().tolist()
    weights['fc2_weight'] = model.fc2.weight.detach().cpu().numpy().tolist()
    weights['fc2_bias'] = model.fc2.bias.detach().cpu().numpy().tolist()

    # Save
    with open(output_path, 'w') as f:
        json.dump(weights, f)

    print(f"[Export] ✓ Model exported to {output_path}")


# ============================================================================
# PACE TRAINING SCRIPT
# ============================================================================

def phase1_generate_data(num_games: int):
    """
    Phase 1: Generate initial training data.
    Play RandomAgent vs GreedyAgent for N games.
    """
    print("=" * 60)
    print("PHASE 1: DATA GENERATION")
    print("=" * 60)

    output_file = f"training_data_phase1_{num_games}.jsonl"
    generate_training_data(num_games, output_file, 'random', 'greedy')

    print(f"\n[Phase 1] Complete! Data saved to {output_file}")
    print(f"[Phase 1] Next: Run Phase 2 to train the network")


def phase2_train_network(data_file: str, epochs: int = 50):
    """
    Phase 2: Train network on generated data.
    """
    print("=" * 60)
    print("PHASE 2: NETWORK TRAINING")
    print("=" * 60)

    output_model = "chicken_eval_model.pth"
    output_numpy = "chicken_eval_model_numpy.json"

    train_network(data_file, output_model, epochs)

    print(f"\n[Phase 2] Complete!")
    print(f"[Phase 2] Model saved to: {output_model}")
    print(f"[Phase 2] NumPy export: {output_numpy}")
    print(f"[Phase 2] Next: Run Phase 3 for self-play refinement")


def phase3_self_play(num_games: int, model_path: str):
    """
    Phase 3: Self-play to generate higher-quality data.
    NetworkAgent vs NetworkAgent (AlphaZero loop).
    """
    print("=" * 60)
    print("PHASE 3: SELF-PLAY REFINEMENT")
    print("=" * 60)

    output_file = f"training_data_phase3_{num_games}.jsonl"

    print(f"[Phase 3] Playing {num_games} self-play games...")
    print(f"[Phase 3] Using model: {model_path}")

    # TODO: Implement actual self-play using NetworkAgent
    generate_training_data(num_games, output_file, 'network', 'network')

    print(f"\n[Phase 3] Complete! Data saved to {output_file}")
    print(f"[Phase 3] Next: Retrain with Phase 2 using this new data")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PACE Training Pipeline')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3],
                       help='Training phase: 1=DataGen, 2=Train, 3=SelfPlay')
    parser.add_argument('--games', type=int, default=10000,
                       help='Number of games to generate (Phase 1 and 3)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (Phase 2)')
    parser.add_argument('--data', type=str, default=None,
                       help='Training data file (Phase 2)')
    parser.add_argument('--model', type=str, default='chicken_eval_model.pth',
                       help='Model file (Phase 3)')

    args = parser.parse_args()

    if args.phase == 1:
        phase1_generate_data(args.games)

    elif args.phase == 2:
        if args.data is None:
            print("[ERROR] Phase 2 requires --data argument")
            return
        phase2_train_network(args.data, args.epochs)

    elif args.phase == 3:
        phase3_self_play(args.games, args.model)


if __name__ == '__main__':
    main()

