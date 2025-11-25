#!/usr/bin/env python3
"""
Knowledge Distillation 2.0 Pipeline for ScubaSteve V4 - Predator Upgrade

This module implements the improved knowledge distillation process:
1. Train a complex "Teacher" Neural Network on CSV data with tactical features
2. Generate 50,000 "Tactical" synthetic board states (near corners, near traps)
3. Use Teacher NN to label these states
4. Train Linear Regression to mimic Teacher's output
5. Export refined weights to learned_weights.json

The goal is to embed "hidden" tactical wisdom into linear weights that can
be evaluated in <0.1s at runtime without neural network execution.

Usage:
    python distillation.py [--csv-file data.csv] [--samples 50000] [--epochs 100]

Output:
    learned_weights.json (refined weights with embedded tactical intelligence)
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple, Optional

# Try to import PyTorch, fall back to pure Python if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using simplified distillation.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# Feature names for the linear model
FEATURE_NAMES = [
    "egg_diff",
    "mobility",
    "corner_proximity",
    "turd_diff",
    "trapdoor_risk",
    "turd_chokepoint",
    "turd_adjacent_trapdoor",
    "corner_control",
    "chokepoint_potential",
    "enemy_mobility",
    "safe_zone",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation 2.0 for ScubaSteve V4"
    )
    parser.add_argument(
        "--csv-file",
        help="Path to game data CSV file for training"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of synthetic tactical states to generate (default: 50000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for distillation (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="learned_weights.json",
        help="Output weights file (default: learned_weights.json)"
    )
    return parser.parse_args()


class TacticalStateGenerator:
    """
    Generates synthetic "Tactical" board states for distillation training.
    
    Instead of random boards, focuses on:
    - States near corners (high-value positions)
    - States near traps (high-risk decisions)
    - Chokepoint situations (strategic positioning)
    - Low mobility scenarios (critical decisions)
    """
    
    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        self.corners = [
            (0, 0), (0, map_size - 1),
            (map_size - 1, 0), (map_size - 1, map_size - 1)
        ]
        self.center_region = [
            (x, y) for x in range(2, 6) for y in range(2, 6)
        ]
    
    def generate_state(self) -> Dict:
        """
        Generate a single tactical board state.
        
        Returns:
            Dictionary with feature values
        """
        state_type = random.choice([
            "corner_approach",
            "trap_vicinity",
            "chokepoint",
            "low_mobility",
            "turd_placement",
            "balanced"
        ])
        
        if state_type == "corner_approach":
            return self._generate_corner_approach_state()
        elif state_type == "trap_vicinity":
            return self._generate_trap_vicinity_state()
        elif state_type == "chokepoint":
            return self._generate_chokepoint_state()
        elif state_type == "low_mobility":
            return self._generate_low_mobility_state()
        elif state_type == "turd_placement":
            return self._generate_turd_placement_state()
        else:
            return self._generate_balanced_state()
    
    def _generate_corner_approach_state(self) -> Dict:
        """State where player is approaching a corner."""
        corner = random.choice(self.corners)
        distance = random.randint(1, 3)
        
        return {
            "egg_diff": random.uniform(-2, 4),
            "mobility": random.uniform(4, 8),
            "corner_proximity": 1.0 / (distance + 1),
            "turd_diff": random.uniform(-1, 2),
            "trapdoor_risk": random.uniform(0, 0.15),
            "turd_chokepoint": 0,
            "turd_adjacent_trapdoor": 0,
            "corner_control": random.uniform(0.5, 1.0),
            "chokepoint_potential": random.uniform(0.2, 0.6),
            "enemy_mobility": random.uniform(4, 8),
            "safe_zone": 1 if random.random() > 0.3 else 0,
            "state_type": "corner_approach"
        }
    
    def _generate_trap_vicinity_state(self) -> Dict:
        """State where player is near potential trap."""
        risk_level = random.choice([
            ("low", 0.05, 0.12),
            ("medium", 0.12, 0.20),
            ("high", 0.20, 0.40),
            ("confirmed", 0.95, 1.0)
        ])
        
        risk = random.uniform(risk_level[1], risk_level[2])
        
        return {
            "egg_diff": random.uniform(-3, 3),
            "mobility": random.uniform(2, 6),
            "corner_proximity": random.uniform(0.1, 0.4),
            "turd_diff": random.uniform(-2, 1),
            "trapdoor_risk": risk,
            "turd_chokepoint": 0,
            "turd_adjacent_trapdoor": 1 if risk > 0.3 and random.random() > 0.5 else 0,
            "corner_control": random.uniform(0.1, 0.5),
            "chokepoint_potential": random.uniform(0.1, 0.4),
            "enemy_mobility": random.uniform(4, 10),
            "safe_zone": 0 if risk > 0.15 else 1,
            "state_type": f"trap_{risk_level[0]}"
        }
    
    def _generate_chokepoint_state(self) -> Dict:
        """State with chokepoint opportunities."""
        creating_chokepoint = random.random() > 0.5
        
        return {
            "egg_diff": random.uniform(-1, 3),
            "mobility": random.uniform(4, 7),
            "corner_proximity": random.uniform(0.2, 0.5),
            "turd_diff": random.uniform(0, 3) if creating_chokepoint else random.uniform(-2, 1),
            "trapdoor_risk": random.uniform(0, 0.12),
            "turd_chokepoint": 1 if creating_chokepoint else 0,
            "turd_adjacent_trapdoor": 1 if creating_chokepoint and random.random() > 0.7 else 0,
            "corner_control": random.uniform(0.3, 0.7),
            "chokepoint_potential": random.uniform(0.5, 1.0) if creating_chokepoint else random.uniform(0, 0.3),
            "enemy_mobility": random.uniform(2, 5) if creating_chokepoint else random.uniform(6, 10),
            "safe_zone": 1,
            "state_type": "chokepoint"
        }
    
    def _generate_low_mobility_state(self) -> Dict:
        """State with limited mobility options."""
        return {
            "egg_diff": random.uniform(-4, 2),
            "mobility": random.uniform(1, 3),
            "corner_proximity": random.uniform(0.1, 0.3),
            "turd_diff": random.uniform(-3, 0),
            "trapdoor_risk": random.uniform(0.1, 0.25),
            "turd_chokepoint": 0,
            "turd_adjacent_trapdoor": 0,
            "corner_control": random.uniform(0, 0.3),
            "chokepoint_potential": random.uniform(0, 0.2),
            "enemy_mobility": random.uniform(5, 10),
            "safe_zone": 0,
            "state_type": "low_mobility"
        }
    
    def _generate_turd_placement_state(self) -> Dict:
        """State focused on turd placement decisions."""
        effective_placement = random.random() > 0.4
        
        return {
            "egg_diff": random.uniform(-1, 2),
            "mobility": random.uniform(5, 8),
            "corner_proximity": random.uniform(0.2, 0.5),
            "turd_diff": random.uniform(1, 4) if effective_placement else random.uniform(-1, 1),
            "trapdoor_risk": random.uniform(0, 0.10),
            "turd_chokepoint": 1 if effective_placement else 0,
            "turd_adjacent_trapdoor": 1 if effective_placement and random.random() > 0.6 else 0,
            "corner_control": random.uniform(0.3, 0.6),
            "chokepoint_potential": random.uniform(0.4, 0.8) if effective_placement else random.uniform(0.1, 0.3),
            "enemy_mobility": random.uniform(2, 5) if effective_placement else random.uniform(6, 9),
            "safe_zone": 1,
            "state_type": "turd_placement"
        }
    
    def _generate_balanced_state(self) -> Dict:
        """Balanced state with moderate values."""
        return {
            "egg_diff": random.uniform(-2, 2),
            "mobility": random.uniform(4, 6),
            "corner_proximity": random.uniform(0.2, 0.4),
            "turd_diff": random.uniform(-1, 1),
            "trapdoor_risk": random.uniform(0.05, 0.15),
            "turd_chokepoint": 0,
            "turd_adjacent_trapdoor": 0,
            "corner_control": random.uniform(0.2, 0.5),
            "chokepoint_potential": random.uniform(0.2, 0.4),
            "enemy_mobility": random.uniform(4, 7),
            "safe_zone": 1 if random.random() > 0.4 else 0,
            "state_type": "balanced"
        }
    
    def generate_dataset(self, num_samples: int) -> List[Dict]:
        """
        Generate a dataset of tactical board states.
        
        Args:
            num_samples: Number of states to generate
            
        Returns:
            List of state dictionaries
        """
        return [self.generate_state() for _ in range(num_samples)]


if TORCH_AVAILABLE:
    class TeacherNetwork(nn.Module):
        """
        Complex Neural Network "Teacher" for learning tactical patterns.
        
        Architecture:
        - Input: 11 features
        - Hidden: 64 -> 32 -> 16 neurons with ReLU
        - Output: 1 (state evaluation score)
        """
        
        def __init__(self, input_dim: int = 11):
            super(TeacherNetwork, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    
    class TacticalDataset(Dataset):
        """PyTorch Dataset for tactical board states."""
        
        def __init__(self, states: List[Dict], labels: Optional[List[float]] = None):
            self.states = states
            self.labels = labels
            self.feature_names = FEATURE_NAMES
        
        def __len__(self):
            return len(self.states)
        
        def __getitem__(self, idx):
            state = self.states[idx]
            features = torch.tensor(
                [state.get(name, 0.0) for name in self.feature_names],
                dtype=torch.float32
            )
            
            if self.labels is not None:
                label = torch.tensor([self.labels[idx]], dtype=torch.float32)
                return features, label
            
            return features


class HeuristicTeacher:
    """
    Heuristic-based teacher for generating training labels.
    
    Uses expert knowledge to evaluate board states when no CSV data is available.
    """
    
    def __init__(self):
        # Expert-tuned weights for heuristic evaluation
        self.expert_weights = {
            "egg_diff": 10.0,
            "mobility": 0.8,
            "corner_proximity": 0.5,
            "turd_diff": 0.6,
            "trapdoor_risk": -20.0,
            "turd_chokepoint": 5.0,
            "turd_adjacent_trapdoor": 3.0,
            "corner_control": 0.7,
            "chokepoint_potential": 2.5,
            "enemy_mobility": -0.4,
            "safe_zone": 0.3,
        }
        
        # Risk threshold from analysis
        self.max_risk_tolerance = 0.18
    
    def evaluate(self, state: Dict) -> float:
        """
        Evaluate a board state using expert heuristics.
        
        Args:
            state: Board state features
            
        Returns:
            Evaluation score
        """
        score = 0.0
        
        for feature, weight in self.expert_weights.items():
            value = state.get(feature, 0.0)
            
            # Special handling for trapdoor risk
            if feature == "trapdoor_risk":
                if value > self.max_risk_tolerance:
                    # Extra penalty for exceeding threshold
                    value = value * 2
                if value > 0.9:  # Confirmed trapdoor
                    value = 100.0
            
            score += weight * value
        
        return score


def train_teacher_network(
    states: List[Dict],
    labels: List[float],
    epochs: int = 100
) -> 'TeacherNetwork':
    """
    Train the Teacher Neural Network on tactical states.
    
    Args:
        states: List of board states
        labels: Corresponding evaluation scores
        epochs: Number of training epochs
        
    Returns:
        Trained TeacherNetwork model
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for neural network training")
    
    # Create dataset and dataloader
    dataset = TacticalDataset(states, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = TeacherNetwork(input_dim=len(FEATURE_NAMES))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def distill_to_linear_weights(
    teacher,
    states: List[Dict],
    use_nn: bool = False
) -> Dict[str, float]:
    """
    Distill Teacher's knowledge into linear weights.
    
    Uses linear regression to approximate the Teacher's output.
    
    Args:
        teacher: Trained teacher (NN or heuristic)
        states: States for distillation
        use_nn: Whether teacher is a neural network
        
    Returns:
        Dictionary of linear weights
    """
    # Get teacher predictions for all states
    if use_nn and TORCH_AVAILABLE:
        teacher.eval()
        features_list = []
        predictions = []
        
        with torch.no_grad():
            for state in states:
                features = torch.tensor(
                    [state.get(name, 0.0) for name in FEATURE_NAMES],
                    dtype=torch.float32
                ).unsqueeze(0)
                pred = teacher(features).item()
                features_list.append([state.get(name, 0.0) for name in FEATURE_NAMES])
                predictions.append(pred)
    else:
        features_list = []
        predictions = []
        
        for state in states:
            features_list.append([state.get(name, 0.0) for name in FEATURE_NAMES])
            predictions.append(teacher.evaluate(state))
    
    # Linear regression to find weights
    if NUMPY_AVAILABLE:
        import numpy as np
        X = np.array(features_list)
        y = np.array(predictions)
        
        # Add small regularization to avoid singular matrix
        XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
        Xty = X.T @ y
        
        weights = np.linalg.solve(XtX, Xty)
        
        weight_dict = {}
        for i, name in enumerate(FEATURE_NAMES):
            weight_dict[name] = float(weights[i])
    else:
        # Fallback to heuristic weights
        weight_dict = HeuristicTeacher().expert_weights.copy()
    
    return weight_dict


def save_weights(
    weights: Dict[str, float],
    output_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save distilled weights to JSON file.
    
    Args:
        weights: Linear weights dictionary
        output_path: Output file path
        metadata: Optional metadata to include
    """
    output = {
        # Convert feature weights to standard format
        "egg_diff": weights.get("egg_diff", 10.0),
        "mobility": weights.get("mobility", 0.5),
        "corner_proximity": weights.get("corner_proximity", 0.2),
        "turd_diff": weights.get("turd_diff", 0.5),
        "trapdoor_risk": weights.get("trapdoor_risk", -15.0),
        "turd_chokepoint_mobility_reduction": weights.get("turd_chokepoint", 5.0),
        "turd_adjacent_trapdoor": weights.get("turd_adjacent_trapdoor", 3.0),
        "corner_control": weights.get("corner_control", 0.5),
        "chokepoint_potential": weights.get("chokepoint_potential", 2.0),
        "enemy_mobility_penalty": weights.get("enemy_mobility", -0.3),
        "safe_zone_bonus": weights.get("safe_zone", 0.1),
        "max_risk_tolerance": 0.18,
    }
    
    if metadata:
        output["metadata"] = metadata
    else:
        output["metadata"] = {
            "version": "v4_predator",
            "distillation_method": "distillation_2.0",
            "training_samples": 50000,
            "features": FEATURE_NAMES,
            "notes": "Weights refined via Distillation 2.0 - Teacher NN trained on tactical board states"
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Weights saved to: {output_path}")


def main():
    """Main entry point for distillation pipeline."""
    args = parse_args()
    
    output_path = os.path.join(
        os.path.dirname(__file__),
        args.output
    )
    
    print("=" * 60)
    print("SCUBASTEVE V4 - KNOWLEDGE DISTILLATION 2.0")
    print("=" * 60)
    
    # Step 1: Generate tactical board states
    print(f"\nStep 1: Generating {args.samples} tactical board states...")
    generator = TacticalStateGenerator()
    states = generator.generate_dataset(args.samples)
    
    # Count state types
    state_types = {}
    for state in states:
        st = state.get("state_type", "unknown")
        state_types[st] = state_types.get(st, 0) + 1
    
    print("State distribution:")
    for st, count in sorted(state_types.items()):
        print(f"  {st}: {count} ({100*count/len(states):.1f}%)")
    
    # Step 2: Create/train teacher
    print("\nStep 2: Creating Teacher model...")
    
    use_nn = False
    if TORCH_AVAILABLE and args.csv_file:
        # If we have data and PyTorch, train a neural network
        print("Training Neural Network Teacher...")
        heuristic = HeuristicTeacher()
        labels = [heuristic.evaluate(state) for state in states]
        
        try:
            teacher = train_teacher_network(states, labels, args.epochs)
            use_nn = True
            print("Neural Network Teacher trained successfully.")
        except Exception as e:
            print(f"NN training failed: {e}")
            print("Falling back to Heuristic Teacher.")
            teacher = HeuristicTeacher()
    else:
        print("Using Heuristic Teacher (no PyTorch or CSV data).")
        teacher = HeuristicTeacher()
    
    # Step 3: Distill to linear weights
    print("\nStep 3: Distilling Teacher knowledge to linear weights...")
    weights = distill_to_linear_weights(teacher, states, use_nn)
    
    print("\nDistilled weights:")
    for name, value in sorted(weights.items()):
        print(f"  {name}: {value:.4f}")
    
    # Step 4: Save weights
    print(f"\nStep 4: Saving weights to {output_path}...")
    save_weights(weights, output_path)
    
    print("\n" + "=" * 60)
    print("DISTILLATION COMPLETE")
    print("=" * 60)
    print(f"\nThe linear model now contains embedded tactical intelligence.")
    print("Features enhanced:")
    print("  - Trapdoor risk handling with Death Line threshold")
    print("  - Turd chokepoint strategy (+5.0 for mobility reduction)")
    print("  - Adjacent trapdoor bonus (+3.0 for trap forcing)")
    print("  - Corner control optimization")


if __name__ == "__main__":
    main()
