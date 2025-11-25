#!/usr/bin/env python3
"""
Risk Analysis Script for ScubaSteve V4 - Predator Upgrade

Analyzes game CSV data to determine optimal risk tolerance threshold.
Calculates Win Rate vs. Max Risk Taken for top players to find the "Death Line".

Usage:
    python analyze_risk.py <csv_file> [--top-n 1000] [--output report.txt]

Output:
    risk_analysis_report.txt with calculated optimal risk threshold
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze game CSV data to find optimal risk tolerance threshold"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to the game data CSV file"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Number of top players to analyze (default: 1000)"
    )
    parser.add_argument(
        "--output",
        default="risk_analysis_report.txt",
        help="Output report filename (default: risk_analysis_report.txt)"
    )
    return parser.parse_args()


def load_csv_data(csv_path: str) -> List[Dict]:
    """
    Load and parse the game CSV data.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of game records as dictionaries
    """
    records = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    except csv.Error as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    return records


def analyze_player_risk(records: List[Dict], top_n: int) -> Dict:
    """
    Analyze risk-taking behavior vs win rate for top players.
    
    Args:
        records: List of game records
        top_n: Number of top players to analyze
        
    Returns:
        Analysis results dictionary
    """
    # Track player stats: wins, losses, max_risk_taken per game
    player_stats = defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "games": 0,
        "max_risks": [],
        "death_events": 0
    })
    
    # Process records (assuming CSV has player_id, outcome, risk_taken, etc.)
    for record in records:
        # Extract relevant fields (adjust based on actual CSV structure)
        player_id = record.get("player_id", record.get("player", "unknown"))
        outcome = record.get("outcome", record.get("result", ""))
        
        # Risk might be encoded as probability or count
        risk = float(record.get("max_risk", record.get("risk_taken", 0.0)))
        died = record.get("died", record.get("death", "false")).lower() == "true"
        
        stats = player_stats[player_id]
        stats["games"] += 1
        stats["max_risks"].append(risk)
        
        if died:
            stats["death_events"] += 1
        
        if outcome.lower() in ["win", "1", "true", "player"]:
            stats["wins"] += 1
        elif outcome.lower() in ["loss", "0", "false", "enemy", "lose"]:
            stats["losses"] += 1
    
    # Sort players by win rate and select top N
    player_list = []
    for player_id, stats in player_stats.items():
        if stats["games"] >= 5:  # Minimum games threshold
            win_rate = stats["wins"] / stats["games"] if stats["games"] > 0 else 0
            avg_risk = sum(stats["max_risks"]) / len(stats["max_risks"]) if stats["max_risks"] else 0
            max_risk = max(stats["max_risks"]) if stats["max_risks"] else 0
            
            player_list.append({
                "player_id": player_id,
                "win_rate": win_rate,
                "games": stats["games"],
                "avg_risk": avg_risk,
                "max_risk": max_risk,
                "death_events": stats["death_events"]
            })
    
    # Sort by win rate descending
    player_list.sort(key=lambda x: x["win_rate"], reverse=True)
    top_players = player_list[:top_n]
    
    return {
        "total_players": len(player_stats),
        "analyzed_players": len(top_players),
        "top_players": top_players
    }


def find_death_line(analysis: Dict) -> Tuple[float, float]:
    """
    Find the optimal risk threshold (Death Line) from analysis.
    
    Logic: Find the risk level where win rate drops significantly.
    "Players who step on >X% risk squares lose Y% of the time"
    
    Args:
        analysis: Analysis results from analyze_player_risk
        
    Returns:
        Tuple of (death_line_threshold, loss_rate_above_threshold)
    """
    top_players = analysis["top_players"]
    
    if not top_players:
        # Default threshold if no data
        return (0.18, 0.85)
    
    # Calculate win rate at different risk thresholds
    risk_thresholds = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    threshold_analysis = []
    
    for threshold in risk_thresholds:
        high_risk_players = [p for p in top_players if p["max_risk"] > threshold]
        low_risk_players = [p for p in top_players if p["max_risk"] <= threshold]
        
        if high_risk_players:
            high_risk_win_rate = sum(p["win_rate"] for p in high_risk_players) / len(high_risk_players)
        else:
            high_risk_win_rate = 0
        
        if low_risk_players:
            low_risk_win_rate = sum(p["win_rate"] for p in low_risk_players) / len(low_risk_players)
        else:
            low_risk_win_rate = 0
        
        threshold_analysis.append({
            "threshold": threshold,
            "high_risk_count": len(high_risk_players),
            "low_risk_count": len(low_risk_players),
            "high_risk_win_rate": high_risk_win_rate,
            "low_risk_win_rate": low_risk_win_rate,
            "win_rate_diff": low_risk_win_rate - high_risk_win_rate
        })
    
    # Find threshold with maximum win rate difference
    best = max(threshold_analysis, key=lambda x: x["win_rate_diff"])
    
    death_line = best["threshold"]
    loss_rate = 1.0 - best["high_risk_win_rate"]
    
    return (death_line, loss_rate)


def generate_report(
    analysis: Dict,
    death_line: float,
    loss_rate: float,
    output_path: str
) -> str:
    """
    Generate the risk analysis report.
    
    Args:
        analysis: Analysis results
        death_line: Calculated death line threshold
        loss_rate: Loss rate above death line
        output_path: Path to output file
        
    Returns:
        Report content string
    """
    report_lines = [
        "=" * 60,
        "SCUBASTEVE V4 - PREDATOR UPGRADE",
        "RISK ANALYSIS REPORT",
        "=" * 60,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        f"Total Players Analyzed: {analysis.get('total_players', 'N/A')}",
        f"Top Players Evaluated: {analysis.get('analyzed_players', 'N/A')}",
        "",
        "KEY FINDINGS",
        "-" * 40,
        f"Optimal Risk Threshold (Death Line): {death_line:.2%}",
        f"Loss Rate Above Threshold: {loss_rate:.2%}",
        "",
        "RECOMMENDATION",
        "-" * 40,
        f"MAX_RISK_TOLERANCE = {death_line:.4f}",
        "",
        f"Players who step on >{death_line:.0%} risk squares",
        f"lose {loss_rate:.0%} of the time.",
        "",
        "IMPLEMENTATION",
        "-" * 40,
        "1. Hard-code MAX_RISK_TOLERANCE in agent.py",
        "2. Apply 2x penalty for moves exceeding threshold",
        "3. Use absolute avoidance for confirmed trapdoors",
        "",
        "DATA-DRIVEN INSIGHTS",
        "-" * 40,
    ]
    
    # Add top player stats
    top_players = analysis.get("top_players", [])[:10]
    if top_players:
        report_lines.append("Top 10 Players by Win Rate:")
        for i, player in enumerate(top_players, 1):
            report_lines.append(
                f"  {i}. Win Rate: {player['win_rate']:.2%}, "
                f"Max Risk: {player['max_risk']:.2%}, "
                f"Games: {player['games']}"
            )
    else:
        report_lines.append("No player data available - using default threshold.")
    
    report_lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    report_content = "\n".join(report_lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return report_content


def generate_default_report(output_path: str) -> str:
    """
    Generate a default report when no CSV data is available.
    
    Args:
        output_path: Path to output file
        
    Returns:
        Report content string
    """
    # Default values based on theoretical analysis
    death_line = 0.18
    loss_rate = 0.85
    
    report_lines = [
        "=" * 60,
        "SCUBASTEVE V4 - PREDATOR UPGRADE",
        "RISK ANALYSIS REPORT",
        "=" * 60,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        "Data Source: Theoretical Analysis + Game Engine Constants",
        "Status: Default values (awaiting CSV data)",
        "",
        "KEY FINDINGS",
        "-" * 40,
        f"Optimal Risk Threshold (Death Line): {death_line:.2%}",
        f"Estimated Loss Rate Above Threshold: {loss_rate:.2%}",
        "",
        "THEORETICAL BASIS",
        "-" * 40,
        "Trapdoor spawn region: center 4x4 (16 cells)",
        "Trapdoor parity: 2 trapdoors (even + odd cells)",
        "Effective trapdoor cells per parity: 8 cells",
        "Prior probability per cell: ~12.5%",
        "",
        "BAYESIAN ANALYSIS",
        "-" * 40,
        "Without sensor data: uniform ~12.5% per valid cell",
        "With negative sensors: probability decreases rapidly",
        "With positive sensors: probability concentrates",
        "",
        f"Threshold {death_line:.0%} corresponds to ~1.5x prior",
        "Crossing this threshold indicates elevated danger.",
        "",
        "RECOMMENDATION",
        "-" * 40,
        f"MAX_RISK_TOLERANCE = {death_line:.4f}",
        "",
        f"Estimated: Players who step on >{death_line:.0%} risk squares",
        f"lose approximately {loss_rate:.0%} of the time.",
        "",
        "IMPLEMENTATION",
        "-" * 40,
        "1. Hard-code MAX_RISK_TOLERANCE in agent.py",
        "2. Apply 2x penalty for moves exceeding threshold",
        "3. Use absolute avoidance for confirmed trapdoors",
        "4. Update threshold when CSV data becomes available",
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ]
    
    report_content = "\n".join(report_lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return report_content


def main():
    """Main entry point for risk analysis."""
    args = parse_args()
    
    output_path = os.path.join(
        os.path.dirname(__file__),
        args.output
    )
    
    if args.csv_file and os.path.exists(args.csv_file):
        print(f"Loading CSV data from: {args.csv_file}")
        records = load_csv_data(args.csv_file)
        print(f"Loaded {len(records)} game records")
        
        print(f"Analyzing top {args.top_n} players...")
        analysis = analyze_player_risk(records, args.top_n)
        
        print("Calculating optimal risk threshold (Death Line)...")
        death_line, loss_rate = find_death_line(analysis)
        
        print("Generating report...")
        report = generate_report(analysis, death_line, loss_rate, output_path)
    else:
        print("No CSV file provided or file not found.")
        print("Generating default report based on theoretical analysis...")
        report = generate_default_report(output_path)
    
    print(f"\nReport saved to: {output_path}")
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    # Extract key values from report
    for line in report.split("\n"):
        if "Death Line" in line or "MAX_RISK_TOLERANCE" in line:
            print(line)


if __name__ == "__main__":
    main()
