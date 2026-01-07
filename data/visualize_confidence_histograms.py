#!/usr/bin/env python3
"""
Visualize confidence score histograms for known vs unknown queries
Creates side-by-side histograms with blue for known scores and red for unknown scores
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_confidence_data(data_path):
    """Load confidence scores data from pickle file"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_histogram_comparison(known_scores, unknown_scores, save_path, title="Confidence Scores Comparison", bins=50):
    """
    Create histogram comparison between known and unknown confidence scores
    
    Args:
        known_scores: List of confidence scores for known queries
        unknown_scores: List of confidence scores for unknown queries  
        save_path: Path to save the histogram image
        title: Title for the plot
        bins: Number of bins for the histogram
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate overall range for consistent binning
    all_scores = known_scores + unknown_scores
    if len(all_scores) == 0:
        print("Warning: No confidence scores to plot")
        return
        
    score_min = min(all_scores)
    score_max = max(all_scores)
    bin_edges = np.linspace(score_min, score_max, bins + 1)
    
    # Plot histograms with transparency for overlap visibility
    alpha = 0.7
    
    if len(known_scores) > 0:
        ax.hist(known_scores, bins=bin_edges, alpha=alpha, color='blue', 
                label=f'Known Queries (n={len(known_scores)})', density=True, edgecolor='darkblue')
    
    if len(unknown_scores) > 0:
        ax.hist(unknown_scores, bins=bin_edges, alpha=alpha, color='red',
                label=f'Unknown Queries (n={len(unknown_scores)})', density=True, edgecolor='darkred')
    
    # Customize plot
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = ""
    if len(known_scores) > 0:
        known_mean = np.mean(known_scores)
        known_std = np.std(known_scores)
        stats_text += f"Known: μ={known_mean:.3f}, σ={known_std:.3f}\n"
    
    if len(unknown_scores) > 0:
        unknown_mean = np.mean(unknown_scores)
        unknown_std = np.std(unknown_scores)
        stats_text += f"Unknown: μ={unknown_mean:.3f}, σ={unknown_std:.3f}"
    
    if stats_text:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {save_path}")


def create_side_by_side_histograms(known_scores, unknown_scores, save_path, title="Confidence Scores Comparison", bins=50):
    """
    Create side-by-side histograms for known and unknown confidence scores
    
    Args:
        known_scores: List of confidence scores for known queries
        unknown_scores: List of confidence scores for unknown queries  
        save_path: Path to save the histogram image
        title: Title for the plot
        bins: Number of bins for the histogram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate overall range for consistent binning
    all_scores = known_scores + unknown_scores
    if len(all_scores) == 0:
        print("Warning: No confidence scores to plot")
        return
        
    score_min = min(all_scores)
    score_max = max(all_scores)
    bin_edges = np.linspace(score_min, score_max, bins + 1)
    
    # Plot known scores
    if len(known_scores) > 0:
        ax1.hist(known_scores, bins=bin_edges, color='blue', alpha=0.7, edgecolor='darkblue')
        ax1.set_title(f'Known Queries (n={len(known_scores)})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Confidence Score', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        known_mean = np.mean(known_scores)
        known_std = np.std(known_scores)
        ax1.text(0.02, 0.98, f"μ={known_mean:.3f}\nσ={known_std:.3f}", 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No Known Scores', transform=ax1.transAxes, 
                fontsize=12, ha='center', va='center')
        ax1.set_title('Known Queries (n=0)', fontsize=12, fontweight='bold')
    
    # Plot unknown scores  
    if len(unknown_scores) > 0:
        ax2.hist(unknown_scores, bins=bin_edges, color='red', alpha=0.7, edgecolor='darkred')
        ax2.set_title(f'Unknown Queries (n={len(unknown_scores)})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Confidence Score', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        unknown_mean = np.mean(unknown_scores)
        unknown_std = np.std(unknown_scores)
        ax2.text(0.02, 0.98, f"μ={unknown_mean:.3f}\nσ={unknown_std:.3f}", 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No Unknown Scores', transform=ax2.transAxes, 
                fontsize=12, ha='center', va='center')
        ax2.set_title('Unknown Queries (n=0)', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Side-by-side histogram saved to: {save_path}")


def visualize_confidence_scores(base_dir="data_analysis/confidence_scores", output_dir="data_analysis/histograms", 
                               plot_type="both", bins=50):
    """
    Visualize confidence scores for all available model/dataset/loss combinations
    
    Args:
        base_dir: Base directory containing confidence score data
        output_dir: Directory to save histogram plots
        plot_type: Type of plot ("overlay", "sidebyside", or "both")
        bins: Number of bins for histograms
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all confidence score files
    data_files = list(base_path.glob("*/*/*/confidence_scores.pkl"))
    
    if not data_files:
        print(f"No confidence score files found in {base_dir}")
        return
    
    print(f"Found {len(data_files)} confidence score files")
    
    for data_file in data_files:
        # Extract model, dataset, and loss from path
        parts = data_file.relative_to(base_path).parts
        if len(parts) >= 4:
            model_name, dataset_name, os_loss_name = parts[0], parts[1], parts[2]
        else:
            print(f"Warning: Unexpected path structure for {data_file}")
            continue
        
        # Load confidence data
        try:
            data = load_confidence_data(data_file)
            known_scores = data['known_scores']
            unknown_scores = data['unknown_scores']
            episode_count = data['episode_count']
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            continue
        
        # Create output directory for this combination
        combo_output_dir = output_path / model_name / dataset_name / os_loss_name
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create title
        title = f"{model_name} on {dataset_name} ({os_loss_name}) - {episode_count} episodes"
        
        # Create overlay histogram
        if plot_type in ["overlay", "both"]:
            overlay_path = combo_output_dir / "confidence_histogram_overlay.png"
            create_histogram_comparison(known_scores, unknown_scores, overlay_path, title, bins)
        
        # Create side-by-side histogram
        if plot_type in ["sidebyside", "both"]:
            sidebyside_path = combo_output_dir / "confidence_histogram_sidebyside.png"
            create_side_by_side_histograms(known_scores, unknown_scores, sidebyside_path, title, bins)
        
        print(f"Processed: {model_name}/{dataset_name}/{os_loss_name} - Known: {len(known_scores)}, Unknown: {len(unknown_scores)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize confidence score histograms")
    parser.add_argument("--data_dir", type=str, default="data_analysis/confidence_scores",
                        help="Directory containing confidence score data")
    parser.add_argument("--output_dir", type=str, default="data_analysis/histograms",
                        help="Directory to save histogram plots")
    parser.add_argument("--plot_type", type=str, choices=["overlay", "sidebyside", "both"], 
                        default="both", help="Type of histogram plot")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of bins for histograms")
    parser.add_argument("--model", type=str, help="Specific model to visualize")
    parser.add_argument("--dataset", type=str, help="Specific dataset to visualize")
    parser.add_argument("--os_loss", type=str, help="Specific open set loss to visualize")
    
    args = parser.parse_args()
    
    # If specific model/dataset/loss specified, process only that combination
    if args.model and args.dataset and args.os_loss:
        data_file = Path(args.data_dir) / args.model / args.dataset / args.os_loss / "confidence_scores.pkl"
        if not data_file.exists():
            print(f"Error: File {data_file} does not exist")
            return
        
        try:
            data = load_confidence_data(data_file)
            known_scores = data['known_scores']
            unknown_scores = data['unknown_scores']
            episode_count = data['episode_count']
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            return
        
        # Create output directory
        output_path = Path(args.output_dir) / args.model / args.dataset / args.os_loss
        output_path.mkdir(parents=True, exist_ok=True)
        
        title = f"{args.model} on {args.dataset} ({args.os_loss}) - {episode_count} episodes"
        
        if args.plot_type in ["overlay", "both"]:
            overlay_path = output_path / "confidence_histogram_overlay.png"
            create_histogram_comparison(known_scores, unknown_scores, overlay_path, title, args.bins)
        
        if args.plot_type in ["sidebyside", "both"]:
            sidebyside_path = output_path / "confidence_histogram_sidebyside.png"
            create_side_by_side_histograms(known_scores, unknown_scores, sidebyside_path, title, args.bins)
        
        print(f"Processed: {args.model}/{args.dataset}/{args.os_loss} - Known: {len(known_scores)}, Unknown: {len(unknown_scores)}")
    
    else:
        # Process all available combinations
        visualize_confidence_scores(args.data_dir, args.output_dir, args.plot_type, args.bins)


if __name__ == "__main__":
    main()