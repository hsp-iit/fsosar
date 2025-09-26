#!/usr/bin/env python3
"""
Confusion Matrix Visualization Script

This script loads and visualizes confusion matrices saved by the save_confusion_matrix function.
It creates publication-ready heatmaps with proper class names, accuracy metrics, and statistics.

Usage:
    python visualize_confusion_matrix.py --model SAFSAR --dataset HMDB51 --os_loss softmax
    python visualize_confusion_matrix.py --all  # Visualize all available confusion matrices
    python visualize_confusion_matrix.py --list  # List all available confusion matrix files
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


def load_confusion_matrix(model, dataset, os_loss):
    """Load confusion matrix data from pickle file"""
    cm_path = f'data_analysis/confusion_matrices/{model}/{dataset}/{os_loss}/confusion_data.pkl'
    
    if not os.path.exists(cm_path):
        raise FileNotFoundError(f"Confusion matrix not found at: {cm_path}")
    
    with open(cm_path, 'rb') as f:
        cm_data = pickle.load(f)
    
    return cm_data, cm_path


def plot_confusion_matrix(cm_data, save_path=None, show=True):
    """
    Plot confusion matrix with enhanced visualization
    
    Args:
        cm_data: Dictionary containing confusion matrix and metadata
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    cm = cm_data['confusion_matrix']
    class_names = cm_data['class_names']
    model_name = cm_data['model_type']
    dataset = cm_data['dataset']
    os_loss = cm_data['os_loss']
    episode_count = cm_data['episode_count']
    
    # Filter out classes with zero entries (not used during testing)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    used_classes = (row_sums > 0) | (col_sums > 0)
    
    # Create filtered confusion matrix and class names
    if used_classes.sum() < len(class_names):
        cm_filtered = cm[np.ix_(used_classes, used_classes)]
        class_names_filtered = [class_names[i] for i in range(len(class_names)) if used_classes[i]]
        print(f"Filtered confusion matrix: {len(class_names)} -> {len(class_names_filtered)} classes (removed {len(class_names) - len(class_names_filtered)} unused classes)")
    else:
        cm_filtered = cm
        class_names_filtered = class_names
    
    # Calculate metrics on filtered matrix
    total_samples = cm_filtered.sum()
    
    # Overall accuracy (diagonal sum / total)
    overall_accuracy = np.diag(cm_filtered).sum() / total_samples if total_samples > 0 else 0
    
    # Per-class accuracy (diagonal / row sum)
    row_sums_filtered = cm_filtered.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm_filtered), row_sums_filtered, out=np.zeros_like(np.diag(cm_filtered), dtype=float), where=row_sums_filtered!=0)
    
    # Open set analysis using separate tracking
    unknown_correctly_rejected = cm_data.get('unknown_correctly_rejected', 0)
    unknown_misclassified_count = cm_data.get('unknown_misclassified_count', 0)
    unknown_misclassified_as = cm_data.get('unknown_misclassified_as', np.zeros(len(class_names)))
    
    if unknown_correctly_rejected > 0 or unknown_misclassified_count > 0:
        # True positives for known classes (diagonal)
        known_tp = np.diag(cm).sum()
        # False positives from unknown samples
        unknown_fp = unknown_misclassified_count
        # True negatives (unknown correctly rejected)
        unknown_tn = unknown_correctly_rejected
        
        # Open set metrics
        total_unknown = unknown_tn + unknown_fp
        unknown_precision = unknown_tn / total_unknown if total_unknown > 0 else 0
        known_contamination = unknown_fp / (known_tp + unknown_fp) if (known_tp + unknown_fp) > 0 else 0
        
    # Calculate figure size based on number of filtered classes
    n_classes = len(class_names_filtered)
    fig_size = max(8, min(20, n_classes * 0.6))  # Scale figure size with number of classes
    
    # Create the plot
    plt.figure(figsize=(fig_size, fig_size))
    
    # Normalize confusion matrix for better visualization (using filtered matrix)
    cm_normalized = cm_filtered.astype('float') / cm_filtered.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Calculate font sizes based on number of classes - increased tick sizes for better readability
    if n_classes <= 10:
        annot_fontsize = 10
        tick_fontsize = 12  # Increased from 9
        title_fontsize = 14
        label_fontsize = 12
    elif n_classes <= 20:
        annot_fontsize = 8
        tick_fontsize = 10  # Increased from 7
        title_fontsize = 12
        label_fontsize = 10
    else:
        annot_fontsize = 6
        tick_fontsize = 8   # Increased from 5
        title_fontsize = 10
        label_fontsize = 8
    
    # Create heatmap with normalized colors but raw count annotations (using filtered data)
    ax = sns.heatmap(cm_normalized, 
                     annot=cm_filtered,  # Show raw counts as numbers
                     fmt='d',   # Integer format
                     cmap='Blues',
                     xticklabels=class_names_filtered,
                     yticklabels=class_names_filtered,
                     cbar_kws={'label': 'Normalized Frequency'},
                     annot_kws={'size': annot_fontsize},  # Control annotation font size
                     square=True)  # Make cells square
    
    plt.title(f'Confusion Matrix: {model_name} on {dataset} ({os_loss})\n'
              f'Episodes: {episode_count}, Total Samples: {total_samples}, '
              f'Overall Accuracy: {overall_accuracy:.3f}', 
              fontsize=title_fontsize, pad=20)
    
    plt.xlabel('Predicted Class', fontsize=label_fontsize)
    plt.ylabel('True Class', fontsize=label_fontsize)
    
    # Rotate labels for better readability with smaller fonts
    plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    
    # Improve layout for many classes
    if n_classes > 15:
        # For many classes, use vertical labels and adjust spacing
        plt.xticks(rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.15, left=0.15)
    
    # Add statistics as text - make it more compact for many classes
    if n_classes <= 15:
        # For few classes, show detailed per-class accuracy (using filtered data)
        stats_text = f'Per-class Accuracy:\n'
        for i, (class_name, acc) in enumerate(zip(class_names_filtered, per_class_acc)):
            stats_text += f'{class_name}: {acc:.3f}\n'
        stats_fontsize = 9
    else:
        # For many classes, show summary statistics
        stats_text = f'Accuracy Summary:\n'
        stats_text += f'Mean Accuracy: {per_class_acc.mean():.3f}\n'
        stats_text += f'Std Accuracy: {per_class_acc.std():.3f}\n'
        stats_text += f'Min Accuracy: {per_class_acc.min():.3f}\n'
        stats_text += f'Max Accuracy: {per_class_acc.max():.3f}\n'
        
        # Show worst and best performing classes (using filtered data)
        worst_idx = np.argmin(per_class_acc)
        best_idx = np.argmax(per_class_acc)
        stats_text += f'\nWorst: {class_names_filtered[worst_idx]} ({per_class_acc[worst_idx]:.3f})\n'
        stats_text += f'Best: {class_names_filtered[best_idx]} ({per_class_acc[best_idx]:.3f})\n'
        stats_fontsize = 8
    
    if unknown_correctly_rejected > 0 or unknown_misclassified_count > 0:
        stats_text += f'\nOpen Set Metrics:\n'
        stats_text += f'Unknown Precision: {unknown_precision:.3f}\n'
        stats_text += f'Known Contamination: {known_contamination:.3f}\n'
        stats_text += f'Unknown Correct: {unknown_tn}, Unknown Wrong: {unknown_fp}'
    
    # Position stats box better for different figure sizes
    if n_classes > 15:
        # For large matrices, put stats at the top
        plt.figtext(0.02, 0.98, stats_text, fontsize=stats_fontsize, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    else:
        # For smaller matrices, put stats on the right
        plt.figtext(1.02, 0.5, stats_text, fontsize=stats_fontsize, 
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def list_available_matrices():
    """List all available confusion matrix files"""
    base_path = "data_analysis/confusion_matrices"
    
    if not os.path.exists(base_path):
        print("No confusion matrices found. Directory does not exist:", base_path)
        return []
    
    # Find all pickle files
    pattern = f"{base_path}/*/*/*/confusion_data.pkl"
    files = glob.glob(pattern)
    
    available_configs = []
    
    print("Available Confusion Matrices:")
    print("=" * 50)
    
    for file_path in files:
        # Extract model, dataset, os_loss from path
        parts = Path(file_path).parts
        if len(parts) >= 4:
            model = parts[-4]
            dataset = parts[-3]
            os_loss = parts[-2]
            
            # Load to get episode count
            try:
                with open(file_path, 'rb') as f:
                    cm_data = pickle.load(f)
                episode_count = cm_data.get('episode_count', 'Unknown')
                total_samples = cm_data['confusion_matrix'].sum()
                
                config = (model, dataset, os_loss)
                available_configs.append(config)
                
                print(f"Model: {model:10} | Dataset: {dataset:12} | OS Loss: {os_loss:12} | "
                      f"Episodes: {episode_count:6} | Samples: {total_samples:6}")
                      
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return available_configs


def main():
    parser = argparse.ArgumentParser(description="Visualize confusion matrices from saved data")
    parser.add_argument('--model', type=str, help='Model name (e.g., SAFSAR, STRM)')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., HMDB51, UCF101)')
    parser.add_argument('--os_loss', type=str, help='Open set loss (e.g., softmax, discriminator)')
    parser.add_argument('--all', action='store_true', help='Visualize all available confusion matrices')
    parser.add_argument('--list', action='store_true', help='List all available confusion matrices')
    parser.add_argument('--save_dir', type=str, default='data_analysis/confusion_matrices/plots', 
                       help='Directory to save plots (default: data_analysis/confusion_matrices/plots)')
    parser.add_argument('--no_show', action='store_true', help='Do not display plots, only save them')
    
    args = parser.parse_args()
    
    # Create save directory
    if not args.no_show or args.all:
        os.makedirs(args.save_dir, exist_ok=True)
    
    if args.list:
        list_available_matrices()
        return
    
    if args.all:
        # Visualize all available confusion matrices
        available_configs = list_available_matrices()
        
        if not available_configs:
            print("No confusion matrices found to visualize.")
            return
        
        print(f"\nGenerating plots for {len(available_configs)} configurations...")
        
        for model, dataset, os_loss in available_configs:
            try:
                cm_data, cm_path = load_confusion_matrix(model, dataset, os_loss)
                
                # Create save path
                plot_filename = f"{model}_{dataset}_{os_loss}_confusion_matrix.png"
                save_path = os.path.join(args.save_dir, plot_filename)
                
                # Plot confusion matrix
                plot_confusion_matrix(cm_data, save_path=save_path, show=not args.no_show)
                
                print(f"✓ Processed: {model}/{dataset}/{os_loss}")
                
            except Exception as e:
                print(f"✗ Error processing {model}/{dataset}/{os_loss}: {e}")
    
    elif args.model and args.dataset and args.os_loss:
        # Visualize specific confusion matrix
        try:
            cm_data, cm_path = load_confusion_matrix(args.model, args.dataset, args.os_loss)
            
            # Create save path
            plot_filename = f"{args.model}_{args.dataset}_{args.os_loss}_confusion_matrix.png"
            save_path = os.path.join(args.save_dir, plot_filename) if not args.no_show else None
            
            plot_confusion_matrix(cm_data, save_path=save_path, show=not args.no_show)
            
            print(f"Loaded confusion matrix from: {cm_path}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Use --list to see available confusion matrices")
        except Exception as e:
            print(f"Error loading confusion matrix: {e}")
    
    else:
        print("Please specify --model, --dataset, and --os_loss, or use --all to visualize all matrices, or --list to see available options")
        parser.print_help()


if __name__ == "__main__":
    main()