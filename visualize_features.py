#!/usr/bin/env python3
"""
Simple t-SNE Feature Visualizer
===============================

Usage: python visualize_features.py [pkl_file_path] [num_classes]
Example: python visualize_features.py saved_features/class_features.pkl 10

Creates a t-SNE plot with each class in a unique color.
If num_classes is specified, randomly selects that many classes to visualize.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys
import os
import random


def visualize_features(pkl_file, num_classes=None):
    """
    Load features from pkl file and create t-SNE visualization
    
    Args:
        pkl_file: Path to the pickle file containing class features
        num_classes: Number of random classes to visualize (if None, use all classes)
    """
    # Load the feature dictionary
    if not os.path.exists(pkl_file):
        print(f"Error: File {pkl_file} not found!")
        return
    
    try:
        with open(pkl_file, 'rb') as f:
            class_feature_dict = pickle.load(f)
        print(f"Loaded {len(class_feature_dict)} classes from {pkl_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    if not class_feature_dict:
        print("No features found in the file!")
        return
    
    # Select random subset of classes if specified
    all_class_names = list(class_feature_dict.keys())
    if num_classes is not None and num_classes < len(all_class_names):
        selected_classes = random.sample(all_class_names, num_classes)
        print(f"Randomly selected {num_classes} classes out of {len(all_class_names)} total classes")
    else:
        selected_classes = all_class_names
        if num_classes is not None:
            print(f"Requested {num_classes} classes but only {len(all_class_names)} available, using all")
    
    # Prepare data for t-SNE
    all_features = []
    all_labels = []
    
    print("Classes to visualize:")
    for class_name in selected_classes:
        data = class_feature_dict[class_name]
        features = data['features']
        print(f"  - {class_name}: {len(features)} features")
        
        all_features.extend(features)
        all_labels.extend([class_name] * len(features))
    
    # Convert to numpy array
    combined_features = np.vstack(all_features)
    total_features = len(combined_features)
    
    print(f"\nTotal features: {total_features}")
    print("Creating t-SNE visualization...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)
    
    # Apply t-SNE
    perplexity = min(30, total_features - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, verbose=1)
    features_2d = tsne.fit_transform(features_scaled)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Get unique classes and assign colors
    unique_classes = selected_classes
    
    # Use maximally distinguishable colors for better visibility
    if len(unique_classes) <= 10:
        # For small number of classes, use manually selected distinguishable colors
        base_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                      '#00FFFF', '#FF8000', '#8000FF', '#FF0080', '#80FF00']
        colors = [base_colors[i % len(base_colors)] for i in range(len(unique_classes))]
    elif len(unique_classes) <= 20:
        # For medium number of classes, use tab20 colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    else:
        # For large number of classes, use hsv colormap for maximum spread
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_classes)))
    
    # Plot each class with its unique color
    for class_idx, class_name in enumerate(unique_classes):
        class_mask = np.array(all_labels) == class_name
        num_features = np.sum(class_mask)
        
        plt.scatter(features_2d[class_mask, 0], features_2d[class_mask, 1], 
                   c=[colors[class_idx]], label=f'{class_name} ({num_features})', 
                   s=60, alpha=0.7)
    
    plt.title(f't-SNE Visualization of Class Features\n{total_features} features from {len(unique_classes)} classes')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_file = pkl_file.replace('.pkl', '_tsne.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nt-SNE plot saved as: {output_file}")
    
    # Also show the plot
    plt.show()


def main():
    # Get input file and num_classes from command line or use defaults
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    else:
        pkl_file = 'saved_features/class_features.pkl'
    
    num_classes = None
    if len(sys.argv) > 2:
        try:
            num_classes = int(sys.argv[2])
            print(f"Will visualize {num_classes} random classes")
        except ValueError:
            print(f"Warning: '{sys.argv[2]}' is not a valid number, using all classes")
    
    print(f"Visualizing features from: {pkl_file}")
    visualize_features(pkl_file, num_classes)


if __name__ == '__main__':
    main()