#!/usr/bin/env python3
"""
Analyze the relationship between semantic connectivity and feature activation.
This script works on any machine (no GPU required) using the pre-computed CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

def load_data(results_dir):
    """Load all the pre-computed data files."""
    results_dir = Path(results_dir)
    
    # Load connectivity + features merged data
    merged_file = results_dir / "connectivity_features_merged.csv"
    if merged_file.exists():
        df_merged = pd.read_csv(merged_file)
        print(f"Loaded {len(df_merged)} word-layer pairs from merged data")
    else:
        print("Warning: No merged data found. Run feature extraction first.")
        df_merged = None
    
    # Load word summaries
    word_summary_file = results_dir / "word_feature_summary.csv"
    if word_summary_file.exists():
        df_word_features = pd.read_csv(word_summary_file)
    else:
        df_word_features = None
    
    word_connectivity_file = results_dir / "word_summary.csv"
    if word_connectivity_file.exists():
        df_word_conn = pd.read_csv(word_connectivity_file)
    else:
        df_word_conn = None
    
    # Load trajectories
    trajectory_file = results_dir / "connectivity_trajectories.csv"
    if trajectory_file.exists():
        df_trajectories = pd.read_csv(trajectory_file)
    else:
        df_trajectories = None
    
    return df_merged, df_word_features, df_word_conn, df_trajectories

def analyze_layer_correlations(df_merged):
    """Analyze correlations between connectivity and features at each layer."""
    print("\n=== Layer-wise Correlation Analysis ===")
    
    layer_correlations = []
    
    for layer in sorted(df_merged['layer'].unique()):
        layer_data = df_merged[df_merged['layer'] == layer]
        
        if len(layer_data) > 10:  # Need enough data points
            # Pearson correlation
            corr, p_value = stats.pearsonr(
                layer_data['connectivity_count'],
                layer_data['active_features_count']
            )
            
            # Spearman correlation (for non-linear relationships)
            spearman_corr, spearman_p = stats.spearmanr(
                layer_data['connectivity_count'],
                layer_data['active_features_count']
            )
            
            layer_correlations.append({
                'layer': layer,
                'pearson_r': corr,
                'pearson_p': p_value,
                'spearman_r': spearman_corr,
                'spearman_p': spearman_p,
                'n_samples': len(layer_data)
            })
            
            print(f"Layer {layer:2d}: Pearson r={corr:.3f} (p={p_value:.3f}), "
                  f"Spearman r={spearman_corr:.3f}, n={len(layer_data)}")
    
    return pd.DataFrame(layer_correlations)

def analyze_word_level_patterns(df_word_features, df_word_conn):
    """Analyze patterns at the word level."""
    print("\n=== Word-level Analysis ===")
    
    if df_word_features is None or df_word_conn is None:
        print("Missing word-level data")
        return None
    
    # Merge word-level data
    df_words = pd.merge(
        df_word_features,
        df_word_conn[['word', 'polysemy_score', 'total_connectivity']],
        on='word',
        how='inner'
    )
    
    # Analyze correlations
    print("\nCorrelations with total active features:")
    correlations = {
        'polysemy_score': df_words['polysemy_score'].corr(df_words['total_active_features']),
        'total_connectivity': df_words['total_connectivity'].corr(df_words['total_active_features']),
        'active_layers': df_words['active_layers'].corr(df_words['total_active_features'])
    }
    
    for metric, corr in correlations.items():
        print(f"  {metric}: {corr:.3f}")
    
    return df_words

def plot_layer_correlations(df_correlations, output_dir):
    """Plot correlation strength across layers."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_correlations['layer'], df_correlations['pearson_r'], 'o-', label='Pearson')
    plt.plot(df_correlations['layer'], df_correlations['spearman_r'], 's-', label='Spearman')
    plt.xlabel('Layer')
    plt.ylabel('Correlation Coefficient')
    plt.title('Connectivity-Features Correlation by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    significant = df_correlations[df_correlations['pearson_p'] < 0.05]
    plt.bar(significant['layer'], significant['pearson_r'])
    plt.xlabel('Layer')
    plt.ylabel('Pearson r (p < 0.05)')
    plt.title('Significant Correlations Only')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_correlations.png', dpi=150)
    print(f"Saved correlation plot to {output_dir / 'layer_correlations.png'}")

def plot_connectivity_vs_features(df_merged, output_dir, sample_layers=None):
    """Scatter plots of connectivity vs features for selected layers."""
    if sample_layers is None:
        # Sample early, middle, and late layers
        all_layers = sorted(df_merged['layer'].unique())
        sample_layers = [
            all_layers[0],  # First
            all_layers[len(all_layers)//4],  # Early
            all_layers[len(all_layers)//2],  # Middle
            all_layers[-1]  # Last
        ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, layer in enumerate(sample_layers[:4]):
        ax = axes[idx]
        layer_data = df_merged[df_merged['layer'] == layer]
        
        # Add some jitter to see overlapping points
        x = layer_data['connectivity_count'] + np.random.normal(0, 0.5, len(layer_data))
        y = layer_data['active_features_count'] + np.random.normal(0, 1, len(layer_data))
        
        ax.scatter(x, y, alpha=0.5, s=20)
        
        # Add regression line
        if len(layer_data) > 10:
            z = np.polyfit(layer_data['connectivity_count'], 
                          layer_data['active_features_count'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(layer_data['connectivity_count'].min(), 
                               layer_data['connectivity_count'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        
        corr = layer_data['connectivity_count'].corr(layer_data['active_features_count'])
        ax.set_title(f'Layer {layer} (r={corr:.3f})')
        ax.set_xlabel('Semantic Connectivity')
        ax.set_ylabel('Active Features')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'connectivity_vs_features_scatter.png', dpi=150)
    print(f"Saved scatter plot to {output_dir / 'connectivity_vs_features_scatter.png'}")

def analyze_evolution_patterns(df_trajectories, df_word_features):
    """Analyze how connectivity evolution relates to feature activation patterns."""
    print("\n=== Evolution Pattern Analysis ===")
    
    if df_trajectories is None or df_word_features is None:
        print("Missing trajectory data")
        return
    
    # Merge trajectory stats with feature stats
    df_evolution = pd.merge(
        df_trajectories[['word', 'connectivity_variance', 'peak_layer', 'trajectory_type']],
        df_word_features[['word', 'total_active_features', 'std_features_per_layer']],
        on='word',
        how='inner'
    )
    
    print("\nCorrelations with connectivity evolution:")
    print(f"  Connectivity variance vs Total features: "
          f"{df_evolution['connectivity_variance'].corr(df_evolution['total_active_features']):.3f}")
    print(f"  Connectivity variance vs Feature spread: "
          f"{df_evolution['connectivity_variance'].corr(df_evolution['std_features_per_layer']):.3f}")
    
    # Group by trajectory type
    print("\nAverage features by trajectory type:")
    grouped = df_evolution.groupby('trajectory_type').agg({
        'total_active_features': 'mean',
        'std_features_per_layer': 'mean',
        'word': 'count'
    }).round(1)
    grouped.columns = ['avg_features', 'avg_feature_spread', 'n_words']
    print(grouped)
    
    return df_evolution

def main():
    parser = argparse.ArgumentParser(description='Analyze connectivity-features relationship')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory with analysis results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output plots (default: results-dir/analysis)')
    
    args = parser.parse_args()
    
    # Setup output directory
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df_merged, df_word_features, df_word_conn, df_trajectories = load_data(results_dir)
    
    if df_merged is not None:
        # Layer-wise correlation analysis
        df_correlations = analyze_layer_correlations(df_merged)
        
        # Save correlation results
        corr_file = output_dir / 'layer_correlations.csv'
        df_correlations.to_csv(corr_file, index=False)
        print(f"\nSaved correlation analysis to {corr_file}")
        
        # Plot correlations
        if len(df_correlations) > 0:
            plot_layer_correlations(df_correlations, output_dir)
            plot_connectivity_vs_features(df_merged, output_dir)
    
    # Word-level analysis
    df_words = analyze_word_level_patterns(df_word_features, df_word_conn)
    
    # Evolution pattern analysis
    if df_trajectories is not None and df_word_features is not None:
        df_evolution = analyze_evolution_patterns(df_trajectories, df_word_features)
        
        # Save evolution analysis
        if df_evolution is not None:
            evolution_file = output_dir / 'evolution_features_analysis.csv'
            df_evolution.to_csv(evolution_file, index=False)
            print(f"\nSaved evolution analysis to {evolution_file}")
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    
    # Print summary of findings
    if df_merged is not None:
        overall_corr = df_merged['connectivity_count'].corr(df_merged['active_features_count'])
        print(f"\n🔍 Key Finding: Overall correlation between semantic connectivity "
              f"and active features: {overall_corr:.3f}")

if __name__ == "__main__":
    main()