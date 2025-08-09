#!/usr/bin/env python3
"""
Merge connectivity and feature data for unified analysis
Shows how to correlate semantic connectivity with circuit complexity
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_data(csv_dir: Path):
    """Load all CSV files into a dictionary of DataFrames."""
    data = {}
    
    # Load each CSV
    csv_files = {
        'word_summary': 'word_summary.csv',
        'layer_connectivity': 'layer_connectivity.csv',
        'trajectories': 'connectivity_trajectories.csv',
        'feature_summary': 'feature_summary.csv',
        'feature_activations': 'feature_activations.csv',
        'neighbors': 'semantic_neighbors.csv'
    }
    
    for name, filename in csv_files.items():
        filepath = csv_dir / filename
        if filepath.exists():
            data[name] = pd.read_csv(filepath)
            print(f"✅ Loaded {name}: {len(data[name])} rows")
        else:
            print(f"⚠️  Missing {filename}")
    
    return data


def compute_circuit_complexity_score(data: dict) -> pd.DataFrame:
    """
    Compute a unified circuit complexity score combining:
    - Total unique features across layers
    - Cross-layer feature diversity
    - Connectivity evolution patterns
    """
    
    # Aggregate feature counts by word
    feature_by_word = data['feature_summary'].groupby('word').agg({
        'total_features': 'sum',
        'mean_activation': 'mean',
        'max_activation': 'max',
        'layer': 'count'  # Number of layers with features
    }).rename(columns={'layer': 'active_layers'})
    
    # Merge with connectivity summary
    complexity = data['word_summary'].merge(feature_by_word, on='word', how='left')
    
    # Compute complexity score
    # Normalize each component to 0-1 range
    complexity['norm_features'] = complexity['total_features'] / complexity['total_features'].max()
    complexity['norm_variance'] = complexity['connectivity_variance'] / complexity['connectivity_variance'].max()
    complexity['norm_layers'] = complexity['active_layers'] / complexity['active_layers'].max()
    
    # Combined complexity score (weights can be adjusted)
    complexity['complexity_score'] = (
        0.4 * complexity['norm_features'] +
        0.3 * complexity['norm_variance'] + 
        0.3 * complexity['norm_layers']
    )
    
    return complexity


def analyze_connectivity_feature_correlation(data: dict):
    """Analyze correlations between connectivity patterns and feature activation."""
    
    # Merge connectivity and feature summaries at layer level
    conn_layer = data['layer_connectivity'][['word', 'layer', 'connectivity_count', 'mean_similarity']]
    feat_layer = data['feature_summary'][['word', 'layer', 'total_features', 'mean_activation']]
    
    layer_merged = conn_layer.merge(feat_layer, on=['word', 'layer'], how='inner')
    
    # Compute correlations by layer
    correlations = []
    for layer in sorted(layer_merged['layer'].unique()):
        layer_data = layer_merged[layer_merged['layer'] == layer]
        if len(layer_data) > 10:  # Need enough data points
            corr_conn_feat = layer_data['connectivity_count'].corr(layer_data['total_features'])
            corr_sim_act = layer_data['mean_similarity'].corr(layer_data['mean_activation'])
            
            correlations.append({
                'layer': layer,
                'connectivity_vs_features': corr_conn_feat,
                'similarity_vs_activation': corr_sim_act,
                'n_words': len(layer_data)
            })
    
    return pd.DataFrame(correlations)


def identify_word_processing_types(complexity_df: pd.DataFrame) -> pd.DataFrame:
    """Classify words into processing types based on their patterns."""
    
    # Define processing types based on patterns
    complexity_df['processing_type'] = 'standard'
    
    # High variance + high features = "complex processing"
    complex_mask = (
        (complexity_df['connectivity_variance'] > complexity_df['connectivity_variance'].quantile(0.75)) &
        (complexity_df['total_features'] > complexity_df['total_features'].quantile(0.75))
    )
    complexity_df.loc[complex_mask, 'processing_type'] = 'complex'
    
    # Low variance + low features = "simple processing"
    simple_mask = (
        (complexity_df['connectivity_variance'] < complexity_df['connectivity_variance'].quantile(0.25)) &
        (complexity_df['total_features'] < complexity_df['total_features'].quantile(0.25))
    )
    complexity_df.loc[simple_mask, 'processing_type'] = 'simple'
    
    # Early peak + high stability = "lexical processing"
    lexical_mask = (
        (complexity_df['peak_layer'] <= 4) &
        (complexity_df['connectivity_stability'] > complexity_df['connectivity_stability'].quantile(0.75))
    )
    complexity_df.loc[lexical_mask, 'processing_type'] = 'lexical'
    
    # Late peak + high features = "semantic processing"
    semantic_mask = (
        (complexity_df['peak_layer'] >= 14) &
        (complexity_df['total_features'] > complexity_df['total_features'].median())
    )
    complexity_df.loc[semantic_mask, 'processing_type'] = 'semantic'
    
    return complexity_df


def create_analysis_report(data: dict, output_dir: Path):
    """Create comprehensive analysis report."""
    
    # Compute complexity scores
    complexity_df = compute_circuit_complexity_score(data)
    complexity_df = identify_word_processing_types(complexity_df)
    
    # Save enhanced dataset
    complexity_df.to_csv(output_dir / 'word_complexity_analysis.csv', index=False)
    print(f"\n✅ Saved word complexity analysis")
    
    # Analyze layer correlations
    layer_corr = analyze_connectivity_feature_correlation(data)
    layer_corr.to_csv(output_dir / 'layer_correlations.csv', index=False)
    print(f"✅ Saved layer correlation analysis")
    
    # Generate summary statistics
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Overall correlations
    print("\n🔗 Key Correlations:")
    print(f"Connectivity variance vs Total features: {complexity_df['connectivity_variance'].corr(complexity_df['total_features']):.3f}")
    print(f"Polysemy vs Complexity score: {complexity_df['polysemy_score'].corr(complexity_df['complexity_score']):.3f}")
    print(f"Peak layer vs Total features: {complexity_df['peak_layer'].corr(complexity_df['total_features']):.3f}")
    
    # Processing type distribution
    print("\n🧠 Processing Type Distribution:")
    for ptype, count in complexity_df['processing_type'].value_counts().items():
        print(f"  {ptype}: {count} words ({count/len(complexity_df)*100:.1f}%)")
    
    # Top complex words
    print("\n🏆 Top 10 Most Complex Words:")
    top_complex = complexity_df.nlargest(10, 'complexity_score')[['word', 'complexity_score', 'total_features', 'connectivity_variance']]
    print(top_complex.to_string(index=False))
    
    # Polysemy analysis
    if 'polysemy_score' in complexity_df.columns:
        print("\n📚 Polysemy Analysis:")
        poly_groups = complexity_df.groupby(pd.cut(complexity_df['polysemy_score'], bins=[0, 1, 3, 10, 100])).agg({
            'complexity_score': 'mean',
            'total_features': 'mean',
            'connectivity_variance': 'mean'
        })
        print(poly_groups)
    
    return complexity_df, layer_corr


def create_visualizations(complexity_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    
    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Connectivity variance vs Total features
    ax = axes[0, 0]
    scatter = ax.scatter(
        complexity_df['connectivity_variance'], 
        complexity_df['total_features'],
        c=complexity_df['polysemy_score'],
        cmap='viridis',
        alpha=0.6
    )
    ax.set_xlabel('Connectivity Variance')
    ax.set_ylabel('Total Features')
    ax.set_title('Connectivity Dynamics vs Circuit Complexity')
    plt.colorbar(scatter, ax=ax, label='Polysemy')
    
    # 2. Complexity score distribution by processing type
    ax = axes[0, 1]
    complexity_df.boxplot(column='complexity_score', by='processing_type', ax=ax)
    ax.set_xlabel('Processing Type')
    ax.set_ylabel('Complexity Score')
    ax.set_title('Complexity by Processing Type')
    
    # 3. Peak layer distribution
    ax = axes[1, 0]
    complexity_df['peak_layer'].hist(bins=20, ax=ax, alpha=0.7)
    ax.set_xlabel('Peak Connectivity Layer')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Peak Connectivity Layers')
    
    # 4. Polysemy vs Complexity
    ax = axes[1, 1]
    if 'polysemy_score' in complexity_df.columns:
        ax.scatter(
            complexity_df['polysemy_score'],
            complexity_df['complexity_score'],
            alpha=0.6
        )
        ax.set_xlabel('Polysemy Score')
        ax.set_ylabel('Complexity Score')
        ax.set_title('Polysemy vs Processing Complexity')
        
        # Add trend line
        z = np.polyfit(complexity_df['polysemy_score'], complexity_df['complexity_score'], 1)
        p = np.poly1d(z)
        ax.plot(complexity_df['polysemy_score'], p(complexity_df['polysemy_score']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_analysis_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization plots")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge and analyze connectivity + feature data"
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        required=True,
        help='Directory containing CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='merged_analysis',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔄 Loading data from CSVs...")
    data = load_all_data(csv_dir)
    
    if not all(key in data for key in ['word_summary', 'feature_summary']):
        print("❌ Missing required CSV files!")
        return
    
    print("\n🧮 Computing merged analysis...")
    complexity_df, layer_corr = create_analysis_report(data, output_dir)
    
    if args.visualize:
        print("\n📊 Creating visualizations...")
        create_visualizations(complexity_df, output_dir)
    
    print(f"\n✨ Analysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()