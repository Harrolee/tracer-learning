#!/usr/bin/env python3
"""
Export layer-wise connectivity results to CSV format for analysis
Creates multiple related CSV files for different aspects of the data
"""

import argparse
import json
import csv
import os
from collections import defaultdict
from typing import Dict, List, Any

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def get_wordnet_synset_count(word: str) -> int:
    """Get number of synsets (meanings) for a word from WordNet."""
    if not NLTK_AVAILABLE:
        return 0
    try:
        synsets = wordnet.synsets(word)
        return len(synsets)
    except:
        return 0


def load_polysemy_scores(filepath: str) -> Dict[str, int]:
    """Load pre-computed polysemy scores if available."""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def export_word_summary_csv(results: Dict, polysemy_scores: Dict[str, int], output_dir: str):
    """
    Export word-level summary statistics to CSV.
    
    Columns:
    - word
    - polysemy_score (from WordNet)
    - max_connectivity
    - min_connectivity 
    - mean_connectivity
    - connectivity_variance
    - connectivity_stability
    - peak_layer
    - trough_layer
    """
    csv_path = os.path.join(output_dir, 'word_summary.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'polysemy_score', 'max_connectivity', 'min_connectivity',
            'mean_connectivity', 'connectivity_variance', 'connectivity_stability',
            'peak_layer', 'trough_layer'
        ])
        writer.writeheader()
        
        for word_result in results['word_results']:
            word = word_result['word']
            evolution = word_result['evolution']
            
            # Get polysemy score
            polysemy = polysemy_scores.get(word, get_wordnet_synset_count(word))
            
            writer.writerow({
                'word': word,
                'polysemy_score': polysemy,
                'max_connectivity': evolution['max_connectivity'],
                'min_connectivity': evolution['min_connectivity'],
                'mean_connectivity': round(evolution['mean_connectivity'], 2),
                'connectivity_variance': round(evolution['variance'], 2),
                'connectivity_stability': round(evolution['stability'], 4),
                'peak_layer': evolution['peak_layer'],
                'trough_layer': evolution['trough_layer']
            })
    
    print(f"✅ Exported word summary to {csv_path}")


def export_layer_connectivity_csv(results: Dict, output_dir: str):
    """
    Export word-layer connectivity data to CSV.
    
    Columns:
    - word
    - layer
    - connectivity_count
    - mean_similarity
    - max_similarity
    """
    csv_path = os.path.join(output_dir, 'layer_connectivity.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'layer', 'connectivity_count', 'mean_similarity', 'max_similarity'
        ])
        writer.writeheader()
        
        for word_result in results['word_results']:
            word = word_result['word']
            
            for layer_key, layer_data in word_result['layer_results'].items():
                layer_num = int(layer_key.split('_')[1])
                
                writer.writerow({
                    'word': word,
                    'layer': layer_num,
                    'connectivity_count': layer_data['connectivity_count'],
                    'mean_similarity': round(layer_data['mean_similarity'], 4),
                    'max_similarity': round(layer_data['max_similarity'], 4)
                })
    
    print(f"✅ Exported layer connectivity to {csv_path}")


def export_top_neighbors_csv(results: Dict, output_dir: str, top_n: int = 5):
    """
    Export top semantic neighbors for each word-layer pair.
    
    Columns:
    - word
    - layer
    - neighbor_rank
    - neighbor_word
    - similarity
    """
    csv_path = os.path.join(output_dir, 'semantic_neighbors.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'layer', 'neighbor_rank', 'neighbor_word', 'similarity'
        ])
        writer.writeheader()
        
        for word_result in results['word_results']:
            word = word_result['word']
            
            for layer_key, layer_data in word_result['layer_results'].items():
                layer_num = int(layer_key.split('_')[1])
                
                # Get top N neighbors
                neighbors = layer_data.get('top_neighbors', [])[:top_n]
                
                for rank, (neighbor_word, similarity) in enumerate(neighbors, 1):
                    writer.writerow({
                        'word': word,
                        'layer': layer_num,
                        'neighbor_rank': rank,
                        'neighbor_word': neighbor_word,
                        'similarity': round(similarity, 4)
                    })
    
    print(f"✅ Exported semantic neighbors to {csv_path}")


def export_connectivity_trajectories_csv(results: Dict, output_dir: str):
    """
    Export connectivity trajectories in wide format for easy plotting.
    
    Columns:
    - word
    - layer_0_connectivity
    - layer_4_connectivity
    - ...
    """
    # Get all unique layers
    layers = results['word_results'][0]['layers_analyzed']
    
    csv_path = os.path.join(output_dir, 'connectivity_trajectories.csv')
    
    fieldnames = ['word'] + [f'layer_{l}_connectivity' for l in layers]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for word_result in results['word_results']:
            row = {'word': word_result['word']}
            
            for i, layer in enumerate(layers):
                connectivity = word_result['evolution']['trajectory'][i]
                row[f'layer_{layer}_connectivity'] = connectivity
            
            writer.writerow(row)
    
    print(f"✅ Exported connectivity trajectories to {csv_path}")


def create_feature_tracking_template(output_dir: str):
    """
    Create a template CSV for future feature tracking.
    
    Columns:
    - word
    - layer
    - feature_id
    - activation_strength
    - feature_type (if known)
    """
    csv_path = os.path.join(output_dir, 'feature_activations_template.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'layer', 'feature_id', 'activation_strength', 'feature_type'
        ])
        writer.writeheader()
        
        # Write a few example rows
        writer.writerow({
            'word': 'example',
            'layer': 0,
            'feature_id': 'f_0_123',
            'activation_strength': 0.85,
            'feature_type': 'syntactic'
        })
    
    print(f"✅ Created feature tracking template at {csv_path}")


def export_metadata_json(results: Dict, output_dir: str):
    """Export metadata about the analysis for reference."""
    metadata_path = os.path.join(output_dir, 'analysis_metadata.json')
    
    metadata = {
        'model': results['metadata']['model'],
        'total_words': results['metadata']['total_words'],
        'layers_analyzed': results['metadata']['layers_analyzed'],
        'threshold': results['metadata']['threshold'],
        'device': results['metadata']['device'],
        'csv_files': {
            'word_summary': 'word_summary.csv',
            'layer_connectivity': 'layer_connectivity.csv',
            'semantic_neighbors': 'semantic_neighbors.csv',
            'connectivity_trajectories': 'connectivity_trajectories.csv',
            'feature_template': 'feature_activations_template.csv'
        },
        'column_descriptions': {
            'polysemy_score': 'Number of WordNet synsets for the word',
            'connectivity_count': 'Number of words with similarity > threshold',
            'connectivity_variance': 'Variance of connectivity across layers',
            'connectivity_stability': '1 / (1 + variance), higher = more stable',
            'peak_layer': 'Layer with highest connectivity',
            'trough_layer': 'Layer with lowest connectivity'
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Exported metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export connectivity results to CSV format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='JSON results file from connectivity analysis'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='connectivity_csv_export',
        help='Directory for CSV outputs'
    )
    parser.add_argument(
        '--polysemy-file',
        type=str,
        help='Pre-computed polysemy scores JSON'
    )
    parser.add_argument(
        '--download-wordnet',
        action='store_true',
        help='Download WordNet data if not available'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download WordNet if requested
    if args.download_wordnet:
        if NLTK_AVAILABLE:
            print("📥 Downloading WordNet data...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        else:
            print("⚠️  NLTK not installed, skipping WordNet download")
    
    # Load results
    print(f"📂 Loading results from {args.input}")
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    # Load polysemy scores
    polysemy_scores = load_polysemy_scores(args.polysemy_file)
    if not polysemy_scores:
        print("⚠️  No polysemy file provided, will compute from WordNet")
    
    # Export CSVs
    print(f"\n📊 Exporting to CSVs in {args.output_dir}/")
    
    export_word_summary_csv(results, polysemy_scores, args.output_dir)
    export_layer_connectivity_csv(results, args.output_dir)
    export_top_neighbors_csv(results, args.output_dir)
    export_connectivity_trajectories_csv(results, args.output_dir)
    create_feature_tracking_template(args.output_dir)
    export_metadata_json(results, args.output_dir)
    
    print(f"\n✨ Export complete! Files created in {args.output_dir}/")
    print("\nYou can now analyze the data using:")
    print("  - pandas: pd.read_csv('word_summary.csv')")
    print("  - R: read.csv('word_summary.csv')")
    print("  - SQL: Import CSVs into a database")
    print("  - Excel/Google Sheets: Open directly")


if __name__ == "__main__":
    main()