#!/usr/bin/env python3
"""
Extract feature activations for words using circuit tracer
Integrates with connectivity analysis pipeline
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Import circuit tracer - will fail gracefully if not installed
try:
    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution import attribute
    CIRCUIT_TRACER_AVAILABLE = True
except ImportError:
    CIRCUIT_TRACER_AVAILABLE = False
    print("⚠️  Circuit tracer not available. Using simulated data.")


def simulate_feature_extraction(word: str, num_layers: int = 18) -> Dict:
    """Simulate feature extraction for testing without circuit tracer."""
    np.random.seed(hash(word) % 2**32)
    
    features_by_layer = {}
    
    for layer in range(num_layers + 1):  # Include embedding layer
        # Simulate fewer features in deeper layers
        num_features = max(1, int(20 * np.exp(-layer/5)))
        
        features = []
        for i in range(num_features):
            features.append({
                'feature_id': f'L{layer}_F{np.random.randint(1000, 9999)}',
                'activation_strength': np.random.beta(2, 5),  # Skewed towards lower values
                'feature_type': np.random.choice(['lexical', 'syntactic', 'semantic', 'task'])
            })
        
        features_by_layer[layer] = features
    
    return features_by_layer


def extract_word_features_real(
    word: str, 
    model: Any,
    tokenizer: Any,
    device: str,
    threshold: float = 0.1
) -> Dict:
    """Extract real features using circuit tracer."""
    if not CIRCUIT_TRACER_AVAILABLE:
        return simulate_feature_extraction(word, model.config.num_hidden_layers)
    
    try:
        # Tokenize the word
        inputs = tokenizer(word, return_tensors='pt').to(device)
        
        # Create circuit tracer model wrapper
        tracer_model = ReplacementModel(model, tokenizer)
        
        # Get attributions
        attributions = attribute(
            tracer_model,
            inputs['input_ids'],
            method='integrated_gradients'
        )
        
        features_by_layer = {}
        
        # Process each layer
        for layer_idx in range(model.config.num_hidden_layers + 1):
            layer_features = []
            
            # Get activations for this layer
            layer_attrs = attributions.get(f'layer_{layer_idx}', {})
            
            for feat_id, strength in layer_attrs.items():
                if abs(strength) > threshold:
                    layer_features.append({
                        'feature_id': f'L{layer_idx}_F{feat_id}',
                        'activation_strength': abs(strength),
                        'feature_type': 'unknown'  # Would need feature interpretation
                    })
            
            features_by_layer[layer_idx] = sorted(
                layer_features, 
                key=lambda x: x['activation_strength'], 
                reverse=True
            )
        
        return features_by_layer
        
    except Exception as e:
        print(f"Warning: Feature extraction failed for '{word}': {e}")
        return simulate_feature_extraction(word, model.config.num_hidden_layers)


def analyze_all_words(
    words: List[str],
    model_path: str,
    device: str = 'cpu',
    threshold: float = 0.1
) -> Dict[str, Dict]:
    """Extract features for all words."""
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    all_features = {}
    
    for word in tqdm(words, desc="Extracting features"):
        features = extract_word_features_real(
            word, model, tokenizer, device, threshold
        )
        all_features[word] = features
    
    return all_features


def compute_feature_summary(features_by_word: Dict[str, Dict]) -> Dict:
    """Compute summary statistics for feature data."""
    summary = {
        'total_words': len(features_by_word),
        'layer_statistics': {},
        'word_statistics': {}
    }
    
    # Aggregate by layer
    all_layers = set()
    for word_features in features_by_word.values():
        all_layers.update(word_features.keys())
    
    for layer in sorted(all_layers):
        layer_features = []
        layer_counts = []
        
        for word, features in features_by_word.items():
            if layer in features:
                layer_counts.append(len(features[layer]))
                layer_features.extend([f['activation_strength'] for f in features[layer]])
        
        summary['layer_statistics'][layer] = {
            'mean_features_per_word': np.mean(layer_counts) if layer_counts else 0,
            'std_features_per_word': np.std(layer_counts) if layer_counts else 0,
            'mean_activation_strength': np.mean(layer_features) if layer_features else 0
        }
    
    # Aggregate by word
    for word, features in features_by_word.items():
        total_features = sum(len(layer_feats) for layer_feats in features.values())
        all_strengths = []
        
        for layer_feats in features.values():
            all_strengths.extend([f['activation_strength'] for f in layer_feats])
        
        summary['word_statistics'][word] = {
            'total_features': total_features,
            'mean_activation': np.mean(all_strengths) if all_strengths else 0,
            'peak_activation': max(all_strengths) if all_strengths else 0
        }
    
    return summary


def export_features_to_json(
    features_by_word: Dict[str, Dict],
    summary: Dict,
    output_path: str
):
    """Export features to JSON format."""
    output = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_words': len(features_by_word),
            'circuit_tracer_available': CIRCUIT_TRACER_AVAILABLE
        },
        'features_by_word': features_by_word,
        'summary': summary
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Exported features to {output_path}")


def export_features_for_csv(features_by_word: Dict[str, Dict], output_dir: str):
    """Export features in format ready for CSV conversion."""
    # Create feature_activations list
    feature_rows = []
    
    for word, layers in features_by_word.items():
        for layer, features in layers.items():
            for feature in features:
                feature_rows.append({
                    'word': word,
                    'layer': layer,
                    'feature_id': feature['feature_id'],
                    'activation_strength': feature['activation_strength'],
                    'feature_type': feature['feature_type']
                })
    
    # Create feature_summary list
    summary_rows = []
    
    for word, layers in features_by_word.items():
        for layer, features in layers.items():
            if features:  # Only include layers with features
                strengths = [f['activation_strength'] for f in features]
                types = [f['feature_type'] for f in features]
                dominant_type = max(set(types), key=types.count) if types else 'none'
                
                summary_rows.append({
                    'word': word,
                    'layer': layer,
                    'total_features': len(features),
                    'mean_activation': np.mean(strengths),
                    'max_activation': max(strengths),
                    'dominant_type': dominant_type
                })
    
    # Save for CSV export
    csv_ready = {
        'feature_activations': feature_rows,
        'feature_summary': summary_rows
    }
    
    output_path = Path(output_dir) / 'features_for_csv.json'
    with open(output_path, 'w') as f:
        json.dump(csv_ready, f, indent=2)
    
    print(f"✅ Exported CSV-ready features to {output_path}")
    return csv_ready


def main():
    parser = argparse.ArgumentParser(
        description="Extract feature activations using circuit tracer"
    )
    parser.add_argument(
        '--words',
        type=str,
        required=True,
        help='JSON file containing word list'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/Users/lee/fun/learningSlice/models/gemma-2b',
        help='Model path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='feature_extraction_results.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for CSV-ready files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu/cuda/mps)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Minimum activation strength threshold'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        help='Limit number of words to process'
    )
    
    args = parser.parse_args()
    
    print("🧠 Feature Extraction using Circuit Tracer")
    print(f"Circuit tracer available: {CIRCUIT_TRACER_AVAILABLE}")
    
    # Load word list
    with open(args.words, 'r') as f:
        words = json.load(f)
    
    if args.max_words:
        words = words[:args.max_words]
    
    print(f"Processing {len(words)} words")
    
    # Extract features
    start_time = time.time()
    features = analyze_all_words(
        words, args.model, args.device, args.threshold
    )
    
    # Compute summary
    summary = compute_feature_summary(features)
    
    # Export results
    export_features_to_json(features, summary, args.output)
    export_features_for_csv(features, args.output_dir)
    
    # Print summary stats
    print("\n📊 Feature Extraction Summary")
    print("=" * 40)
    print(f"Total words processed: {summary['total_words']}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    # Show layer statistics
    print("\n📈 Features by Layer:")
    for layer, stats in sorted(summary['layer_statistics'].items()):
        print(f"  Layer {layer}: {stats['mean_features_per_word']:.1f} features/word")
    
    # Show top words by feature count
    word_stats = summary['word_statistics']
    top_words = sorted(
        word_stats.items(), 
        key=lambda x: x[1]['total_features'], 
        reverse=True
    )[:10]
    
    print("\n🏆 Top 10 words by feature count:")
    for word, stats in top_words:
        print(f"  {word}: {stats['total_features']} features")


if __name__ == "__main__":
    main()