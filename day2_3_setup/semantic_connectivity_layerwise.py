#!/usr/bin/env python3
"""
Layer-wise Semantic Connectivity Analysis
Computes semantic connectivity at each layer for correlation with feature activation and circuit complexity
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_word_list(filepath: str) -> List[str]:
    """Load word list from JSON file."""
    try:
        with open(filepath, 'r') as f:
            words = json.load(f)
        
        if not isinstance(words, list):
            raise ValueError("JSON file must contain a list of words")
        
        cleaned_words = []
        for word in words:
            if isinstance(word, str) and len(word.strip()) > 0:
                cleaned_words.append(word.strip().lower())
        
        print(f"Loaded {len(cleaned_words)} words from {filepath}")
        return cleaned_words
        
    except Exception as e:
        print(f"Error loading word list from {filepath}: {e}")
        sys.exit(1)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1 = emb1.float().flatten()
    emb2 = emb2.float().flatten()
    
    if emb1.norm() == 0 or emb2.norm() == 0:
        return 0.0
    
    similarity = torch.dot(emb1, emb2) / (emb1.norm() * emb2.norm())
    return similarity.item()


def get_word_embedding_at_layer(
    word: str, 
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    device: str,
    layer: int
) -> torch.Tensor:
    """Get embedding for a word at a specific layer."""
    try:
        tokens = tokenizer(word, return_tensors='pt', padding=False, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        if tokens['input_ids'].shape[1] == 0:
            return torch.zeros(model.config.hidden_size).to(device)
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            # Extract from specific layer
            word_emb = outputs.hidden_states[layer].mean(dim=1).squeeze()
        
        return word_emb
        
    except Exception as e:
        print(f"Error getting embedding for '{word}' at layer {layer}: {e}")
        return torch.zeros(model.config.hidden_size).to(device)


def compute_layer_connectivity(
    word: str,
    layer: int,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    vocab_sample: List[str],
    device: str,
    threshold: float = 0.7,
    sample_size: int = 1000
) -> Dict:
    """Compute semantic connectivity for a word at a specific layer."""
    try:
        word_emb = get_word_embedding_at_layer(word, model, tokenizer, device, layer)
        
        if word_emb.norm() == 0:
            return {
                'connectivity_count': 0,
                'mean_similarity': 0.0,
                'max_similarity': 0.0,
                'neighbors': []
            }
        
        # Sample vocabulary
        comparison_vocab = [w for w in vocab_sample if w != word]
        actual_sample_size = min(sample_size, len(comparison_vocab))
        sampled_vocab = random.sample(comparison_vocab, actual_sample_size)
        
        # Compute similarities
        similarities = []
        high_similarity_neighbors = []
        
        for other_word in sampled_vocab:
            try:
                other_emb = get_word_embedding_at_layer(other_word, model, tokenizer, device, layer)
                
                if other_emb.norm() == 0:
                    continue
                
                similarity = cosine_similarity(word_emb, other_emb)
                similarities.append(similarity)
                
                if similarity > threshold:
                    high_similarity_neighbors.append((other_word, similarity))
                    
            except Exception:
                continue
        
        # Sort neighbors by similarity
        high_similarity_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'connectivity_count': len(high_similarity_neighbors),
            'mean_similarity': np.mean(similarities) if similarities else 0.0,
            'max_similarity': max(similarities) if similarities else 0.0,
            'neighbors': high_similarity_neighbors[:10]  # Top 10 neighbors
        }
        
    except Exception as e:
        print(f"Error computing connectivity for '{word}' at layer {layer}: {e}")
        return {
            'connectivity_count': 0,
            'mean_similarity': 0.0,
            'max_similarity': 0.0,
            'neighbors': []
        }


def analyze_word_across_layers(
    word: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    vocab_sample: List[str],
    device: str,
    layers: Optional[List[int]] = None,
    threshold: float = 0.7,
    sample_size: int = 1000
) -> Dict:
    """Analyze semantic connectivity for a word across all layers."""
    if layers is None:
        # Use all layers
        num_layers = model.config.num_hidden_layers
        layers = list(range(num_layers + 1))  # Include embedding layer
    
    layer_results = {}
    
    for layer in layers:
        layer_results[f'layer_{layer}'] = compute_layer_connectivity(
            word, layer, model, tokenizer, vocab_sample, device, threshold, sample_size
        )
    
    # Compute summary statistics
    connectivity_counts = [r['connectivity_count'] for r in layer_results.values()]
    
    return {
        'word': word,
        'layer_results': layer_results,
        'summary': {
            'max_connectivity': max(connectivity_counts),
            'min_connectivity': min(connectivity_counts),
            'mean_connectivity': np.mean(connectivity_counts),
            'connectivity_evolution': connectivity_counts,
            'peak_layer': layers[np.argmax(connectivity_counts)]
        }
    }


def find_layer_specific_outliers(
    all_results: List[Dict],
    layer: int
) -> Dict:
    """Find outliers for a specific layer."""
    # Extract connectivity scores for the specified layer
    layer_key = f'layer_{layer}'
    scores = []
    
    for result in all_results:
        word = result['word']
        connectivity = result['layer_results'][layer_key]['connectivity_count']
        scores.append((word, connectivity))
    
    # Sort by connectivity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return {
        'top_20': sorted_scores[:20],
        'bottom_20': sorted_scores[-20:],
        'stats': {
            'mean': np.mean([s[1] for s in scores]),
            'std': np.std([s[1] for s in scores]),
            'median': np.median([s[1] for s in scores])
        }
    }


def save_checkpoint(filepath: str, data: dict):
    """Save checkpoint data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint(filepath: str) -> dict:
    """Load checkpoint data from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute layer-wise semantic connectivity for vocabulary sample"
    )
    parser.add_argument(
        '--words', 
        type=str, 
        required=True,
        help='JSON file containing list of words to analyze'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='google/gemma-2-2b',
        help='Model name or path (default: google/gemma-2-2b)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='layerwise_connectivity_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='layerwise_checkpoint.json',
        help='Checkpoint file for saving progress'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        help='Specific layers to analyze (default: all layers)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for connectivity (default: 0.7)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of words to sample for each comparison (default: 1000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        help='Maximum number of words to process (for testing)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Layer-wise Semantic Connectivity Analysis")
    print(f"Using device: {args.device}")
    
    # Load word list
    print(f"Loading words from: {args.words}")
    sample_words = load_word_list(args.words)
    
    if args.max_words:
        sample_words = sample_words[:args.max_words]
        print(f"Limited to {len(sample_words)} words for analysis")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(args.device)
        model.eval()
        print(f"✅ Model loaded successfully")
        print(f"Model has {model.config.num_hidden_layers} layers")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Determine layers to analyze
    if args.layers:
        layers = args.layers
    else:
        layers = list(range(model.config.num_hidden_layers + 1))
    
    print(f"Analyzing layers: {layers}")
    
    # Check for checkpoint
    all_results = []
    processed_words = set()
    
    if args.resume and os.path.exists(args.checkpoint):
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint:
            all_results = checkpoint.get('results', [])
            processed_words = set(r['word'] for r in all_results)
            print(f"📁 Resumed from checkpoint: {len(processed_words)} words already processed")
    
    # Process words
    remaining_words = [w for w in sample_words if w not in processed_words]
    
    print(f"🔄 Computing layer-wise connectivity for {len(remaining_words)} words...")
    print(f"📊 Using {args.sample_size} comparison samples per word, threshold {args.threshold}")
    
    try:
        for i, word in enumerate(tqdm(remaining_words, desc="Processing words")):
            result = analyze_word_across_layers(
                word, model, tokenizer, sample_words, args.device, 
                layers, args.threshold, args.sample_size
            )
            all_results.append(result)
            
            # Save checkpoint every 50 words
            if (i + 1) % 50 == 0:
                save_checkpoint(args.checkpoint, {'results': all_results})
        
        # Final save
        save_checkpoint(args.checkpoint, {'results': all_results})
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving checkpoint...")
        save_checkpoint(args.checkpoint, {'results': all_results})
        sys.exit(1)
    
    # Analyze results
    print("\n🔍 Analyzing layer-specific patterns...")
    
    layer_analyses = {}
    for layer in layers:
        layer_analyses[f'layer_{layer}'] = find_layer_specific_outliers(all_results, layer)
    
    # Find words with interesting connectivity evolution
    connectivity_evolutions = []
    for result in all_results:
        evolution = result['summary']['connectivity_evolution']
        # Calculate variance in connectivity across layers
        variance = np.var(evolution)
        peak_layer = result['summary']['peak_layer']
        connectivity_evolutions.append({
            'word': result['word'],
            'variance': variance,
            'peak_layer': peak_layer,
            'evolution': evolution
        })
    
    # Sort by variance to find words with most dramatic changes
    connectivity_evolutions.sort(key=lambda x: x['variance'], reverse=True)
    
    # Prepare final results
    results = {
        'metadata': {
            'model': args.model,
            'word_source': args.words,
            'total_words': len(sample_words),
            'analyzed_words': len(all_results),
            'layers_analyzed': layers,
            'threshold': args.threshold,
            'sample_size': args.sample_size,
            'device': args.device
        },
        'word_results': all_results,
        'layer_analyses': layer_analyses,
        'connectivity_evolution': {
            'highest_variance': connectivity_evolutions[:20],
            'lowest_variance': connectivity_evolutions[-20:]
        }
    }
    
    # Save results
    print(f"\n💾 Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 LAYER-WISE CONNECTIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total words analyzed: {len(all_results)}")
    print(f"Layers analyzed: {len(layers)}")
    
    # Print layer-specific stats
    print("\n📈 Connectivity by Layer:")
    for layer in layers[:10]:  # Show first 10 layers
        stats = layer_analyses[f'layer_{layer}']['stats']
        print(f"  Layer {layer:2d}: mean={stats['mean']:5.1f}, std={stats['std']:5.1f}")
    
    # Print words with highest variance
    print("\n🎢 Words with highest connectivity variance across layers:")
    for i, item in enumerate(connectivity_evolutions[:10], 1):
        print(f"  {i:2d}. {item['word']:15}: variance={item['variance']:6.1f}, peak at layer {item['peak_layer']}")
    
    # Clean up checkpoint
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"\n🗑️ Removed checkpoint file: {args.checkpoint}")
    
    print(f"\n✅ Layer-wise semantic connectivity analysis completed!")
    print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()