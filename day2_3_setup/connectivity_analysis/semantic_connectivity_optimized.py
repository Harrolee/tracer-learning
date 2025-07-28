#!/usr/bin/env python3
"""
Optimized Layer-wise Semantic Connectivity Analysis
Pre-computes all embeddings for efficient similarity computation
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

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


def precompute_embeddings(
    words: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    layers: List[int],
    device: str,
    batch_size: int = 32
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Pre-compute embeddings for all words at all specified layers.
    
    Returns:
        Dictionary mapping word -> layer -> embedding tensor
    """
    embeddings = {}
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(words), batch_size), desc="Pre-computing embeddings"):
        batch_words = words[i:i + batch_size]
        
        # Tokenize batch
        tokens = tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            
            # Extract embeddings for each word and layer
            for word_idx, word in enumerate(batch_words):
                if word not in embeddings:
                    embeddings[word] = {}
                
                # Get attention mask for this word to compute mean only over actual tokens
                mask = tokens['attention_mask'][word_idx]
                
                for layer_idx in layers:
                    # Get embeddings for this layer
                    layer_output = outputs.hidden_states[layer_idx][word_idx]
                    
                    # Mean pooling over tokens (excluding padding)
                    if mask.sum() > 0:
                        word_embedding = (layer_output * mask.unsqueeze(-1)).sum(dim=0) / mask.sum()
                    else:
                        word_embedding = torch.zeros(model.config.hidden_size).to(device)
                    
                    embeddings[word][layer_idx] = word_embedding
    
    return embeddings


def compute_connectivity_from_embeddings(
    word: str,
    layer: int,
    embeddings: Dict[str, Dict[int, torch.Tensor]],
    threshold: float = 0.7
) -> Dict:
    """Compute connectivity metrics using pre-computed embeddings."""
    if word not in embeddings or layer not in embeddings[word]:
        return {
            'connectivity_count': 0,
            'mean_similarity': 0.0,
            'max_similarity': 0.0,
            'top_neighbors': []
        }
    
    word_emb = embeddings[word][layer]
    
    if word_emb.norm() == 0:
        return {
            'connectivity_count': 0,
            'mean_similarity': 0.0,
            'max_similarity': 0.0,
            'top_neighbors': []
        }
    
    similarities = []
    high_similarity_neighbors = []
    
    # Compare with all other words
    for other_word, other_embeddings in embeddings.items():
        if other_word == word or layer not in other_embeddings:
            continue
        
        other_emb = other_embeddings[layer]
        if other_emb.norm() == 0:
            continue
        
        # Cosine similarity
        similarity = torch.dot(word_emb, other_emb) / (word_emb.norm() * other_emb.norm())
        similarity = similarity.item()
        similarities.append(similarity)
        
        if similarity > threshold:
            high_similarity_neighbors.append((other_word, similarity))
    
    # Sort neighbors by similarity
    high_similarity_neighbors.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'connectivity_count': len(high_similarity_neighbors),
        'mean_similarity': np.mean(similarities) if similarities else 0.0,
        'max_similarity': max(similarities) if similarities else 0.0,
        'top_neighbors': high_similarity_neighbors[:10]
    }


def analyze_word_evolution(
    word: str,
    embeddings: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    threshold: float = 0.7
) -> Dict:
    """Analyze connectivity evolution for a word across layers."""
    layer_results = {}
    connectivity_trajectory = []
    
    for layer in layers:
        result = compute_connectivity_from_embeddings(
            word, layer, embeddings, threshold
        )
        layer_results[f'layer_{layer}'] = result
        connectivity_trajectory.append(result['connectivity_count'])
    
    # Compute evolution metrics
    evolution_metrics = {
        'trajectory': connectivity_trajectory,
        'max_connectivity': max(connectivity_trajectory) if connectivity_trajectory else 0,
        'min_connectivity': min(connectivity_trajectory) if connectivity_trajectory else 0,
        'mean_connectivity': np.mean(connectivity_trajectory) if connectivity_trajectory else 0,
        'variance': np.var(connectivity_trajectory) if connectivity_trajectory else 0,
        'peak_layer': layers[np.argmax(connectivity_trajectory)] if connectivity_trajectory else -1,
        'trough_layer': layers[np.argmin(connectivity_trajectory)] if connectivity_trajectory else -1,
        'stability': 1.0 / (1.0 + np.var(connectivity_trajectory)) if connectivity_trajectory else 0
    }
    
    return {
        'word': word,
        'layer_results': layer_results,
        'evolution': evolution_metrics,
        'layers_analyzed': layers
    }


def find_evolution_outliers(all_results: List[Dict]) -> Dict[str, List]:
    """Identify words with interesting connectivity evolution patterns."""
    # Extract key metrics for sorting
    by_max_connectivity = [(r['word'], r['evolution']['max_connectivity']) for r in all_results]
    by_variance = [(r['word'], r['evolution']['variance']) for r in all_results]
    by_stability = [(r['word'], r['evolution']['stability']) for r in all_results]
    
    # Sort by different criteria
    by_max_connectivity.sort(key=lambda x: x[1], reverse=True)
    by_variance.sort(key=lambda x: x[1], reverse=True)
    by_stability.sort(key=lambda x: x[1], reverse=True)
    
    # Get outliers
    result = {
        'highest_connectivity': by_max_connectivity[:50],
        'lowest_connectivity': by_max_connectivity[-50:],
        'highest_variance': by_variance[:50],
        'most_stable': by_stability[:50],
        'total_analyzed': len(all_results)
    }
    
    # Add words that peak in different layers
    peak_layers = {}
    for r in all_results:
        peak = r['evolution']['peak_layer']
        if peak not in peak_layers:
            peak_layers[peak] = []
        peak_layers[peak].append((r['word'], r['evolution']['max_connectivity']))
    
    result['peak_by_layer'] = {
        f'layer_{layer}': sorted(words, key=lambda x: x[1], reverse=True)[:10]
        for layer, words in peak_layers.items()
    }
    
    return result


def analyze_polysemy_evolution_relationship(
    all_results: List[Dict],
    polysemy_scores_file: str = None
) -> Dict:
    """Analyze relationship between polysemy and connectivity evolution."""
    # Basic evolution stats
    evolution_stats = {
        'total_words': len(all_results),
        'mean_max_connectivity': np.mean([r['evolution']['max_connectivity'] for r in all_results]),
        'mean_variance': np.mean([r['evolution']['variance'] for r in all_results]),
        'mean_stability': np.mean([r['evolution']['stability'] for r in all_results])
    }
    
    analysis = {'evolution_stats': evolution_stats}
    
    # Add polysemy analysis if data available
    if polysemy_scores_file and os.path.exists(polysemy_scores_file):
        try:
            with open(polysemy_scores_file, 'r') as f:
                polysemy_data = json.load(f)
            
            # Group results by polysemy level
            polysemy_groups = {
                'monosemous': [],
                'low_polysemy': [],
                'medium_polysemy': [],
                'high_polysemy': []
            }
            
            for result in all_results:
                word = result['word']
                polysemy = polysemy_data.get(word, 1)
                
                if polysemy == 1:
                    polysemy_groups['monosemous'].append(result)
                elif polysemy <= 3:
                    polysemy_groups['low_polysemy'].append(result)
                elif polysemy <= 10:
                    polysemy_groups['medium_polysemy'].append(result)
                else:
                    polysemy_groups['high_polysemy'].append(result)
            
            # Calculate evolution patterns for each polysemy group
            polysemy_evolution = {}
            for group, results in polysemy_groups.items():
                if results:
                    polysemy_evolution[group] = {
                        'count': len(results),
                        'mean_max_connectivity': np.mean([r['evolution']['max_connectivity'] for r in results]),
                        'mean_variance': np.mean([r['evolution']['variance'] for r in results]),
                        'mean_stability': np.mean([r['evolution']['stability'] for r in results]),
                        'mean_peak_layer': np.mean([r['evolution']['peak_layer'] for r in results])
                    }
            
            analysis['polysemy_evolution'] = polysemy_evolution
            
        except Exception as e:
            print(f"Warning: Could not load polysemy analysis: {e}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Optimized layer-wise semantic connectivity analysis"
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
        help='Model name or path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='optimized_connectivity_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        help='Specific layers to analyze (default: sample across model)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for connectivity (default: 0.7)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding computation (default: 32)'
    )
    parser.add_argument(
        '--polysemy-file',
        type=str,
        help='Optional JSON file with polysemy scores for analysis'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        help='Maximum number of words to process (for testing)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Optimized Layer-wise Semantic Connectivity Analysis")
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
        # Sample layers: embedding, early, middle, late
        num_layers = model.config.num_hidden_layers
        layers = [
            0,  # Embedding layer
            num_layers // 4,  # Early
            num_layers // 2,  # Middle
            3 * num_layers // 4,  # Late-middle
            num_layers  # Final layer
        ]
    
    print(f"Analyzing layers: {layers}")
    
    # Pre-compute all embeddings
    print(f"\n📊 Pre-computing embeddings for {len(sample_words)} words at {len(layers)} layers...")
    start_time = time.time()
    
    embeddings = precompute_embeddings(
        sample_words, model, tokenizer, layers, args.device, args.batch_size
    )
    
    embedding_time = time.time() - start_time
    print(f"✅ Embeddings computed in {embedding_time:.2f} seconds")
    
    # Analyze connectivity evolution for each word
    print(f"\n🔄 Computing connectivity evolution...")
    all_results = []
    
    for word in tqdm(sample_words, desc="Analyzing connectivity"):
        result = analyze_word_evolution(
            word, embeddings, layers, args.threshold
        )
        all_results.append(result)
    
    # Find evolution outliers
    print("\n🔍 Identifying connectivity evolution patterns...")
    outliers = find_evolution_outliers(all_results)
    
    # Analyze connectivity patterns with polysemy
    print("📈 Analyzing connectivity evolution patterns...")
    analysis = analyze_polysemy_evolution_relationship(
        all_results, args.polysemy_file
    )
    
    # Prepare results
    results = {
        'metadata': {
            'model': args.model,
            'word_source': args.words,
            'total_words': len(sample_words),
            'threshold': args.threshold,
            'device': args.device,
            'layers_analyzed': layers,
            'total_time': time.time() - start_time,
            'embedding_time': embedding_time
        },
        'outliers': outliers,
        'analysis': analysis,
        'word_results': all_results
    }
    
    # Save results
    print(f"\n💾 Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    stats = analysis['evolution_stats']
    print("\n" + "="*60)
    print("📊 LAYER-WISE CONNECTIVITY EVOLUTION SUMMARY")
    print("="*60)
    print(f"Total words processed: {stats['total_words']:,}")
    print(f"Mean max connectivity: {stats['mean_max_connectivity']:.2f}")
    print(f"Mean variance across layers: {stats['mean_variance']:.2f}")
    print(f"Mean stability: {stats['mean_stability']:.2f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    # Print polysemy analysis if available
    if 'polysemy_evolution' in analysis:
        print(f"\n📚 POLYSEMY vs CONNECTIVITY EVOLUTION")
        print("-" * 40)
        for group, stats in analysis['polysemy_evolution'].items():
            print(f"{group.replace('_', ' ').title():15}: {stats['count']:4d} words, "
                  f"max conn: {stats['mean_max_connectivity']:5.1f}, "
                  f"variance: {stats['mean_variance']:5.1f}")
    
    print(f"\n🏆 Top 10 highest connectivity words:")
    for i, (word, score) in enumerate(outliers['highest_connectivity'][:10], 1):
        print(f"  {i:2d}. {word:15}: {score:3.0f} max connections")
    
    print(f"\n🎢 Top 10 most dynamic words (highest variance):")
    for i, (word, variance) in enumerate(outliers['highest_variance'][:10], 1):
        print(f"  {i:2d}. {word:15}: variance={variance:6.1f}")
    
    print(f"\n✅ Analysis completed!")


if __name__ == "__main__":
    main()