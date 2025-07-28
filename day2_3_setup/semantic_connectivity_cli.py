#!/usr/bin/env python3
"""
Layer-wise Semantic Connectivity Analysis CLI - Day 2-3 Implementation
Research Plan: Semantic Connectivity Evolution vs Circuit Complexity

This script computes layer-wise semantic connectivity patterns for vocabulary samples
to understand how word representations evolve through model layers and identify
connectivity patterns that may predict circuit complexity.


Lee,you c/p this text from Claude code before leaving for shuv coffee:
1. Revised semantic_connectivity_cli.py to compute
  layer-wise connectivity evolution instead of just
  using the last hidden layer
  2. Updated the research plan to reflect the new
  approach

  Key changes made:

  In the code:
  - Added get_word_embedding_at_layer() to extract
  embeddings from specific layers
  - Created compute_layer_connectivity() for
  layer-specific analysis
  - Added analyze_word_across_layers() to track
  connectivity evolution
  - New metrics: variance, peak layer, stability,
  trajectory
  - Updated output to identify words with interesting
  evolution patterns

  In the research plan:
  - Shifted focus from polysemy-connectivity
  correlation to connectivity evolution patterns
  - New hypothesis: dynamic patterns (high variance) →
  complex circuits
  - Layer-specific predictions about processing
  strategies
  - Updated analysis plan to compare evolution metrics
  vs single-layer metrics

  The new approach will give you much richer data about
   how words' semantic neighborhoods change through the
   model, which should correlate better with circuit
  complexity than a single snapshot at the final layer.


"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_word_list(filepath: str) -> List[str]:
    """
    Load word list from JSON file.
    
    Args:
        filepath: Path to JSON file containing list of words
    
    Returns:
        List of words for analysis
    """
    try:
        with open(filepath, 'r') as f:
            words = json.load(f)
        
        if not isinstance(words, list):
            raise ValueError("JSON file must contain a list of words")
        
        # Filter and clean words
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


def get_word_embedding_at_layer(word: str, model: AutoModel, tokenizer: AutoTokenizer, device: str, layer: int) -> torch.Tensor:
    """
    Get embedding for a single word at a specific layer.
    
    Args:
        word: Word to get embedding for
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        layer: Layer index (0 = embedding layer, n = nth transformer layer)
    
    Returns:
        Word embedding tensor
    """
    try:
        # Tokenize word
        tokens = tokenizer(word, return_tensors='pt', padding=False, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        if tokens['input_ids'].shape[1] == 0:
            return torch.zeros(model.config.hidden_size).to(device)
        
        with torch.no_grad():
            # Get embeddings from the model
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
    """
    Compute semantic connectivity for a word at a specific layer.
    
    Args:
        word: Word to analyze
        layer: Layer index
        model: Language model
        tokenizer: Tokenizer
        vocab_sample: Sample of vocabulary tokens to compare against
        device: Device to run on
        threshold: Similarity threshold for considering words connected
        sample_size: Number of words to sample for comparison
    
    Returns:
        Dictionary with connectivity metrics
    """
    try:
        # Get word embedding at specific layer
        word_emb = get_word_embedding_at_layer(word, model, tokenizer, device, layer)
        
        if word_emb.norm() == 0:
            return {
                'connectivity_count': 0,
                'mean_similarity': 0.0,
                'max_similarity': 0.0,
                'top_neighbors': []
            }
        
        # Sample vocabulary for comparison (exclude the word itself)
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
            'top_neighbors': high_similarity_neighbors[:10]
        }
        
    except Exception as e:
        print(f"Error processing word '{word}' at layer {layer}: {e}")
        return {
            'connectivity_count': 0,
            'mean_similarity': 0.0,
            'max_similarity': 0.0,
            'top_neighbors': []
        }


def analyze_word_across_layers(
    word: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    vocab_sample: List[str],
    device: str,
    layers_to_analyze: List[int] = None,
    threshold: float = 0.7,
    sample_size: int = 1000
) -> Dict:
    """
    Analyze semantic connectivity for a word across multiple layers.
    
    Args:
        word: Word to analyze
        model: Language model
        tokenizer: Tokenizer
        vocab_sample: Sample of vocabulary tokens
        device: Device to run on
        layers_to_analyze: Specific layers to analyze (None = sample across model)
        threshold: Similarity threshold
        sample_size: Number of comparison words
    
    Returns:
        Dictionary with layer-wise results and evolution metrics
    """
    if layers_to_analyze is None:
        # Sample layers: embedding, early, middle, late
        num_layers = model.config.num_hidden_layers
        layers_to_analyze = [
            0,  # Embedding layer
            num_layers // 4,  # Early
            num_layers // 2,  # Middle
            3 * num_layers // 4,  # Late-middle
            num_layers  # Final layer
        ]
    
    layer_results = {}
    connectivity_trajectory = []
    
    for layer in layers_to_analyze:
        result = compute_layer_connectivity(
            word, layer, model, tokenizer, vocab_sample, device, threshold, sample_size
        )
        layer_results[f'layer_{layer}'] = result
        connectivity_trajectory.append(result['connectivity_count'])
    
    # Compute evolution metrics
    evolution_metrics = {
        'trajectory': connectivity_trajectory,
        'max_connectivity': max(connectivity_trajectory),
        'min_connectivity': min(connectivity_trajectory),
        'mean_connectivity': np.mean(connectivity_trajectory),
        'variance': np.var(connectivity_trajectory),
        'peak_layer': layers_to_analyze[np.argmax(connectivity_trajectory)],
        'trough_layer': layers_to_analyze[np.argmin(connectivity_trajectory)],
        'stability': 1.0 / (1.0 + np.var(connectivity_trajectory))  # Higher = more stable
    }
    
    return {
        'word': word,
        'layer_results': layer_results,
        'evolution': evolution_metrics,
        'layers_analyzed': layers_to_analyze
    }


def find_evolution_outliers(
    all_results: List[Dict]
) -> Dict[str, List]:
    """
    Identify words with interesting connectivity evolution patterns.
    
    Args:
        all_results: List of word analysis results
    
    Returns:
        Dictionary with different types of outliers
    """
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
        'highest_variance': by_variance[:50],  # Most dynamic
        'most_stable': by_stability[:50],  # Most consistent across layers
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
    """
    Analyze relationship between polysemy and connectivity evolution patterns.
    
    Args:
        all_results: List of word analysis results with layer-wise data
        polysemy_scores_file: Optional path to polysemy scores JSON file
    
    Returns:
        Analysis results dictionary
    """
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
            
            # Check if polysemy correlates with evolution patterns
            if len(all_results) > 10:
                from scipy import stats
                polysemy_values = []
                max_connectivity_values = []
                variance_values = []
                
                for result in all_results:
                    word = result['word']
                    if word in polysemy_data:
                        polysemy_values.append(polysemy_data[word])
                        max_connectivity_values.append(result['evolution']['max_connectivity'])
                        variance_values.append(result['evolution']['variance'])
                
                if len(polysemy_values) > 10:
                    # Compute correlations
                    corr_connectivity, p_connectivity = stats.pearsonr(polysemy_values, max_connectivity_values)
                    corr_variance, p_variance = stats.pearsonr(polysemy_values, variance_values)
                    
                    analysis['correlations'] = {
                        'polysemy_vs_max_connectivity': {
                            'correlation': corr_connectivity,
                            'p_value': p_connectivity
                        },
                        'polysemy_vs_variance': {
                            'correlation': corr_variance,
                            'p_value': p_variance
                        }
                    }
            
        except Exception as e:
            print(f"Warning: Could not load polysemy analysis: {e}")
    
    return analysis


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
        description="Compute layer-wise semantic connectivity evolution for vocabulary samples"
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
        default='semantic_connectivity_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='semantic_connectivity_checkpoint.json',
        help='Checkpoint file for saving progress'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
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
        '--polysemy-file',
        type=str,
        help='Optional JSON file with polysemy scores for analysis'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Day 2-3: Layer-wise Semantic Connectivity Analysis")
    print(f"Using device: {args.device}")
    
    # Load word list
    print(f"Loading words from: {args.words}")
    sample_words = load_word_list(args.words)
    
    if len(sample_words) == 0:
        print("Error: No words to process!")
        sys.exit(1)
    
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
                threshold=args.threshold, sample_size=args.sample_size
            )
            all_results.append(result)
            
            # Save checkpoint every 100 words
            if (i + 1) % 100 == 0:
                save_checkpoint(args.checkpoint, {'results': all_results})
        
        # Final save
        save_checkpoint(args.checkpoint, {'results': all_results})
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving checkpoint...")
        save_checkpoint(args.checkpoint, {'results': all_results})
        sys.exit(1)
    
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
            'sample_size': args.sample_size,
            'device': args.device,
            'total_processed': len(all_results),
            'layers_analyzed': all_results[0]['layers_analyzed'] if all_results else []
        },
        'outliers': outliers,
        'analysis': analysis,
        'word_results': all_results  # Full layer-wise data for each word
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
    
    # Print polysemy analysis if available
    if 'polysemy_evolution' in analysis:
        print(f"\n📚 POLYSEMY vs CONNECTIVITY EVOLUTION")
        print("-" * 40)
        for group, stats in analysis['polysemy_evolution'].items():
            print(f"{group.replace('_', ' ').title():15}: {stats['count']:4d} words, "
                  f"max conn: {stats['mean_max_connectivity']:5.1f}, "
                  f"variance: {stats['mean_variance']:5.1f}")
    
    # Print correlations if available
    if 'correlations' in analysis:
        print(f"\n📊 CORRELATIONS")
        print("-" * 40)
        corr = analysis['correlations']
        if 'polysemy_vs_max_connectivity' in corr:
            r = corr['polysemy_vs_max_connectivity']['correlation']
            p = corr['polysemy_vs_max_connectivity']['p_value']
            print(f"Polysemy vs Max Connectivity: r={r:.3f}, p={p:.3f}")
        if 'polysemy_vs_variance' in corr:
            r = corr['polysemy_vs_variance']['correlation']
            p = corr['polysemy_vs_variance']['p_value']
            print(f"Polysemy vs Variance: r={r:.3f}, p={p:.3f}")
    
    print(f"\n🏆 Top 10 highest connectivity words:")
    for i, (word, score) in enumerate(outliers['highest_connectivity'][:10], 1):
        print(f"  {i:2d}. {word:15}: {score:3.0f} max connections")
    
    print(f"\n🎢 Top 10 most dynamic words (highest variance):")
    for i, (word, variance) in enumerate(outliers['highest_variance'][:10], 1):
        print(f"  {i:2d}. {word:15}: variance={variance:6.1f}")
    
    # Print peak layer distribution
    if 'peak_by_layer' in outliers:
        print(f"\n📍 Peak connectivity by layer:")
        for layer_key, words in sorted(outliers['peak_by_layer'].items()):
            if words:
                layer_num = int(layer_key.split('_')[1])
                print(f"  Layer {layer_num:2d}: {len(words)} words peak here")
    
    # Clean up checkpoint if successful
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"\n🗑️ Removed checkpoint file: {args.checkpoint}")
    
    print(f"\n✅ Day 2-3 semantic connectivity analysis completed!")
    print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 