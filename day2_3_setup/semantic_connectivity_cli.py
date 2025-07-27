#!/usr/bin/env python3
"""
Semantic Connectivity Analysis CLI - Day 2-3 Implementation
Research Plan: Polysemy-Based Semantic Connectivity vs Circuit Complexity

This script computes semantic connectivity scores for a polysemy-based vocabulary sample
and identifies outliers for circuit complexity analysis.
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


def get_word_embedding(word: str, model: AutoModel, tokenizer: AutoTokenizer, device: str) -> torch.Tensor:
    """
    Get embedding for a single word using the model.
    
    Args:
        word: Word to get embedding for
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
    
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
            # Use the last hidden state's mean as word embedding
            word_emb = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return word_emb
        
    except Exception as e:
        print(f"Error getting embedding for '{word}': {e}")
        return torch.zeros(model.config.hidden_size).to(device)


def semantic_connectivity(
    word: str, 
    model: AutoModel, 
    tokenizer: AutoTokenizer,
    vocab_sample: List[str],
    device: str,
    threshold: float = 0.7,
    sample_size: int = 1000
) -> int:
    """
    Compute semantic connectivity score for a word.
    
    Args:
        word: Word to analyze
        model: Language model
        tokenizer: Tokenizer
        vocab_sample: Sample of vocabulary tokens to compare against
        device: Device to run on
        threshold: Similarity threshold for considering words connected
        sample_size: Number of words to sample for comparison
    
    Returns:
        Connectivity score (number of high-similarity neighbors)
    """
    try:
        # Get word embedding
        word_emb = get_word_embedding(word, model, tokenizer, device)
        
        if word_emb.norm() == 0:
            return 0
        
        # Sample vocabulary for comparison (exclude the word itself)
        comparison_vocab = [w for w in vocab_sample if w != word]
        actual_sample_size = min(sample_size, len(comparison_vocab))
        sampled_vocab = random.sample(comparison_vocab, actual_sample_size)
        
        # Compute similarities
        high_similarity_count = 0
        
        for other_word in sampled_vocab:
            try:
                other_emb = get_word_embedding(other_word, model, tokenizer, device)
                
                if other_emb.norm() == 0:
                    continue
                
                similarity = cosine_similarity(word_emb, other_emb)
                
                if similarity > threshold:
                    high_similarity_count += 1
                    
            except Exception:
                continue
        
        return high_similarity_count
        
    except Exception as e:
        print(f"Error processing word '{word}': {e}")
        return 0


def find_connectivity_outliers(
    connectivity_scores: List[Tuple[str, int]]
) -> Dict[str, List]:
    """
    Identify connectivity outliers from scores.
    
    Args:
        connectivity_scores: List of (word, score) tuples
    
    Returns:
        Dictionary with outlier categories
    """
    # Sort by connectivity
    sorted_words = sorted(connectivity_scores, key=lambda x: x[1], reverse=True)
    
    # Get outliers
    result = {
        'top_50_connected': sorted_words[:50],
        'bottom_50_connected': sorted_words[-50:],
        'all_scores': sorted_words
    }
    
    # Add random sample if we have enough words
    if len(sorted_words) > 200:
        middle_words = sorted_words[100:-100]
        random_sample = random.sample(middle_words, min(100, len(middle_words)))
        result['random_100'] = sorted(random_sample, key=lambda x: x[1], reverse=True)
    else:
        result['random_100'] = []
    
    return result


def analyze_polysemy_connectivity_relationship(
    connectivity_scores: List[Tuple[str, int]],
    polysemy_scores_file: str = None
) -> Dict:
    """
    Analyze relationship between polysemy and connectivity if polysemy data available.
    
    Args:
        connectivity_scores: List of (word, connectivity_score) tuples
        polysemy_scores_file: Optional path to polysemy scores JSON file
    
    Returns:
        Analysis results dictionary
    """
    analysis = {
        'connectivity_stats': {
            'total_words': len(connectivity_scores),
            'mean_connectivity': np.mean([score for _, score in connectivity_scores]),
            'std_connectivity': np.std([score for _, score in connectivity_scores]),
            'min_connectivity': min(score for _, score in connectivity_scores),
            'max_connectivity': max(score for _, score in connectivity_scores),
            'median_connectivity': np.median([score for _, score in connectivity_scores])
        }
    }
    
    # Add polysemy analysis if data available
    if polysemy_scores_file and os.path.exists(polysemy_scores_file):
        try:
            with open(polysemy_scores_file, 'r') as f:
                polysemy_data = json.load(f)
            
            # Group by polysemy level
            polysemy_groups = {'monosemous': [], 'low_polysemy': [], 'medium_polysemy': [], 'high_polysemy': []}
            
            for word, connectivity in connectivity_scores:
                polysemy = polysemy_data.get(word, 1)
                
                if polysemy == 1:
                    polysemy_groups['monosemous'].append(connectivity)
                elif polysemy <= 3:
                    polysemy_groups['low_polysemy'].append(connectivity)
                elif polysemy <= 10:
                    polysemy_groups['medium_polysemy'].append(connectivity)
                else:
                    polysemy_groups['high_polysemy'].append(connectivity)
            
            # Calculate stats for each group
            polysemy_analysis = {}
            for group, connectivities in polysemy_groups.items():
                if connectivities:
                    polysemy_analysis[group] = {
                        'count': len(connectivities),
                        'mean_connectivity': np.mean(connectivities),
                        'std_connectivity': np.std(connectivities),
                        'median_connectivity': np.median(connectivities)
                    }
            
            analysis['polysemy_analysis'] = polysemy_analysis
            
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
        description="Compute semantic connectivity scores for polysemy-based vocabulary sample"
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
    
    print(f"🚀 Day 2-3: Semantic Connectivity Analysis")
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
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Check for checkpoint
    connectivity_scores = []
    processed_words = set()
    
    if args.resume and os.path.exists(args.checkpoint):
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint:
            connectivity_scores = [(w, s) for w, s in checkpoint.get('scores', [])]
            processed_words = set(w for w, _ in connectivity_scores)
            print(f"📁 Resumed from checkpoint: {len(processed_words)} words already processed")
    
    # Process words
    remaining_words = [w for w in sample_words if w not in processed_words]
    
    print(f"🔄 Computing semantic connectivity for {len(remaining_words)} words...")
    print(f"📊 Using {args.sample_size} comparison samples per word, threshold {args.threshold}")
    
    try:
        for i, word in enumerate(tqdm(remaining_words, desc="Processing words")):
            score = semantic_connectivity(
                word, model, tokenizer, sample_words, args.device, args.threshold, args.sample_size
            )
            connectivity_scores.append((word, score))
            
            # Save checkpoint every 100 words
            if (i + 1) % 100 == 0:
                save_checkpoint(args.checkpoint, {'scores': connectivity_scores})
        
        # Final save
        save_checkpoint(args.checkpoint, {'scores': connectivity_scores})
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving checkpoint...")
        save_checkpoint(args.checkpoint, {'scores': connectivity_scores})
        sys.exit(1)
    
    # Find outliers
    print("\n🔍 Identifying connectivity outliers...")
    outliers = find_connectivity_outliers(connectivity_scores)
    
    # Analyze connectivity patterns
    print("📈 Analyzing connectivity patterns...")
    analysis = analyze_polysemy_connectivity_relationship(
        connectivity_scores, args.polysemy_file
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
            'total_processed': len(connectivity_scores)
        },
        'outliers': {
            'top_50': outliers['top_50_connected'],
            'bottom_50': outliers['bottom_50_connected'],
            'random_100': outliers['random_100']
        },
        'analysis': analysis
    }
    
    # Save results
    print(f"\n💾 Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    stats = analysis['connectivity_stats']
    print("\n" + "="*60)
    print("📊 SEMANTIC CONNECTIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total words processed: {stats['total_words']:,}")
    print(f"Mean connectivity: {stats['mean_connectivity']:.2f}")
    print(f"Standard deviation: {stats['std_connectivity']:.2f}")
    print(f"Median connectivity: {stats['median_connectivity']:.2f}")
    print(f"Range: [{stats['min_connectivity']:.0f}, {stats['max_connectivity']:.0f}]")
    
    # Print polysemy analysis if available
    if 'polysemy_analysis' in analysis:
        print(f"\n📚 POLYSEMY vs CONNECTIVITY ANALYSIS")
        print("-" * 40)
        for group, stats in analysis['polysemy_analysis'].items():
            print(f"{group.replace('_', ' ').title():15}: {stats['count']:4d} words, "
                  f"mean connectivity: {stats['mean_connectivity']:5.2f}")
    
    print(f"\n🏆 Top 10 most connected words:")
    for i, (word, score) in enumerate(outliers['top_50_connected'][:10], 1):
        print(f"  {i:2d}. {word:15}: {score:3d} connections")
    
    print(f"\n🔻 Bottom 10 least connected words:")
    for i, (word, score) in enumerate(outliers['bottom_50_connected'][-10:], 1):
        print(f"  {i:2d}. {word:15}: {score:3d} connections")
    
    # Clean up checkpoint if successful
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"\n🗑️ Removed checkpoint file: {args.checkpoint}")
    
    print(f"\n✅ Day 2-3 semantic connectivity analysis completed!")
    print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 