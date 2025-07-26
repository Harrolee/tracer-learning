#!/usr/bin/env python3
"""
Semantic Connectivity Analysis CLI

This script computes semantic connectivity scores for a vocabulary sample
and identifies outliers for the research project.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
import torch
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading WordNet data...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def get_vocabulary_sample(sample_size: int = 5000) -> List[str]:
    """
    Get vocabulary sample from WordNet.
    
    Args:
        sample_size: Number of words to sample (default 5000)
    
    Returns:
        List of sampled words
    """
    ensure_nltk_data()
    
    # Use WordNet synsets as comprehensive vocabulary guide
    wordnet_words = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            # Clean up word format
            word = lemma.name().replace('_', ' ')
            # Skip multi-word expressions and very short words
            if len(word.split()) == 1 and len(word) > 2:
                wordnet_words.add(word.lower())
    
    wordnet_list = sorted(list(wordnet_words))
    
    # Sample words: first half + last half (alphabetically)
    half_size = sample_size // 2
    if len(wordnet_list) < sample_size:
        print(f"Warning: Only {len(wordnet_list)} words available, using all")
        return wordnet_list
    
    sample_words = wordnet_list[:half_size] + wordnet_list[-half_size:]
    
    return sample_words


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1 = emb1.float().flatten()
    emb2 = emb2.float().flatten()
    
    if emb1.norm() == 0 or emb2.norm() == 0:
        return 0.0
    
    similarity = torch.dot(emb1, emb2) / (emb1.norm() * emb2.norm())
    return similarity.item()


def semantic_connectivity(
    word: str, 
    model: AutoModel, 
    tokenizer: AutoTokenizer,
    vocab_sample: List[str],
    threshold: float = 0.7
) -> float:
    """
    Compute semantic connectivity score for a word.
    
    Args:
        word: Word to analyze
        model: Language model
        tokenizer: Tokenizer
        vocab_sample: Sample of vocabulary tokens to compare against
        threshold: Similarity threshold for considering words connected
    
    Returns:
        Connectivity score (number of high-similarity neighbors)
    """
    try:
        # Get word embedding
        tokens = tokenizer(word, return_tensors='pt', padding=False, truncation=True)
        if tokens['input_ids'].shape[1] == 0:
            return 0.0
        
        with torch.no_grad():
            # Get embeddings from the model
            outputs = model(**tokens, output_hidden_states=True)
            # Use the last hidden state's mean as word embedding
            word_emb = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Sample vocabulary for comparison
        sample_size = min(1000, len(vocab_sample))
        sampled_vocab = random.sample(vocab_sample, sample_size)
        
        # Compute similarities
        high_similarity_count = 0
        
        for other_word in sampled_vocab:
            if other_word == word:
                continue
                
            try:
                other_tokens = tokenizer(other_word, return_tensors='pt', padding=False, truncation=True)
                if other_tokens['input_ids'].shape[1] == 0:
                    continue
                
                with torch.no_grad():
                    other_outputs = model(**other_tokens, output_hidden_states=True)
                    other_emb = other_outputs.last_hidden_state.mean(dim=1).squeeze()
                
                similarity = cosine_similarity(word_emb, other_emb)
                
                if similarity > threshold:
                    high_similarity_count += 1
                    
            except Exception:
                continue
        
        return high_similarity_count
        
    except Exception as e:
        print(f"Error processing word '{word}': {e}")
        return 0.0


def find_connectivity_outliers(
    connectivity_scores: List[Tuple[str, float]]
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
        'top_50_connected': [(word, score) for word, score in sorted_words[:50]],
        'bottom_50_connected': [(word, score) for word, score in sorted_words[-50:]],
        'all_scores': sorted_words
    }
    
    # Add random sample if we have enough words
    if len(sorted_words) > 200:
        middle_words = sorted_words[100:-100]
        random_sample = random.sample(middle_words, min(100, len(middle_words)))
        result['random_100'] = sorted(random_sample, key=lambda x: x[1], reverse=True)
    
    return result


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
        description="Compute semantic connectivity scores for vocabulary sample"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='google/gemma-2-2b',
        help='Model name or path (default: google/gemma-2-2b)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=5000,
        help='Size of vocabulary sample (default: 5000)'
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
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()
    
    # Get vocabulary sample
    print(f"Getting vocabulary sample of {args.vocab_size} words...")
    sample_words = get_vocabulary_sample(args.vocab_size)
    print(f"Sampled {len(sample_words)} words")
    
    # Check for checkpoint
    connectivity_scores = []
    processed_words = set()
    
    if args.resume and os.path.exists(args.checkpoint):
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint:
            connectivity_scores = [(w, s) for w, s in checkpoint.get('scores', [])]
            processed_words = set(w for w, _ in connectivity_scores)
            print(f"Resumed from checkpoint: {len(processed_words)} words already processed")
    
    # Process words
    remaining_words = [w for w in sample_words if w not in processed_words]
    
    print(f"Computing semantic connectivity for {len(remaining_words)} words...")
    
    # Use a subset of vocabulary for similarity comparisons
    vocab_for_comparison = random.sample(sample_words, min(2000, len(sample_words)))
    
    try:
        for i, word in enumerate(tqdm(remaining_words, desc="Processing words")):
            score = semantic_connectivity(
                word, model, tokenizer, vocab_for_comparison, args.threshold
            )
            connectivity_scores.append((word, score))
            
            # Save checkpoint every 100 words
            if (i + 1) % 100 == 0:
                save_checkpoint(args.checkpoint, {'scores': connectivity_scores})
        
        # Final save
        save_checkpoint(args.checkpoint, {'scores': connectivity_scores})
        
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        save_checkpoint(args.checkpoint, {'scores': connectivity_scores})
        sys.exit(1)
    
    # Find outliers
    print("\nIdentifying outliers...")
    outliers = find_connectivity_outliers(connectivity_scores)
    
    # Prepare results
    results = {
        'model': args.model,
        'vocab_size': len(sample_words),
        'threshold': args.threshold,
        'total_processed': len(connectivity_scores),
        'outliers': {
            'top_50': outliers['top_50_connected'],
            'bottom_50': outliers['bottom_50_connected'],
            'random_100': outliers.get('random_100', [])
        },
        'statistics': {
            'mean_connectivity': np.mean([s for _, s in connectivity_scores]),
            'std_connectivity': np.std([s for _, s in connectivity_scores]),
            'min_connectivity': min(s for _, s in connectivity_scores),
            'max_connectivity': max(s for _, s in connectivity_scores)
        }
    }
    
    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total words processed: {len(connectivity_scores)}")
    print(f"Mean connectivity: {results['statistics']['mean_connectivity']:.2f}")
    print(f"Std connectivity: {results['statistics']['std_connectivity']:.2f}")
    print(f"Range: [{results['statistics']['min_connectivity']:.0f}, {results['statistics']['max_connectivity']:.0f}]")
    
    print("\nTop 10 most connected words:")
    for word, score in outliers['top_50_connected'][:10]:
        print(f"  {word}: {score:.0f}")
    
    print("\nBottom 10 least connected words:")
    for word, score in outliers['bottom_50_connected'][-10:]:
        print(f"  {word}: {score:.0f}")
    
    # Clean up checkpoint if successful
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"\nRemoved checkpoint file: {args.checkpoint}")


if __name__ == "__main__":
    main()