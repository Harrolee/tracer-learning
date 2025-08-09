"""
Day 1: Semantic Connectivity Measurement Implementation
Research Plan: Semantic Connectivity vs Circuit Complexity
"""

import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
import os

class SemanticConnectivityAnalyzer:
    """Analyze semantic connectivity of words using Gemma2 embeddings"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_matrix = None
        self.vocab_tokens = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Setup embedding matrix and vocabulary tokens"""
        print("Setting up embeddings for semantic connectivity analysis...")
        
        # Get embedding matrix
        self.embedding_matrix = self.model.get_input_embeddings().weight.detach()
        if self.device == "cpu":
            self.embedding_matrix = self.embedding_matrix.cpu()
        
        # Get vocabulary tokens (exclude special tokens)
        vocab = self.tokenizer.get_vocab()
        # Filter out special tokens and non-word tokens
        self.vocab_tokens = [
            token for token, token_id in vocab.items()
            if token.isalpha() and len(token) > 1 and not token.startswith('<')
        ]
        
        print(f"Embedding matrix shape: {self.embedding_matrix.shape}")
        print(f"Vocabulary tokens for sampling: {len(self.vocab_tokens)}")
    
    def get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding for a single word"""
        try:
            # Tokenize the word
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            
            if len(token_ids) == 0:
                return None
            
            # For multi-token words, take the mean embedding
            if len(token_ids) == 1:
                embedding = self.embedding_matrix[token_ids[0]]
            else:
                embeddings = self.embedding_matrix[token_ids]
                embedding = torch.mean(embeddings, dim=0)
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding for '{word}': {e}")
            return None
    
    def semantic_connectivity(self, word: str, vocab_sample_size: int = 1000, 
                            similarity_threshold: float = 0.7) -> int:
        """
        Calculate semantic connectivity for a word
        
        Args:
            word: Target word to analyze
            vocab_sample_size: Number of vocabulary words to sample for comparison
            similarity_threshold: Minimum cosine similarity to count as connected
        
        Returns:
            Number of high-similarity neighbors
        """
        # Get word embedding
        word_embedding = self.get_word_embedding(word)
        if word_embedding is None:
            return 0
        
        # Sample vocabulary for comparison
        vocab_sample = random.sample(self.vocab_tokens, 
                                   min(vocab_sample_size, len(self.vocab_tokens)))
        
        # Calculate similarities
        high_similarity_count = 0
        
        for vocab_word in vocab_sample:
            vocab_embedding = self.get_word_embedding(vocab_word)
            if vocab_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(
                word_embedding.unsqueeze(0), 
                vocab_embedding.unsqueeze(0)
            ).item()
            
            if similarity > similarity_threshold:
                high_similarity_count += 1
        
        return high_similarity_count
    
    def analyze_word_list(self, words: List[str], vocab_sample_size: int = 1000,
                         similarity_threshold: float = 0.7) -> Dict[str, int]:
        """
        Analyze semantic connectivity for a list of words
        
        Returns:
            Dictionary mapping words to their connectivity scores
        """
        print(f"Analyzing semantic connectivity for {len(words)} words...")
        print(f"Using vocab sample size: {vocab_sample_size}, threshold: {similarity_threshold}")
        
        connectivity_scores = {}
        
        for word in tqdm(words, desc="Computing connectivity"):
            score = self.semantic_connectivity(
                word, vocab_sample_size, similarity_threshold
            )
            connectivity_scores[word] = score
        
        return connectivity_scores
    
    def find_connectivity_outliers(self, words: List[str], 
                                 vocab_sample_size: int = 1000) -> Dict[str, any]:
        """
        Find connectivity outliers in a word list
        Implementation of the method from research plan
        """
        print("Finding connectivity outliers...")
        
        # Get connectivity scores for all words
        connectivity_scores = self.analyze_word_list(words, vocab_sample_size)
        
        # Convert to list of tuples and sort
        scored_words = [(word, score) for word, score in connectivity_scores.items()]
        sorted_words = sorted(scored_words, key=lambda x: x[1], reverse=True)
        
        # Extract outliers and random sample
        result = {
            'top_50_connected': [word for word, score in sorted_words[:50]],
            'bottom_50_connected': [word for word, score in sorted_words[-50:]],
            'all_scores': sorted_words
        }
        
        # Random sample from middle range (avoiding top/bottom 100)
        if len(sorted_words) > 200:
            middle_range = sorted_words[100:-100]
            if len(middle_range) >= 100:
                random_sample = random.sample(middle_range, 100)
                result['random_100'] = [word for word, score in random_sample]
            else:
                result['random_100'] = [word for word, score in middle_range]
        else:
            result['random_100'] = []
        
        return result
    
    def save_connectivity_analysis(self, results: Dict, filename: str = "connectivity_analysis.pkl"):
        """Save connectivity analysis results"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Connectivity analysis saved to {filepath}")
    
    def print_connectivity_summary(self, results: Dict):
        """Print summary of connectivity analysis"""
        print("\n=== Semantic Connectivity Analysis Summary ===")
        
        all_scores = results['all_scores']
        scores_only = [score for word, score in all_scores]
        
        print(f"Total words analyzed: {len(all_scores)}")
        print(f"Connectivity range: {min(scores_only)} - {max(scores_only)}")
        print(f"Average connectivity: {np.mean(scores_only):.2f}")
        print(f"Median connectivity: {np.median(scores_only):.2f}")
        
        print(f"\nTop 10 most connected words:")
        for i, (word, score) in enumerate(all_scores[:10]):
            print(f"  {i+1:2d}. {word}: {score}")
        
        print(f"\nBottom 10 least connected words:")
        for i, (word, score) in enumerate(all_scores[-10:]):
            print(f"  {i+1:2d}. {word}: {score}")

def main():
    """Main function for testing semantic connectivity analysis"""
    print("=== Day 1: Semantic Connectivity Testing ===")
    
    # This is a placeholder - in practice, this would be called with loaded model
    print("Note: This module requires a loaded Gemma2 model and tokenizer")
    print("Use this in conjunction with gemma_setup.py and vocab_sampling.py")
    
    # Example usage (commented out since we need actual model):
    """
    from gemma_setup import Gemma2Setup
    from vocab_sampling import main as get_vocab_sample
    
    # Load model
    gemma_setup = Gemma2Setup()
    model, tokenizer = gemma_setup.load_model()
    
    # Get vocabulary sample
    vocab_sample = get_vocab_sample()
    
    # Analyze connectivity
    analyzer = SemanticConnectivityAnalyzer(model, tokenizer)
    results = analyzer.find_connectivity_outliers(vocab_sample[:100])  # Test with first 100
    analyzer.print_connectivity_summary(results)
    """

if __name__ == "__main__":
    main() 