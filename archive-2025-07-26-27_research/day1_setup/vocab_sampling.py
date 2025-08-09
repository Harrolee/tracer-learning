"""
Day 1: WordNet Vocabulary Sampling Implementation
Research Plan: Semantic Connectivity vs Circuit Complexity
"""

import nltk
import random
import pickle
import os
from typing import List, Dict, Tuple
from nltk.corpus import wordnet
from tqdm import tqdm
from collections import defaultdict

# Download required NLTK data
def setup_nltk():
    """Download required NLTK datasets"""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading WordNet corpus...")
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading OMW corpus...")
        nltk.download('omw-1.4')

def calculate_polysemy_scores() -> Dict[str, int]:
    """
    Calculate polysemy scores for all WordNet words
    Polysemy = number of different synsets (meanings) a word appears in
    """
    print("Calculating polysemy scores...")
    
    polysemy_scores = defaultdict(int)
    
    for synset in tqdm(wordnet.all_synsets(), desc="Processing synsets"):
        for lemma in synset.lemmas():
            # Clean the lemma name
            word = lemma.name().replace('_', ' ').lower()
            # Filter out words with special characters or numbers
            if word.isalpha() and len(word) > 1:
                polysemy_scores[word] += 1
    
    print(f"Calculated polysemy scores for {len(polysemy_scores)} words")
    return dict(polysemy_scores)

def analyze_polysemy_distribution(polysemy_scores: Dict[str, int]):
    """Analyze the distribution of polysemy scores"""
    scores_list = list(polysemy_scores.values())
    
    print("\n=== Polysemy Distribution Analysis ===")
    print(f"Total words: {len(polysemy_scores)}")
    print(f"Polysemy range: {min(scores_list)} - {max(scores_list)}")
    print(f"Average polysemy: {sum(scores_list)/len(scores_list):.2f}")
    
    # Distribution by polysemy level
    polysemy_dist = defaultdict(int)
    for score in scores_list:
        if score == 1:
            polysemy_dist['monosemous'] += 1
        elif score <= 3:
            polysemy_dist['low_polysemy'] += 1
        elif score <= 10:
            polysemy_dist['medium_polysemy'] += 1
        else:
            polysemy_dist['high_polysemy'] += 1
    
    print("\nPolysemy categories:")
    for category, count in polysemy_dist.items():
        percentage = count / len(polysemy_scores) * 100
        print(f"  {category}: {count} words ({percentage:.1f}%)")
    
    # Top polysemous words
    sorted_words = sorted(polysemy_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most polysemous words:")
    for i, (word, score) in enumerate(sorted_words[:10]):
        print(f"  {i+1:2d}. {word}: {score} senses")
    
    print(f"\n10 monosemous words (examples):")
    monosemous = [word for word, score in sorted_words if score == 1]
    for i, word in enumerate(monosemous[:10]):
        print(f"  {i+1:2d}. {word}: 1 sense")

def sample_by_polysemy(polysemy_scores: Dict[str, int], 
                      strategy: str = "extreme_contrast",
                      total_words: int = 5000) -> List[str]:
    """
    Sample words based on polysemy scores using different strategies
    
    Args:
        polysemy_scores: Dictionary mapping words to their polysemy scores
        strategy: Sampling strategy ('extreme_contrast', 'balanced', 'high_polysemy')
        total_words: Total number of words to sample
    
    Returns:
        List of sampled words
    """
    
    sorted_words = sorted(polysemy_scores.items(), key=lambda x: x[1], reverse=True)
    
    if strategy == "extreme_contrast":
        # Sample high-polysemy vs monosemous words for maximum contrast
        print(f"Sampling {total_words} words using extreme contrast strategy...")
        
        # Get high polysemy words (top 25%)
        high_poly_cutoff = int(len(sorted_words) * 0.25)
        high_poly_words = [word for word, score in sorted_words[:high_poly_cutoff]]
        
        # Get monosemous words (score = 1)
        monosemous_words = [word for word, score in sorted_words if score == 1]
        
        # Sample half from each group
        half_words = total_words // 2
        
        sampled_high = random.sample(high_poly_words, min(half_words, len(high_poly_words)))
        sampled_mono = random.sample(monosemous_words, min(half_words, len(monosemous_words)))
        
        sample_words = sampled_high + sampled_mono
        
        print(f"  - High polysemy: {len(sampled_high)} words")
        print(f"  - Monosemous: {len(sampled_mono)} words")
        
    elif strategy == "balanced":
        # Sample across polysemy spectrum
        print(f"Sampling {total_words} words using balanced polysemy strategy...")
        
        # Divide into quartiles
        quartile_size = len(sorted_words) // 4
        words_per_quartile = total_words // 4
        
        sample_words = []
        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else len(sorted_words)
            quartile_words = [word for word, score in sorted_words[start_idx:end_idx]]
            
            sampled = random.sample(quartile_words, 
                                  min(words_per_quartile, len(quartile_words)))
            sample_words.extend(sampled)
        
    elif strategy == "high_polysemy":
        # Focus on highly polysemous words only
        print(f"Sampling {total_words} words using high polysemy strategy...")
        
        # Take top 50% most polysemous words
        high_poly_cutoff = len(sorted_words) // 2
        high_poly_words = [word for word, score in sorted_words[:high_poly_cutoff]]
        
        sample_words = random.sample(high_poly_words, 
                                   min(total_words, len(high_poly_words)))
        
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Ensure we have the right number of words
    if len(sample_words) > total_words:
        sample_words = random.sample(sample_words, total_words)
    
    print(f"Final sample: {len(sample_words)} words")
    return sample_words

def get_vocabulary_sample(strategy: str = "extreme_contrast", 
                         total_words: int = 5000) -> List[str]:
    """
    Use WordNet synsets as comprehensive vocabulary guide
    Sample words based on polysemy scores
    """
    print("Extracting vocabulary from WordNet...")
    
    # Calculate polysemy scores for all words
    polysemy_scores = calculate_polysemy_scores()
    
    # Analyze polysemy distribution
    analyze_polysemy_distribution(polysemy_scores)
    
    # Sample words based on strategy
    sample_words = sample_by_polysemy(polysemy_scores, strategy, total_words)
    
    # Add polysemy information to sample analysis
    sample_polysemy = {word: polysemy_scores[word] for word in sample_words}
    sample_scores = list(sample_polysemy.values())
    
    print(f"\n=== Sample Polysemy Statistics ===")
    print(f"Sample range: '{min(sample_words)}' to '{max(sample_words)}'")
    print(f"Polysemy range in sample: {min(sample_scores)} - {max(sample_scores)}")
    print(f"Average polysemy in sample: {sum(sample_scores)/len(sample_scores):.2f}")
    
    return sample_words

def save_vocabulary_sample(sample_words: List[str], filename: str = "wordnet_sample_5k.pkl"):
    """Save vocabulary sample to file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'wb') as f:
        pickle.dump(sample_words, f)
    print(f"Vocabulary sample saved to {filepath}")

def load_vocabulary_sample(filename: str = "wordnet_sample_5k.pkl") -> List[str]:
    """Load vocabulary sample from file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            sample_words = pickle.load(f)
        print(f"Loaded {len(sample_words)} words from {filepath}")
        return sample_words
    else:
        print(f"File {filepath} not found. Generating new sample...")
        return None

def analyze_vocabulary_sample(sample_words: List[str]):
    """Analyze the vocabulary sample statistics"""
    print("\n=== Vocabulary Sample Analysis ===")
    print(f"Total words: {len(sample_words)}")
    print(f"First 10 words: {sample_words[:10]}")
    print(f"Last 10 words: {sample_words[-10:]}")
    
    # Length distribution
    lengths = [len(word) for word in sample_words]
    print(f"Average word length: {sum(lengths)/len(lengths):.2f}")
    print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")
    
    # First letter distribution
    first_letters = {}
    for word in sample_words:
        first_letter = word[0].upper()
        first_letters[first_letter] = first_letters.get(first_letter, 0) + 1
    
    print("\nFirst letter distribution (top 10):")
    sorted_letters = sorted(first_letters.items(), key=lambda x: x[1], reverse=True)
    for letter, count in sorted_letters[:10]:
        print(f"  {letter}: {count} words")

def main(strategy: str = "extreme_contrast"):
    """Main function to execute Day 1 vocabulary sampling"""
    print("=== Day 1: WordNet Vocabulary Sampling with Polysemy ===")
    
    # Setup NLTK
    setup_nltk()
    
    # Generate new vocabulary sample with polysemy-based sampling
    sample_words = get_vocabulary_sample(strategy=strategy)
    
    # Save the sample for future use
    filename = f"wordnet_sample_5k_{strategy}.pkl"
    save_vocabulary_sample(sample_words, filename)
    
    # Analyze the sample
    analyze_vocabulary_sample(sample_words)
    
    return sample_words

if __name__ == "__main__":
    import sys
    
    # Allow strategy to be specified as command line argument
    strategy = sys.argv[1] if len(sys.argv) > 1 else "extreme_contrast"
    
    print(f"Using sampling strategy: {strategy}")
    print("Available strategies: extreme_contrast, balanced, high_polysemy")
    
    vocabulary_sample = main(strategy) 