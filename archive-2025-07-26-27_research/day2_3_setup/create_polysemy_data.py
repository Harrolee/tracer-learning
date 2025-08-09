#!/usr/bin/env python3
"""
Create polysemy data file for Day 2-3 analysis
Extracts polysemy scores from Day 1 implementation for connectivity correlation
"""

import sys
import os
import json
import pickle
from collections import defaultdict

# Add Day 1 directory to path
sys.path.append('../day1_setup')

try:
    from vocab_sampling import calculate_polysemy_scores
except ImportError:
    print("Error: Cannot import Day 1 vocabulary sampling module")
    print("Make sure you're running this from the tracer-learning directory")
    sys.exit(1)

def create_polysemy_data_file(words_json_file: str, output_file: str = "polysemy_scores.json"):
    """
    Create polysemy scores file for the words in our analysis
    
    Args:
        words_json_file: JSON file containing the word list
        output_file: Output JSON file for polysemy scores
    """
    
    # Load word list
    print("Loading word list...")
    with open(words_json_file, 'r') as f:
        words = json.load(f)
    
    print(f"Loaded {len(words)} words from {words_json_file}")
    
    # Calculate polysemy scores for all WordNet words
    print("Calculating polysemy scores from WordNet...")
    all_polysemy_scores = calculate_polysemy_scores()
    
    # Extract scores for our specific words
    word_polysemy_scores = {}
    missing_words = []
    
    for word in words:
        if word in all_polysemy_scores:
            word_polysemy_scores[word] = all_polysemy_scores[word]
        else:
            # Default to 1 if word not found in WordNet
            word_polysemy_scores[word] = 1
            missing_words.append(word)
    
    # Save polysemy scores
    with open(output_file, 'w') as f:
        json.dump(word_polysemy_scores, f, indent=2)
    
    print(f"✅ Polysemy scores saved to {output_file}")
    print(f"📊 Coverage: {len(word_polysemy_scores) - len(missing_words)}/{len(words)} words found in WordNet")
    
    if missing_words:
        print(f"⚠️  {len(missing_words)} words not found in WordNet (defaulted to polysemy=1)")
        if len(missing_words) <= 10:
            print(f"Missing words: {missing_words}")
    
    # Print polysemy distribution
    polysemy_values = list(word_polysemy_scores.values())
    polysemy_dist = defaultdict(int)
    
    for score in polysemy_values:
        if score == 1:
            polysemy_dist['monosemous'] += 1
        elif score <= 3:
            polysemy_dist['low_polysemy'] += 1
        elif score <= 10:
            polysemy_dist['medium_polysemy'] += 1
        else:
            polysemy_dist['high_polysemy'] += 1
    
    print(f"\n📈 Polysemy distribution in analysis set:")
    for category, count in polysemy_dist.items():
        percentage = count / len(words) * 100
        print(f"  {category.replace('_', ' ').title():15}: {count:4d} words ({percentage:5.1f}%)")
    
    return word_polysemy_scores

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create polysemy data for Day 2-3 analysis")
    parser.add_argument('--words', type=str, default='extreme_contrast_words.json',
                       help='JSON file containing word list')
    parser.add_argument('--output', type=str, default='polysemy_scores.json',
                       help='Output file for polysemy scores')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.words):
        print(f"Error: Word list file {args.words} not found")
        print("Make sure to run this from the day2_3_setup directory")
        sys.exit(1)
    
    create_polysemy_data_file(args.words, args.output)

if __name__ == "__main__":
    main() 