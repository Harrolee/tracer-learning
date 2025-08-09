"""
Polysemy Sampling Strategy Comparison Tool
Research Plan: Semantic Connectivity vs Circuit Complexity

This tool compares different vocabulary sampling strategies based on polysemy.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
from vocab_sampling import calculate_polysemy_scores

def load_vocabulary_samples() -> Dict[str, List[str]]:
    """Load all available vocabulary samples from different strategies"""
    samples = {}
    
    strategies = ["extreme_contrast", "balanced", "high_polysemy"]
    
    for strategy in strategies:
        filename = f"wordnet_sample_5k_{strategy}.pkl"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                samples[strategy] = pickle.load(f)
            print(f"✅ Loaded {strategy} sample: {len(samples[strategy])} words")
        else:
            print(f"❌ {strategy} sample not found: {filepath}")
    
    return samples

def analyze_strategy_polysemy(samples: Dict[str, List[str]], 
                            polysemy_scores: Dict[str, int]) -> Dict[str, Dict]:
    """Analyze polysemy characteristics of each sampling strategy"""
    
    analysis = {}
    
    for strategy, words in samples.items():
        # Get polysemy scores for this sample
        sample_polysemy = [polysemy_scores.get(word, 1) for word in words]
        
        # Calculate statistics
        stats = {
            'total_words': len(words),
            'polysemy_mean': np.mean(sample_polysemy),
            'polysemy_std': np.std(sample_polysemy),
            'polysemy_min': min(sample_polysemy),
            'polysemy_max': max(sample_polysemy),
            'polysemy_median': np.median(sample_polysemy)
        }
        
        # Categorize by polysemy level
        categories = defaultdict(int)
        for score in sample_polysemy:
            if score == 1:
                categories['monosemous'] += 1
            elif score <= 3:
                categories['low_polysemy'] += 1
            elif score <= 10:
                categories['medium_polysemy'] += 1
            else:
                categories['high_polysemy'] += 1
        
        stats['categories'] = dict(categories)
        stats['category_percentages'] = {
            cat: count / len(words) * 100 
            for cat, count in categories.items()
        }
        
        analysis[strategy] = stats
    
    return analysis

def print_strategy_comparison(analysis: Dict[str, Dict]):
    """Print detailed comparison of sampling strategies"""
    
    print("\n" + "="*80)
    print("POLYSEMY SAMPLING STRATEGY COMPARISON")
    print("="*80)
    
    # Print overall statistics
    print(f"{'Strategy':<15} {'Mean':<6} {'Std':<6} {'Min':<4} {'Max':<4} {'Median':<6}")
    print("-" * 50)
    
    for strategy, stats in analysis.items():
        print(f"{strategy:<15} {stats['polysemy_mean']:<6.2f} {stats['polysemy_std']:<6.2f} "
              f"{stats['polysemy_min']:<4} {stats['polysemy_max']:<4} {stats['polysemy_median']:<6.1f}")
    
    # Print category distributions
    print(f"\n{'Strategy':<15} {'Monosemous':<12} {'Low':<8} {'Medium':<8} {'High':<8}")
    print("-" * 60)
    
    for strategy, stats in analysis.items():
        cat_pct = stats['category_percentages']
        print(f"{strategy:<15} {cat_pct.get('monosemous', 0):<12.1f}% "
              f"{cat_pct.get('low_polysemy', 0):<8.1f}% "
              f"{cat_pct.get('medium_polysemy', 0):<8.1f}% "
              f"{cat_pct.get('high_polysemy', 0):<8.1f}%")

def plot_polysemy_distributions(samples: Dict[str, List[str]], 
                               polysemy_scores: Dict[str, int]):
    """Create visualization comparing polysemy distributions"""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Polysemy Distribution Comparison Across Sampling Strategies', fontsize=16)
        
        strategies = list(samples.keys())
        
        for i, strategy in enumerate(strategies[:4]):  # Limit to 4 strategies
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Get polysemy scores for this sample
            sample_polysemy = [polysemy_scores.get(word, 1) for word in samples[strategy]]
            
            # Create histogram
            ax.hist(sample_polysemy, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{strategy.replace("_", " ").title()} Strategy')
            ax.set_xlabel('Polysemy Score (Number of Senses)')
            ax.set_ylabel('Number of Words')
            
            # Add statistics text
            mean_poly = np.mean(sample_polysemy)
            std_poly = np.std(sample_polysemy)
            ax.text(0.7, 0.9, f'Mean: {mean_poly:.2f}\nStd: {std_poly:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), "results", "polysemy_comparison.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Polysemy distribution plot saved to: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Plotting failed: {e}")

def recommend_strategy_for_research():
    """Provide recommendations for different research scenarios"""
    
    print("\n" + "="*80)
    print("STRATEGY RECOMMENDATIONS FOR RESEARCH SCENARIOS")
    print("="*80)
    
    recommendations = {
        "Testing Polysemy-Connectivity Hypothesis": {
            "strategy": "extreme_contrast",
            "reason": "Maximizes difference between high and low polysemy groups",
            "pros": ["Clear contrast", "Strong statistical power", "Easy interpretation"],
            "cons": ["May miss middle-range effects", "Not representative of full spectrum"]
        },
        
        "Comprehensive Polysemy Analysis": {
            "strategy": "balanced", 
            "reason": "Provides even coverage across polysemy spectrum",
            "pros": ["Representative sample", "Can detect non-linear relationships", "Good for correlation analysis"],
            "cons": ["May dilute strong effects", "Requires larger sample sizes"]
        },
        
        "Semantic Complexity Focus": {
            "strategy": "high_polysemy",
            "reason": "Concentrates on semantically rich words",
            "pros": ["Studies complex semantic networks", "High connectivity expected", "Relevant for language understanding"],
            "cons": ["Skewed sample", "May miss simple word patterns", "Limited generalizability"]
        },
        

    }
    
    for scenario, info in recommendations.items():
        print(f"\n🎯 {scenario}:")
        print(f"   Recommended Strategy: {info['strategy']}")
        print(f"   Reason: {info['reason']}")
        print(f"   Pros: {', '.join(info['pros'])}")
        print(f"   Cons: {', '.join(info['cons'])}")

def main():
    """Main comparison function"""
    print("🔍 Polysemy Sampling Strategy Comparison Tool")
    
    # Load vocabulary samples
    samples = load_vocabulary_samples()
    
    if not samples:
        print("❌ No vocabulary samples found. Run vocabulary sampling first.")
        return
    
    # Calculate polysemy scores
    print("\nCalculating polysemy scores...")
    polysemy_scores = calculate_polysemy_scores()
    
    # Analyze strategies
    print("\nAnalyzing strategy characteristics...")
    analysis = analyze_strategy_polysemy(samples, polysemy_scores)
    
    # Print comparison
    print_strategy_comparison(analysis)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_polysemy_distributions(samples, polysemy_scores)
    
    # Provide recommendations
    recommend_strategy_for_research()
    
    print("\n🎉 Strategy comparison completed!")
    
    return analysis

if __name__ == "__main__":
    analysis = main() 