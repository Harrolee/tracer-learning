"""
Day 1 Main Execution Script
Research Plan: Semantic Connectivity vs Circuit Complexity

This script orchestrates all Day 1 tasks:
1. Setup Gemma2 2B model
2. Implement WordNet vocabulary sampling (5,000 words) with polysemy-based strategies
3. Initial connectivity analysis setup
"""

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def setup_day1_environment():
    """Setup the environment for Day 1 execution"""
    print("=== Day 1 Environment Setup ===")
    print(f"Execution time: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")
    
    return output_dir

def execute_vocabulary_sampling(output_dir: str, strategy: str = "extreme_contrast"):
    """Execute vocabulary sampling task with specified strategy"""
    print("\n" + "="*50)
    print("TASK 1: WordNet Vocabulary Sampling with Polysemy")
    print("="*50)
    print(f"Using sampling strategy: {strategy}")
    
    try:
        from vocab_sampling import main as vocab_main
        vocabulary_sample = vocab_main(strategy)
        
        # Save vocabulary sample to results directory
        import pickle
        vocab_file = os.path.join(output_dir, f"day1_vocabulary_5k_{strategy}.pkl")
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocabulary_sample, f)
        
        print(f"✅ Vocabulary sampling completed successfully!")
        print(f"   - Generated {len(vocabulary_sample)} words")
        print(f"   - Strategy: {strategy}")
        print(f"   - Saved to: {vocab_file}")
        
        return vocabulary_sample
        
    except Exception as e:
        print(f"❌ Vocabulary sampling failed: {e}")
        return None

def execute_model_setup(output_dir: str):
    """Execute Gemma2 model setup"""
    print("\n" + "="*50)
    print("TASK 2: Gemma2 2B Model Setup")
    print("="*50)
    
    try:
        from gemma_setup import Gemma2Setup, main as gemma_main
        gemma_setup = gemma_main()
        
        if gemma_setup is not None:
            print("✅ Gemma2 model setup completed successfully!")
            print("   - Model loaded and tested")
            print("   - Ready for embedding analysis")
            return gemma_setup
        else:
            print("❌ Gemma2 model setup failed")
            return None
            
    except Exception as e:
        print(f"❌ Gemma2 model setup failed: {e}")
        return None

def execute_initial_connectivity_test(gemma_setup, vocabulary_sample, output_dir: str, strategy: str):
    """Execute initial connectivity analysis test"""
    print("\n" + "="*50)
    print("TASK 3: Initial Connectivity Analysis Test")
    print("="*50)
    
    if gemma_setup is None or vocabulary_sample is None:
        print("❌ Cannot run connectivity test - missing model or vocabulary")
        return None
    
    try:
        # Test with a small sample (100 words)
        test_sample = vocabulary_sample[:100]
        
        from semantic_connectivity import SemanticConnectivityAnalyzer
        analyzer = SemanticConnectivityAnalyzer(
            gemma_setup.model, 
            gemma_setup.tokenizer, 
            gemma_setup.device
        )
        
        # Run connectivity analysis on test sample
        print(f"Running connectivity test on {len(test_sample)} words...")
        connectivity_scores = analyzer.analyze_word_list(
            test_sample, 
            vocab_sample_size=500,  # Smaller sample for testing
            similarity_threshold=0.6  # Lower threshold for testing
        )
        
        # Save test results
        import pickle
        test_file = os.path.join(output_dir, f"day1_connectivity_test_{strategy}.pkl")
        with open(test_file, 'wb') as f:
            pickle.dump(connectivity_scores, f)
        
        # Print summary
        scores = list(connectivity_scores.values())
        print(f"✅ Initial connectivity test completed!")
        print(f"   - Analyzed {len(connectivity_scores)} words")
        print(f"   - Connectivity range: {min(scores)} - {max(scores)}")
        print(f"   - Average connectivity: {sum(scores)/len(scores):.2f}")
        print(f"   - Results saved to: {test_file}")
        
        return connectivity_scores
        
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")
        return None

def analyze_polysemy_vs_connectivity(connectivity_scores: dict, vocabulary_sample: list, strategy: str):
    """Analyze relationship between polysemy and connectivity for initial insights"""
    print("\n" + "="*50)
    print("POLYSEMY vs CONNECTIVITY ANALYSIS")
    print("="*50)
    
    try:
        from vocab_sampling import calculate_polysemy_scores
        from collections import defaultdict
        
        # Get polysemy scores
        print("Calculating polysemy scores for analysis...")
        all_polysemy_scores = calculate_polysemy_scores()
        
        # Get polysemy scores for tested words
        test_polysemy = {word: all_polysemy_scores.get(word, 1) 
                        for word in connectivity_scores.keys()}
        
        # Group by polysemy level
        polysemy_groups = defaultdict(list)
        for word in connectivity_scores.keys():
            polysemy = test_polysemy[word]
            connectivity = connectivity_scores[word]
            
            if polysemy == 1:
                polysemy_groups['monosemous'].append(connectivity)
            elif polysemy <= 3:
                polysemy_groups['low_polysemy'].append(connectivity)
            elif polysemy <= 10:
                polysemy_groups['medium_polysemy'].append(connectivity)
            else:
                polysemy_groups['high_polysemy'].append(connectivity)
        
        # Analyze patterns
        print(f"Strategy: {strategy}")
        print("Connectivity by polysemy level:")
        
        for level, connectivities in polysemy_groups.items():
            if connectivities:
                avg_conn = sum(connectivities) / len(connectivities)
                print(f"  {level}: {len(connectivities)} words, avg connectivity: {avg_conn:.2f}")
        
        return polysemy_groups
        
    except Exception as e:
        print(f"❌ Polysemy analysis failed: {e}")
        return None

def print_day1_summary(vocabulary_sample, gemma_setup, connectivity_test, strategy):
    """Print Day 1 execution summary"""
    print("\n" + "="*60)
    print("DAY 1 EXECUTION SUMMARY")
    print("="*60)
    
    # Task completion status
    tasks_completed = 0
    total_tasks = 3
    
    print("Task Completion Status:")
    if vocabulary_sample is not None:
        print(f"  ✅ WordNet Vocabulary Sampling (5,000 words, {strategy} strategy)")
        tasks_completed += 1
    else:
        print("  ❌ WordNet Vocabulary Sampling")
    
    if gemma_setup is not None:
        print("  ✅ Gemma2 2B Model Setup")
        tasks_completed += 1
    else:
        print("  ❌ Gemma2 2B Model Setup")
    
    if connectivity_test is not None:
        print("  ✅ Initial Connectivity Analysis Test")
        tasks_completed += 1
    else:
        print("  ❌ Initial Connectivity Analysis Test")
    
    print(f"\nOverall Progress: {tasks_completed}/{total_tasks} tasks completed")
    
    # Strategy-specific insights
    print(f"\nSampling Strategy Insights:")
    if strategy == "extreme_contrast":
        print("  • Maximizes contrast between high-polysemy and monosemous words")
        print("  • Best for testing if polysemy predicts connectivity differences")
    elif strategy == "balanced":
        print("  • Balanced representation across polysemy spectrum")
        print("  • Good for understanding overall polysemy-connectivity relationship")
    elif strategy == "high_polysemy":
        print("  • Focuses on semantically rich, highly polysemous words")
        print("  • Best for studying complex semantic relationships")

    
    # Next steps
    print("\nNext Steps for Day 2-3:")
    if tasks_completed == total_tasks:
        print("  • Run full semantic connectivity analysis on all 5,000 words")
        print("  • Compare connectivity patterns across polysemy levels")
        print("  • Identify optimal word sets for circuit complexity analysis")
        print("  • Consider comparing multiple sampling strategies")
    else:
        print("  • Complete remaining Day 1 tasks first")
        print("  • Troubleshoot any setup issues")
    
    print("="*60)

def compare_strategies():
    """Compare different sampling strategies"""
    print("\n" + "="*60)
    print("POLYSEMY SAMPLING STRATEGY COMPARISON")
    print("="*60)
    
    strategies = {
        "extreme_contrast": "High-polysemy vs Monosemous (maximum contrast)",
        "balanced": "Balanced across polysemy spectrum (quartiles)",
        "high_polysemy": "Focus on highly polysemous words only",

    }
    
    print("Available sampling strategies:")
    for strategy, description in strategies.items():
        print(f"  • {strategy}: {description}")
    
    print("\nRecommendations:")
    print("  🎯 For research hypothesis testing: extreme_contrast")
    print("  📊 For comprehensive analysis: balanced") 
    print("  🔬 For semantic complexity focus: high_polysemy")


def main():
    """Main execution function for Day 1"""
    parser = argparse.ArgumentParser(description="Day 1: Setup and Vocabulary Sampling with Polysemy")
    parser.add_argument("--strategy", default="extreme_contrast",
                       choices=["extreme_contrast", "balanced", "high_polysemy"],
                       help="Vocabulary sampling strategy based on polysemy")
    parser.add_argument("--skip-model", action="store_true", 
                       help="Skip model loading (for testing)")
    parser.add_argument("--test-only", action="store_true",
                       help="Run in test mode with smaller samples")
    parser.add_argument("--compare-strategies", action="store_true",
                       help="Show comparison of sampling strategies")
    args = parser.parse_args()
    
    if args.compare_strategies:
        compare_strategies()
        return None, None
    
    print("🚀 Starting Day 1 Execution")
    print("Research Plan: Semantic Connectivity vs Circuit Complexity")
    print(f"Polysemy Sampling Strategy: {args.strategy}")
    
    # Setup environment
    output_dir = setup_day1_environment()
    
    # Execute tasks
    vocabulary_sample = execute_vocabulary_sampling(output_dir, args.strategy)
    
    if not args.skip_model:
        gemma_setup = execute_model_setup(output_dir)
        
        if not args.test_only:
            connectivity_test = execute_initial_connectivity_test(
                gemma_setup, vocabulary_sample, output_dir, args.strategy
            )
            
            # Analyze polysemy vs connectivity relationship
            if connectivity_test is not None:
                analyze_polysemy_vs_connectivity(
                    connectivity_test, vocabulary_sample, args.strategy
                )
        else:
            print("Skipping connectivity test (test mode)")
            connectivity_test = "skipped"
    else:
        print("Skipping model setup (skip-model flag)")
        gemma_setup = None
        connectivity_test = None
    
    # Print summary
    print_day1_summary(vocabulary_sample, gemma_setup, connectivity_test, args.strategy)
    
    print(f"\n🎉 Day 1 execution completed with {args.strategy} strategy!")
    return vocabulary_sample, gemma_setup

if __name__ == "__main__":
    vocabulary_sample, gemma_setup = main() 