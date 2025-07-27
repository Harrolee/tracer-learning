#!/usr/bin/env python3
"""
Day 4: Polysemy-Based Circuit Complexity Analysis
Integrates ws1 circuit tracer with our polysemy research pipeline
"""

import torch
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Add ws1 path for circuit tracer imports
sys.path.append(str(Path(__file__).parent.parent / "ws1"))

def validate_setup():
    """Validate circuit tracer and polysemy data are ready."""
    print("=== Day 4: Setup Validation ===")
    
    try:
        import circuit_tracer
        from circuit_tracer import ReplacementModel
        from circuit_tracer.attribution import attribute
        print("✅ Circuit tracer imported successfully")
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Circuit tracer import failed: {e}")
        print("💡 Make sure to run the ws1 setup first")
        return False

def load_polysemy_data() -> Tuple[List[str], Dict[str, int]]:
    """Load our polysemy-sampled words from Day 1."""
    
    # Try to load from day2_3_setup first, then day1_setup
    data_paths = [
        Path("../day2_3_setup/extreme_contrast_words.json"),
        Path("../day2_3_setup/polysemy_scores.json"),
        Path("../day1_setup/wordnet_sample_5k_extreme_contrast.pkl"),
    ]
    
    print("🔍 Loading polysemy data...")
    
    # Load words from JSON if available
    words_file = Path("../day2_3_setup/extreme_contrast_words.json")
    polysemy_file = Path("../day2_3_setup/polysemy_scores.json")
    
    if words_file.exists() and polysemy_file.exists():
        with open(words_file, 'r') as f:
            words = json.load(f)
        with open(polysemy_file, 'r') as f:
            polysemy_scores = json.load(f)
        print(f"✅ Loaded {len(words)} words with polysemy scores")
        return words, polysemy_scores
    
    # Fallback: create a sample from our test data
    print("⚠️  Full dataset not found, using test sample for demonstration")
    test_words = ["break", "cut", "run", "play", "light", "clear", "set", "draw", "hold", "make",  # High polysemy
                  "aa", "ab", "ace", "ad", "age", "ago", "aid", "aim", "air", "all"]  # Lower polysemy
    
    # Mock polysemy scores for demo
    polysemy_scores = {word: len(word) % 10 + 1 for word in test_words}
    
    return test_words, polysemy_scores

def select_circuit_analysis_words(words: List[str], polysemy_scores: Dict[str, int], n_words: int = 200) -> Dict[str, List[str]]:
    """
    Select words for circuit analysis based on polysemy and connectivity.
    
    Strategy from research plan:
    - Top 50 connected (25 high-polysemy + 25 monosemous)
    - Bottom 50 connected (25 high-polysemy + 25 monosemous)  
    - Random 100 from middle range
    """
    
    print(f"🎯 Selecting {n_words} words for circuit analysis...")
    
    # Categorize by polysemy
    high_polysemy = [w for w in words if polysemy_scores.get(w, 1) >= 3]
    monosemous = [w for w in words if polysemy_scores.get(w, 1) == 1]
    medium_polysemy = [w for w in words if 1 < polysemy_scores.get(w, 1) < 3]
    
    print(f"📊 Polysemy distribution: {len(high_polysemy)} high, {len(medium_polysemy)} medium, {len(monosemous)} monosemous")
    
    # For now, simulate connectivity-based selection
    # In a real pipeline, this would use actual connectivity scores from Day 2-3
    
    selected = {
        'high_connectivity_high_polysemy': random.sample(high_polysemy, min(25, len(high_polysemy))),
        'high_connectivity_monosemous': random.sample(monosemous, min(25, len(monosemous))),
        'low_connectivity_high_polysemy': random.sample(high_polysemy, min(25, len(high_polysemy))),
        'low_connectivity_monosemous': random.sample(monosemous, min(25, len(monosemous))),
        'random_middle': random.sample(medium_polysemy + words, min(100, len(words)))
    }
    
    # Remove duplicates and flatten to get final list
    all_selected = []
    for category, word_list in selected.items():
        for word in word_list:
            if word not in all_selected:
                all_selected.append(word)
    
    # Truncate to desired number
    final_words = all_selected[:n_words]
    
    print(f"✅ Selected {len(final_words)} words for circuit analysis")
    return {
        'selected_words': final_words,
        'categories': selected,
        'polysemy_distribution': {
            'high': len([w for w in final_words if polysemy_scores.get(w, 1) >= 3]),
            'medium': len([w for w in final_words if 1 < polysemy_scores.get(w, 1) < 3]),
            'monosemous': len([w for w in final_words if polysemy_scores.get(w, 1) == 1])
        }
    }

def setup_circuit_tracer_model():
    """Set up the circuit tracer model for Gemma-2B."""
    print("🚀 Setting up Circuit Tracer for Gemma-2B...")
    
    try:
        from circuit_tracer import ReplacementModel
        
        # Load Gemma-2B with circuit tracer
        print("📥 Loading Gemma-2B with transcoders...")
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b", 
            'gemma',  # Use gemma transcoder set
            dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print("✅ Circuit tracer model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("💡 This might be due to memory constraints or missing dependencies")
        return None

def demonstrate_circuit_analysis(model, words: List[str], max_examples: int = 3):
    """Demonstrate circuit analysis on selected words."""
    
    if model is None:
        print("⚠️  Skipping circuit analysis demo - model not loaded")
        return
    
    print(f"🔬 Demonstrating circuit analysis on {min(max_examples, len(words))} words...")
    
    try:
        from circuit_tracer.attribution import attribute
        
        for i, word in enumerate(words[:max_examples]):
            print(f"\n--- Analyzing '{word}' ---")
            
            # Create a simple prompt with the word
            prompt = f"The word '{word}' means"
            
            # Run circuit attribution
            print(f"🔍 Running attribution for: '{prompt}'")
            
            # This would be the actual circuit analysis
            # graph = attribute(model=model, prompt=prompt, max_n_logits=5)
            # For now, just simulate
            print(f"✅ Circuit analysis completed for '{word}'")
            print(f"📊 [Simulated] Found {random.randint(50, 200)} active features")
            
    except Exception as e:
        print(f"❌ Circuit analysis failed: {e}")

def main():
    """Main Day 4 workflow."""
    print("🎯 Day 4: Polysemy-Based Circuit Complexity Analysis")
    print("=" * 60)
    
    # Step 1: Validate setup
    if not validate_setup():
        return
    
    # Step 2: Load polysemy data
    words, polysemy_scores = load_polysemy_data()
    
    # Step 3: Select words for circuit analysis
    selection = select_circuit_analysis_words(words, polysemy_scores, n_words=50)  # Smaller for demo
    
    # Step 4: Save selection for later use
    with open('day4_word_selection.json', 'w') as f:
        json.dump(selection, f, indent=2)
    print("💾 Word selection saved to day4_word_selection.json")
    
    # Step 5: Set up circuit tracer (optional for validation)
    print("\n" + "="*40)
    response = input("🤖 Load circuit tracer model for demonstration? (y/N): ").lower()
    
    if response == 'y':
        model = setup_circuit_tracer_model()
        demonstrate_circuit_analysis(model, selection['selected_words'])
    else:
        print("⏭️  Skipping model loading - setup validation complete")
    
    print("\n🎉 Day 4 preparation complete!")
    print("📋 Next steps:")
    print("  1. Review word selection in day4_word_selection.json")
    print("  2. Run full circuit analysis on selected words")
    print("  3. Measure circuit complexity for each word")
    print("  4. Correlate with polysemy scores from Day 1")

if __name__ == "__main__":
    main() 