#!/usr/bin/env python3
"""
Test script to verify overnight job setup is working correctly
"""

import sys
import torch
from datetime import datetime
import traceback

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic Python imports...")
    try:
        import json, numpy, logging
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_vocab_sampling():
    """Test vocabulary sampling functionality."""
    print("🧪 Testing vocabulary sampling...")
    try:
        sys.path.append('../day1_setup')
        from vocab_sampling import sample_by_polysemy, calculate_polysemy_scores
        
        # Test small sample
        print("   • Calculating polysemy scores...")
        polysemy_scores = calculate_polysemy_scores()
        print(f"   • Got {len(polysemy_scores)} words with polysemy scores")
        test_words = sample_by_polysemy(polysemy_scores, strategy='extreme_contrast', total_words=3)
        
        print(f"✅ Vocabulary sampling successful - sampled {len(test_words)} words")
        print(f"   Sample: {test_words}")
        return True
    except Exception as e:
        print(f"❌ Vocabulary sampling failed: {e}")
        print(traceback.format_exc())
        return False

def test_semantic_connectivity():
    """Test semantic connectivity functionality."""
    print("🧪 Testing semantic connectivity...")
    try:
        sys.path.append('../day2_3_setup')
        from semantic_connectivity_cli import get_word_embedding, semantic_connectivity
        from transformers import AutoTokenizer
        
        print("   • Semantic connectivity imports successful")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        print("   • Tokenizer loaded successfully")
        
        print("✅ Semantic connectivity setup successful")
        return True
    except Exception as e:
        print(f"❌ Semantic connectivity failed: {e}")
        print("💡 This might fail without GPU/full model - that's OK for testing")
        return False

def test_circuit_tracer():
    """Test circuit tracer functionality."""
    print("🧪 Testing circuit tracer...")
    try:
        sys.path.append('../circuit-tracer')
        from circuit_tracer import ReplacementModel
        from circuit_tracer.attribution import attribute
        
        print("✅ Circuit tracer imports successful")
        return True
    except Exception as e:
        print(f"❌ Circuit tracer failed: {e}")
        print(traceback.format_exc())
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("🧪 Testing GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - jobs will run on CPU (much slower)")
            return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Overnight Jobs Setup Test")
    print("=" * 40)
    print(f"⏰ Test time: {datetime.now()}")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Vocabulary Sampling", test_vocab_sampling),
        ("Semantic Connectivity", test_semantic_connectivity),
        ("Circuit Tracer", test_circuit_tracer),
        ("GPU Availability", test_gpu_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
        print()
    
    print("=" * 40)
    print("📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print()
    print(f"📈 Results: {passed}/{len(results)} tests passed")
    
    if passed >= 4:  # Allow semantic connectivity to fail in testing
        print("🎉 Setup looks good! Ready for overnight jobs")
        print("💡 Start with: ./run_overnight_jobs.sh semantic")
    elif passed >= 2:
        print("⚠️  Some issues detected but basic functionality works")
        print("💡 You can try running jobs but expect possible errors")
    else:
        print("❌ Major issues detected - fix setup before running jobs")
    
    return passed >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 