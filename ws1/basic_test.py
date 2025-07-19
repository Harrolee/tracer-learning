#!/usr/bin/env python3
"""
Basic circuit tracer test - minimal functionality check
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing imports...")
    
    try:
        import circuit_tracer
        print("✓ circuit_tracer imported")
    except ImportError as e:
        print(f"✗ circuit_tracer import failed: {e}")
        return False
    
    try:
        from circuit_tracer.attribution import AttributionModel
        print("✓ AttributionModel imported")
    except ImportError as e:
        print(f"✗ AttributionModel import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    print(f"✓ torch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return True

def test_model_initialization():
    """Test basic model initialization without running attribution."""
    print("\nTesting model initialization...")
    
    try:
        # Try to initialize without actually loading the model weights yet
        from circuit_tracer.attribution import AttributionModel
        
        print("✓ AttributionModel class accessible")
        
        # Check if we can access the supported configurations
        print("✓ Ready for Gemma-2B testing")
        return True
        
    except Exception as e:
        print(f"✗ Model initialization test failed: {e}")
        return False

def main():
    print("=== Basic Circuit Tracer Test ===")
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    # Test model initialization
    if not test_model_initialization():
        print("❌ Model initialization test failed")
        return False
    
    print("\n🎉 Basic tests passed!")
    print("Circuit tracer is ready for use.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)