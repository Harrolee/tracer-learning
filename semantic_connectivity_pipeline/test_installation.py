#!/usr/bin/env python3
"""Test installation of all required packages"""

import sys
print("Testing pipeline dependencies...")
print(f"Python version: {sys.version}")

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

# Test Transformers
try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

# Test NLTK
try:
    import nltk
    from nltk.corpus import wordnet
    print(f"✓ NLTK installed")
    # Test WordNet access
    test_synsets = list(wordnet.synsets('test'))
    print(f"  WordNet accessible: {len(test_synsets)} synsets for 'test'")
except ImportError as e:
    print(f"✗ NLTK import failed: {e}")
except LookupError:
    print("✗ WordNet data not downloaded")

# Test circuit-tracer
try:
    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution import attribute
    print("✓ Circuit-tracer installed")
except ImportError as e:
    print(f"✗ Circuit-tracer import failed: {e}")
    print("  Note: Circuit-tracer is required for feature extraction")

# Test other dependencies
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    print("✓ Other dependencies installed (numpy, pandas, tqdm)")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")

print("\n--- Installation Summary ---")
print("If all checks show ✓, you're ready to run the pipeline!")
print("\nNext steps:")
print("1. Place your model in the 'models' directory")
print("2. Run: python precompute_dictionary_embeddings.py --help")
print("3. Run: python unified_analysis_pipeline.py --help")
