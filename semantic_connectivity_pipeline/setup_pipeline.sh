#!/bin/bash
# Complete Setup Script for Semantic Connectivity Pipeline
# This script installs all dependencies needed to run the unified analysis

set -e  # Exit on error

echo "=== Semantic Connectivity Pipeline Setup ==="
echo "This script will install all dependencies for the analysis pipeline"
echo

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo "Detected OS: $OS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -e "\n1. Checking Python installation..."
if ! command_exists python3; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo -e "\n2. Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\n3. Upgrading pip..."
pip install --upgrade pip

# Install PyTorch based on OS and available hardware
echo -e "\n4. Installing PyTorch..."
if [[ "$OS" == "macos" ]]; then
    # macOS - use MPS-enabled PyTorch
    echo "Installing PyTorch for macOS (MPS support)..."
    pip install torch torchvision torchaudio
elif command_exists nvidia-smi; then
    # Linux with NVIDIA GPU
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # Linux CPU only
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo -e "\n5. Installing core dependencies..."
pip install transformers>=4.30.0
pip install numpy pandas matplotlib seaborn
pip install tqdm
pip install nltk
pip install scikit-learn scipy

# Download NLTK data
echo -e "\n6. Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('words', quiet=True)
print('NLTK data downloaded successfully')
"

# Install circuit-tracer
echo -e "\n7. Installing circuit-tracer..."
if [ -f "circuit_tracer_setup.sh" ]; then
    echo "Found circuit_tracer_setup.sh - running it..."
    # Create a minimal version that installs in our venv
    pip install git+https://github.com/safety-research/circuit-tracer.git
else
    echo "Installing circuit-tracer from GitHub..."
    pip install git+https://github.com/safety-research/circuit-tracer.git
fi

# Create directories
echo -e "\n8. Creating required directories..."
mkdir -p models
mkdir -p dictionary_embeddings
mkdir -p results

# Create activation script
echo -e "\n9. Creating activation script..."
cat > activate_pipeline.sh << 'EOF'
#!/bin/bash
# Activate pipeline environment

source venv/bin/activate
echo "Pipeline environment activated!"
echo
echo "Quick start:"
echo "1. First time only - precompute dictionary embeddings:"
echo "   python precompute_dictionary_embeddings.py --model <model_path> --output-dir dictionary_embeddings"
echo
echo "2. Run analysis:"
echo "   python unified_analysis_pipeline.py --model <model_path> --dictionary-embeddings dictionary_embeddings"
echo
EOF

chmod +x activate_pipeline.sh

# Create test script
echo -e "\n10. Creating test script..."
cat > test_installation.py << 'EOF'
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
EOF

chmod +x test_installation.py

# Create requirements.txt for reference
echo -e "\n11. Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core ML frameworks
torch>=2.0.0
transformers>=4.30.0

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

# NLP
nltk>=3.6.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0

# Circuit analysis (install from git)
# circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git
EOF

# Final message
echo -e "\n=== Setup Complete! ==="
echo
echo "Environment created in: ./venv"
echo
echo "To activate the environment:"
echo "  source activate_pipeline.sh"
echo
echo "To test installation:"
echo "  python test_installation.py"
echo
echo "To run the pipeline:"
echo "  1. Activate environment: source activate_pipeline.sh"
echo "  2. Precompute embeddings (one-time, ~4-6 hours):"
echo "     python precompute_dictionary_embeddings.py \\"
echo "         --model path/to/model \\"
echo "         --output-dir dictionary_embeddings"
echo "  3. Run analysis (~30-60 minutes):"
echo "     python unified_analysis_pipeline.py \\"
echo "         --model path/to/model \\"
echo "         --dictionary-embeddings dictionary_embeddings \\"
echo "         --output-dir results"
echo
echo "For detailed usage, see README.md"

# Run test
echo -e "\nRunning installation test..."
python test_installation.py