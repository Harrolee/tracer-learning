#!/bin/bash
# Setup script for fresh GPU machine

set -e

echo "=== Setting up GPU environment for circuit analysis ==="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install pandas numpy tqdm
pip install matplotlib seaborn scipy

# Install circuit-tracer
echo "Installing circuit-tracer..."
# Clone circuit-tracer if not present
if [ ! -d "circuit-tracer" ]; then
    git clone https://github.com/jbloomAus/circuit-tracer.git
fi
cd circuit-tracer
pip install -e .
cd ..

# Install transformer-lens (dependency of circuit-tracer)
pip install transformer-lens

# Create necessary directories
mkdir -p logs
mkdir -p models

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Copy your results: scp -r results_full_analysis_5k_5000words ubuntu@<this-machine>:~/"
echo "2. Copy the scripts:"
echo "   scp extract_circuit_features_standalone.py ubuntu@<this-machine>:~/"
echo "   scp run_feature_extraction_tmux.sh ubuntu@<this-machine>:~/"
echo "3. Run: ./run_feature_extraction_tmux.sh results_full_analysis_5k_5000words"