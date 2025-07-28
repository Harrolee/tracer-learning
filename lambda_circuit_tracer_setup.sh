#!/bin/bash
# Circuit Tracer Setup Script for Lambda Labs
# This script installs circuit-tracer as a package on any Lambda Labs instance

set -e  # Exit on error

echo "=== Circuit Tracer Lambda Labs Setup ==="
echo "This script will install circuit-tracer and its dependencies"
echo

# Check if running on Lambda Labs (they typically have CUDA)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. This may not be a GPU instance."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "1. Updating system packages..."
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv build-essential

# Create working directory
WORK_DIR="$HOME/circuit-tracer-workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone circuit-tracer repository
echo -e "\n2. Cloning circuit-tracer repository..."
if [ -d "circuit-tracer" ]; then
    echo "circuit-tracer directory already exists. Pulling latest changes..."
    cd circuit-tracer
    git pull
    cd ..
else
    git clone https://github.com/DavidUdell/circuit-tracer.git
fi

# Create virtual environment
echo -e "\n3. Creating Python virtual environment..."
python3 -m venv circuit-tracer-env

# Activate virtual environment
source circuit-tracer-env/bin/activate

# Upgrade pip
echo -e "\n4. Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo -e "\n5. Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install circuit-tracer as a package
echo -e "\n6. Installing circuit-tracer..."
cd circuit-tracer
pip install -e .
cd ..

# Install additional useful packages
echo -e "\n7. Installing additional packages..."
pip install wandb jupyter ipykernel matplotlib seaborn

# Create activation script
echo -e "\n8. Creating activation script..."
cat > "$WORK_DIR/activate_circuit_tracer.sh" << 'EOF'
#!/bin/bash
# Activate circuit-tracer environment

WORK_DIR="$HOME/circuit-tracer-workspace"
source "$WORK_DIR/circuit-tracer-env/bin/activate"
echo "Circuit Tracer environment activated!"
echo "You can now use: circuit-tracer trace --model-name <model> --prompt <prompt>"
EOF

chmod +x "$WORK_DIR/activate_circuit_tracer.sh"

# Create example script
echo -e "\n9. Creating example script..."
cat > "$WORK_DIR/test_circuit_tracer.py" << 'EOF'
#!/usr/bin/env python3
"""Test Circuit Tracer installation"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing Circuit Tracer installation...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Test importing circuit_tracer
try:
    import circuit_tracer
    print("✓ Circuit Tracer imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Circuit Tracer: {e}")

# Test loading a small model
print("\nTesting model loading with GPT-2...")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("✓ Model loading successful")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

print("\nSetup complete! You can now use circuit-tracer from the command line.")
print("\nExample commands:")
print("  circuit-tracer trace --model-name gpt2 --prompt 'Hello world'")
print("  circuit-tracer trace --model-name google/gemma-2-2b --prompt 'The crystal clear water'")
EOF

chmod +x "$WORK_DIR/test_circuit_tracer.py"

# Final setup info
echo -e "\n=== Setup Complete! ==="
echo
echo "Circuit Tracer has been installed in: $WORK_DIR"
echo
echo "To activate the environment in future sessions, run:"
echo "  source $WORK_DIR/activate_circuit_tracer.sh"
echo
echo "To test the installation, run:"
echo "  python $WORK_DIR/test_circuit_tracer.py"
echo
echo "Example circuit-tracer commands:"
echo "  circuit-tracer trace --model-name gpt2 --prompt 'Hello world'"
echo "  circuit-tracer trace --model-name google/gemma-2-2b --prompt 'The sky is blue'"
echo
echo "For your fine-tuned models:"
echo "  circuit-tracer trace --model-path /path/to/model --prompt 'Your prompt'"
echo

# Activate environment for current session
echo "Activating environment for current session..."
source "$WORK_DIR/circuit-tracer-env/bin/activate"

# Run test
echo -e "\nRunning installation test..."
python "$WORK_DIR/test_circuit_tracer.py"