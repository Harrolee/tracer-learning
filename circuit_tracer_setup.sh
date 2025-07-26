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
    git clone https://github.com/safety-research/circuit-tracer
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
pip install .
cd ..

# Install additional useful packages (including huggingface_hub for transcoder access)
echo -e "\n7. Installing additional packages..."
pip install wandb jupyter ipykernel matplotlib seaborn huggingface_hub

# Create activation script
echo -e "\n8. Creating activation script..."
cat > "$WORK_DIR/activate_circuit_tracer.sh" << 'EOF'
#!/bin/bash
# Activate circuit-tracer environment

WORK_DIR="$HOME/circuit-tracer-workspace"
source "$WORK_DIR/circuit-tracer-env/bin/activate"
echo "Circuit Tracer environment activated!"
echo "You can now use: circuit-tracer attribute --prompt <prompt> --transcoder_set <set> --slug <name> --graph_file_dir <dir> --dtype=bfloat16 --server"
echo ""
echo "Available transcoder sets:"
echo "  - gemma: for google/gemma-2-2b (from GemmaScope)"
echo "  - llama: for meta-llama/Llama-3.2-1B"
echo ""
echo "Example usage:"
echo "  circuit-tracer attribute -p 'The capital of France is' -t gemma --slug france-demo --graph_file_dir ./graphs --dtype=bfloat16 --server"
EOF

chmod +x "$WORK_DIR/activate_circuit_tracer.sh"

# Create example script
echo -e "\n9. Creating example script..."
cat > "$WORK_DIR/test_circuit_tracer.py" << 'EOF'
#!/usr/bin/env python3
"""Test Circuit Tracer installation"""

import torch
import os

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

# Test importing required dependencies
try:
    import huggingface_hub
    print("✓ Hugging Face Hub imported successfully")
except ImportError as e:
    print(f"✗ Failed to import huggingface_hub: {e}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ Transformers imported successfully")
except ImportError as e:
    print(f"✗ Failed to import transformers: {e}")

print("\nSetup complete! You can now use circuit-tracer from the command line.")
print("\nCurrent CLI Usage:")
print("  circuit-tracer attribute --prompt <PROMPT> --transcoder_set <SET> --slug <NAME> --graph_file_dir <DIR> --dtype=bfloat16 [--server]")
print("\nAvailable transcoder sets:")
print("  - gemma: for google/gemma-2-2b model")
print("  - llama: for meta-llama/Llama-3.2-1B model")
print("\nExample commands:")
print("  # Complete workflow with visualization:")
print('  circuit-tracer attribute -p "The capital of France is" -t gemma --slug france-demo --graph_file_dir ./graphs --dtype=bfloat16 --server')
print("")
print("  # Attribution only (save raw graph):")
print('  circuit-tracer attribute -p "The sky is blue" -t llama --graph_output_path sky_analysis.pt --dtype=bfloat16')
print("")
print("  # With custom parameters:")
print('  circuit-tracer attribute -p "Machine learning is" -t gemma --slug ml-demo --graph_file_dir ./graphs --max_n_logits 5 --batch_size 128 --dtype=bfloat16 --server')
print("\nFor more options, run: circuit-tracer attribute --help")
EOF

chmod +x "$WORK_DIR/test_circuit_tracer.py"

# Create example graph directory
mkdir -p "$WORK_DIR/example_graphs"

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
echo "Current Circuit Tracer CLI Usage:"
echo "  circuit-tracer attribute --prompt <PROMPT> --transcoder_set <SET> --slug <NAME> --graph_file_dir <DIR> --dtype=bfloat16 [OPTIONS]"
echo
echo "Available transcoder sets:"
echo "  - gemma: for google/gemma-2-2b (requires ~15GB GPU RAM)"
echo "  - llama: for meta-llama/Llama-3.2-1B"
echo
echo "Example commands:"
echo '  # Complete analysis with visualization:'
echo '  circuit-tracer attribute -p "The capital of France is" -t gemma --slug france-demo --graph_file_dir ./example_graphs --dtype=bfloat16 --server'
echo
echo '  # Save raw attribution graph only:'
echo '  circuit-tracer attribute -p "Hello world" -t llama --graph_output_path hello_analysis.pt --dtype=bfloat16'
echo
echo "Note: When using --server, your browser will open to visualize the graph."
echo "If running on a remote machine, make sure to enable port forwarding!"
echo

# Activate environment for current session
echo "Activating environment for current session..."
source "$WORK_DIR/circuit-tracer-env/bin/activate"

# Run test
echo -e "\nRunning installation test..."
python "$WORK_DIR/test_circuit_tracer.py"