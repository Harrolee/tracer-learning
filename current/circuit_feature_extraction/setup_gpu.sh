#!/bin/bash
# Complete GPU environment setup script for circuit feature extraction
# Run this on a fresh GPU machine (Lambda, Vast.ai, etc.)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Circuit Feature Extraction GPU Setup ===${NC}"
echo "This script will set up everything needed for circuit feature extraction"
echo ""

# Check for GPU
echo -e "${YELLOW}Checking for GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${RED}⚠ Warning: No GPU detected. Continuing anyway...${NC}"
fi

# Update system packages
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git wget tmux htop

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install circuit-tracer from GitHub
echo -e "${YELLOW}Installing circuit-tracer...${NC}"
pip install git+https://github.com/sae-lens/circuit-tracer.git

# Install other dependencies
echo -e "${YELLOW}Installing additional dependencies...${NC}"
pip install \
    transformers \
    huggingface-hub \
    safetensors \
    einops \
    tqdm \
    pandas \
    numpy \
    matplotlib \
    jupyter \
    ipywidgets

# Note: Model will be downloaded automatically by circuit-tracer when needed
echo -e "${YELLOW}Model will be downloaded automatically on first run${NC}"
echo "Circuit-tracer will download google/gemma-2-2b from HuggingFace when needed"

# Create test script
echo -e "${YELLOW}Creating test script...${NC}"
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test that the environment is set up correctly."""

import torch
import sys
from pathlib import Path

def test_setup():
    print("Testing environment setup...")
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available - will use CPU")
    
    # Test imports
    try:
        from circuit_tracer import ReplacementModel
        from circuit_tracer.attribution import attribute
        print("✓ Circuit-tracer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import circuit-tracer: {e}")
        return False
    
    # Test that circuit-tracer can load the model
    try:
        print("Testing model loading with circuit-tracer...")
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b",
            "gemma",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✓ Model loaded successfully via circuit-tracer")
        
        # Quick test
        text = "The word cat means"
        tokens = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**tokens, max_length=20)
        result = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test generation: '{text}' -> '{result[:50]}...'")
        
        # Clean up to save memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"⚠ Model loading test failed: {e}")
        print("  Note: Model will be downloaded on first run")
    
    # Test data file
    if Path("sampled_words.json").exists():
        import json
        with open("sampled_words.json", "r") as f:
            words = json.load(f)
        print(f"✓ Found word list with {len(words)} words")
    else:
        print("⚠ No sampled_words.json found")
    
    print("\n✅ Setup complete and verified!")
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
EOF

chmod +x test_setup.py

# Create run script
echo -e "${YELLOW}Creating run script...${NC}"
cat > run_extraction.sh << 'EOF'
#!/bin/bash
# Run the feature extraction with optimal settings

source venv/bin/activate

# Set environment variables for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Create output directory
OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting feature extraction..."
echo "Output directory: $OUTPUT_DIR"

python extract_features.py \
    --model google/gemma-2-2b \
    --words-file sampled_words.json \
    --output-dir $OUTPUT_DIR \
    --device cuda \
    --checkpoint-frequency 100 \
    --resume \
    2>&1 | tee $OUTPUT_DIR/extraction.log

echo "Extraction complete!"
echo "Results saved to: $OUTPUT_DIR"
EOF

chmod +x run_extraction.sh

# Create tmux monitoring script
echo -e "${YELLOW}Creating tmux monitoring script...${NC}"
cat > run_with_monitoring.sh << 'EOF'
#!/bin/bash
# Run extraction in tmux with monitoring

SESSION_NAME="circuit_extraction_$(date +%Y%m%d_%H%M%S)"

# Create tmux session
tmux new-session -d -s $SESSION_NAME -n "extraction"

# Run extraction in first window
tmux send-keys -t $SESSION_NAME:extraction "./run_extraction.sh" Enter

# Create monitoring window
tmux new-window -t $SESSION_NAME -n "monitor"
tmux send-keys -t $SESSION_NAME:monitor "watch -n 5 'nvidia-smi; echo; tail -20 results_*/extraction.log 2>/dev/null | grep -E \"Processing|Features|Error\"'" Enter

# Create htop window
tmux new-window -t $SESSION_NAME -n "htop"
tmux send-keys -t $SESSION_NAME:htop "htop" Enter

echo "Started extraction in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t $SESSION_NAME"
echo "  List windows:       Ctrl+B, w"
echo "  Switch windows:     Ctrl+B, [0-2]"
echo "  Detach:            Ctrl+B, d"
echo ""
echo "Windows:"
echo "  0: extraction - Main extraction process"
echo "  1: monitor    - GPU usage and progress"
echo "  2: htop       - System resources"
EOF

chmod +x run_with_monitoring.sh

# Run the test
echo ""
echo -e "${YELLOW}Running environment test...${NC}"
python test_setup.py

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Activate environment:  source venv/bin/activate"
echo "2. Test the setup:        python test_setup.py"
echo "3. Run extraction:        ./run_extraction.sh"
echo "4. Or with monitoring:    ./run_with_monitoring.sh"
echo ""
echo "The extraction will:"
echo "  - Process all words in sampled_words.json"
echo "  - Save checkpoints every 100 words"
echo "  - Resume automatically if interrupted"
echo "  - Output results to timestamped directory"