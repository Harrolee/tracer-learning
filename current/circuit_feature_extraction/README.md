# Circuit Feature Extraction for Semantic Connectivity Analysis

This directory contains a self-contained pipeline for extracting circuit features from the Gemma-2-2B model using Circuit Tracer. This is Part 2 of the semantic connectivity research project.

## Overview

The goal is to extract circuit activation features for 5000 words that have already been analyzed for semantic connectivity patterns. We want to correlate how words' semantic neighborhoods evolve across layers with their circuit complexity.

## Quick Start (GPU Machine)

```bash
# 1. Clone/copy this directory to GPU machine
scp -r circuit_feature_extraction/ user@gpu-machine:~/

# 2. SSH to GPU machine
ssh user@gpu-machine

# 3. Run the setup script (installs everything)
cd circuit_feature_extraction
bash setup_gpu.sh

# 4. Run extraction
./run_extraction.sh
# Or with monitoring in tmux:
./run_with_monitoring.sh
```

## Files

- `extract_features.py` - Main extraction script with checkpointing
- `setup_gpu.sh` - Complete setup script for fresh GPU environment
- `run_extraction.sh` - Simple run script
- `run_with_monitoring.sh` - Run with tmux monitoring windows
- `sampled_words.json` - 5000 words to analyze
- `requirements.txt` - Python dependencies

## Manual Setup (if needed)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install circuit-tracer
pip install git+https://github.com/sae-lens/circuit-tracer.git

# Install other dependencies
pip install -r requirements.txt

# Note: The model (google/gemma-2-2b) will be downloaded automatically
# by circuit-tracer on first run from HuggingFace
```

## Running the Extraction

### Basic Usage
```bash
python extract_features.py \
    --model google/gemma-2-2b \
    --words-file sampled_words.json \
    --output-dir results \
    --device cuda
```

### With Checkpointing (Recommended)
```bash
python extract_features.py \
    --model google/gemma-2-2b \
    --words-file sampled_words.json \
    --output-dir results \
    --device cuda \
    --checkpoint-frequency 100 \
    --resume
```

### Command-line Options
- `--model`: Model name (ignored - always uses google/gemma-2-2b)
- `--words-file`: JSON file with words to analyze (default: sampled_words.json)
- `--output-dir`: Where to save results (default: results)
- `--device`: cuda, cpu, or mps (default: cpu)
- `--checkpoint-frequency`: Save progress every N words (default: 100)
- `--resume`: Resume from checkpoint if interrupted
- `--top-k-features`: Keep top K features per layer (default: 10)
- `--test-words`: Process only first N words for testing (0=all)

## Output Files

The extraction creates these files in the output directory:

- `feature_activations.csv` - Main output with features per word per layer
- `circuit_checkpoint.json` - Checkpoint file for resuming
- `extraction.log` - Detailed log of the extraction process

### Output Format

`feature_activations.csv` contains:
- `word`: The analyzed word
- `layer`: Layer index (0-26 for Gemma)
- `feature_id`: ID of the activated feature
- `activation_strength`: Strength of activation

## Monitoring Progress

When using `run_with_monitoring.sh`, you get three tmux windows:
1. **extraction** - Main extraction process
2. **monitor** - GPU usage and extraction progress
3. **htop** - System resource usage

Tmux commands:
- Attach: `tmux attach -t circuit_extraction_[timestamp]`
- Switch windows: `Ctrl+B`, then `0`/`1`/`2`
- Detach: `Ctrl+B`, then `d`

## Troubleshooting

### CUDA out of memory
Reduce batch size or use CPU offloading:
```python
# In extract_features.py, modify the attribute call:
graph = attribute(
    prompt,
    self.model,
    batch_size=1,  # Reduce this
    offload="cpu",  # Enable CPU offloading
    ...
)
```

### No features found
This can happen if:
1. The prompt doesn't activate many features
2. The thresholds are too high

Try modifying the prompt format in `extract_features.py`:
```python
# Current: "The word {word} means"
# Try: "{word}" or "The {word} is"
```

### Checkpoint issues
If checkpoint is corrupted:
```bash
rm results/circuit_checkpoint.json
# Then restart without --resume flag
```

## Research Context

This extraction is part of a larger project testing whether semantic connectivity evolution predicts circuit complexity. The pipeline:

1. **Part 1** (Completed): Semantic connectivity analysis - tracking how word neighborhoods change across layers
2. **Part 2** (This): Circuit feature extraction - identifying which features activate for each word
3. **Part 3** (Future): Correlation analysis - testing if connectivity patterns predict feature complexity

The hypothesis is that words with more dynamic semantic trajectories (changing neighborhoods across layers) will show more complex circuit activation patterns.

## GPU Requirements

- Minimum: 16GB VRAM (e.g., V100, A10)
- Recommended: 24GB+ VRAM (e.g., A100, A6000)
- For 5000 words: ~8-12 hours on A100

## Contact

For issues or questions about this extraction pipeline, refer to the main project documentation or CLAUDE.md in the parent directory.