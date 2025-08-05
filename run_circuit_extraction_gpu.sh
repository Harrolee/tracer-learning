#!/bin/bash
# Script to run circuit feature extraction on Lambda Labs GPU
# This completes Part 2 of the unified analysis pipeline

set -e

# Configuration
RESULTS_DIR="${1:-results_full_analysis_5k_5000words}"
NUM_TEST_WORDS="${2:-0}"  # 0 = all words, or specify number for testing
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/tracer-learning/models/gemma-2b}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Circuit Feature Extraction (Part 2) ===${NC}"
echo "Results directory: $RESULTS_DIR"
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory not found at $RESULTS_DIR${NC}"
    echo "Please run semantic connectivity analysis (Part 1) first"
    exit 1
fi

# Check if sampled_words.json exists
if [ ! -f "$RESULTS_DIR/sampled_words.json" ]; then
    echo -e "${RED}Error: No sampled_words.json found in $RESULTS_DIR${NC}"
    echo "Please run semantic connectivity analysis (Part 1) first"
    exit 1
fi

# Check if feature_activations.csv already exists and has content
if [ -f "$RESULTS_DIR/feature_activations.csv" ]; then
    lines=$(wc -l < "$RESULTS_DIR/feature_activations.csv")
    if [ $lines -gt 1 ]; then
        echo -e "${YELLOW}Warning: feature_activations.csv already exists with $lines lines${NC}"
        echo "Do you want to resume from checkpoint? (y/n)"
        read -r response
        if [ "$response" = "y" ]; then
            RESUME_FLAG="--resume"
        else
            echo "Backing up existing file..."
            mv "$RESULTS_DIR/feature_activations.csv" "$RESULTS_DIR/feature_activations.csv.bak"
            RESUME_FLAG=""
        fi
    else
        RESUME_FLAG=""
    fi
else
    RESUME_FLAG=""
fi

# Create log directory
LOG_DIR="logs/circuit_extraction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}Starting circuit feature extraction...${NC}"
echo "Logs will be saved to: $LOG_DIR/extraction.log"

# Set environment for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run the extraction
if [ $NUM_TEST_WORDS -gt 0 ]; then
    echo -e "${YELLOW}Running in test mode with $NUM_TEST_WORDS words${NC}"
    TEST_FLAG="--test-words $NUM_TEST_WORDS"
else
    echo "Running full extraction for all words..."
    TEST_FLAG=""
fi

# Execute the script
python extract_circuit_features_standalone.py \
    --model "$MODEL_PATH" \
    --results-dir "$RESULTS_DIR" \
    --device "$DEVICE" \
    --top-k-features 10 \
    --checkpoint-frequency 100 \
    $RESUME_FLAG \
    $TEST_FLAG \
    2>&1 | tee "$LOG_DIR/extraction.log"

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✨ Circuit feature extraction completed successfully!${NC}"
    
    # Generate summary
    echo "Generating summary..."
    python -c "
import pandas as pd
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
features_file = results_dir / 'feature_activations.csv'
words_file = results_dir / 'sampled_words.json'

if features_file.exists():
    df = pd.read_csv(features_file)
    with open(words_file, 'r') as f:
        total_words = len(json.load(f))
    
    print(f'\n📊 Extraction Summary:')
    print(f'  - Total words in dataset: {total_words}')
    if len(df) > 0:
        print(f'  - Words with features: {df["word"].nunique()}')
        print(f'  - Total feature activations: {len(df)}')
        if df["word"].nunique() > 0:
            print(f'  - Average features per word: {len(df) / df["word"].nunique():.1f}')
        print(f'  - Layers with features: {sorted(df["layer"].unique())}')
    else:
        print('  - No features extracted yet')
"
    
    # Update completion status
    echo -e "\n${GREEN}Both parts of the analysis are now complete!${NC}"
    echo "You can now run correlation analysis with:"
    echo "  python analyze_connectivity_circuit_correlation.py --results-dir $RESULTS_DIR"
    
else
    echo -e "${RED}Error: Circuit feature extraction failed!${NC}"
    echo "Check the log file: $LOG_DIR/extraction.log"
    exit 1
fi