#!/bin/bash
# Script to run the complete fine-tuning pipeline

set -e  # Exit on error

echo "WS3: Fine-Tuning Pipeline for Gemma-2B"
echo "======================================"

# Activate virtual environment
echo "Activating virtual environment..."
source /Users/lee/fun/learningSlice/.venv/bin/activate

# Check if data exists
DATA_PATH="/Users/lee/fun/learningSlice/data/ws2_synthetic_corpus_hf"
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Synthetic corpus not found at $DATA_PATH"
    echo "Please run WS2 first to generate the corpus"
    exit 1
fi

echo -e "\n1. Dataset found at $DATA_PATH"

# Test model loading
echo -e "\n2. Testing model loading..."
python test_model.py

# Run test training
echo -e "\n3. Running test training (10 examples)..."
python finetune_gemma.py \
    --data_path "$DATA_PATH" \
    --output_dir ./outputs \
    --test_run \
    --use_lora \
    --batch_size 1 \
    --max_length 256

echo -e "\nTest run complete!"
echo "To run full training, use:"
echo "python finetune_gemma.py --data_path \"$DATA_PATH\" --output_dir ./outputs --use_lora --batch_size 1"