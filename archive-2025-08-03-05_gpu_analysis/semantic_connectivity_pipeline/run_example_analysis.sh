#!/bin/bash
# Example script showing complete analysis workflow
# This demonstrates a small-scale analysis for testing

set -e

echo "=== Semantic Connectivity Pipeline Example ==="
echo "This will run a small example analysis (100 words)"
echo

# Check if environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source activate_pipeline.sh
fi

# Set model path (update this to your model location)
MODEL_PATH="${MODEL_PATH:-models/gemma-2b}"
echo "Using model: $MODEL_PATH"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or download a model"
    echo "Example: export MODEL_PATH=/path/to/your/model"
    exit 1
fi

# Step 1: Precompute dictionary embeddings (small subset for demo)
DICT_DIR="dictionary_embeddings_demo"
if [ ! -d "$DICT_DIR" ]; then
    echo -e "\n--- Step 1: Precomputing dictionary embeddings ---"
    echo "Note: Using only 1000 words for demo (full dictionary would be ~77k words)"
    
    python precompute_dictionary_embeddings.py \
        --model "$MODEL_PATH" \
        --output-dir "$DICT_DIR" \
        --device cpu \
        --batch-size 16 \
        --max-words 1000
else
    echo -e "\n--- Step 1: Dictionary embeddings already exist ---"
    echo "Delete $DICT_DIR to recompute"
fi

# Step 2: Run analysis pipeline
RESULTS_DIR="results_demo_100"
echo -e "\n--- Step 2: Running analysis pipeline ---"
echo "Analyzing 100 words with extreme contrast sampling"

python unified_analysis_pipeline.py \
    --model "$MODEL_PATH" \
    --dictionary-embeddings "$DICT_DIR" \
    --output-dir "$RESULTS_DIR" \
    --sampling-strategy extreme_contrast \
    --num-words 100 \
    --device cpu

# Step 3: Show results
echo -e "\n--- Step 3: Analysis Results ---"
echo "Results saved to: $RESULTS_DIR/"
echo

# Display summary statistics
echo "Word Summary (first 10 rows):"
head -n 11 "$RESULTS_DIR/word_summary.csv" | column -t -s ','

echo -e "\nAnalysis complete!"
echo
echo "To analyze the results:"
echo "1. Open the CSV files in $RESULTS_DIR/"
echo "2. Use the provided Python analysis code in README.md"
echo "3. Or import into Excel/Google Sheets"
echo
echo "For full analysis with 5,000 words:"
echo "1. Precompute full dictionary: remove --max-words flag"
echo "2. Run pipeline with --num-words 5000"
echo "3. Use GPU with --device cuda for faster processing"