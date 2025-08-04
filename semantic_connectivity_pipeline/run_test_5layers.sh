#!/bin/bash
# Quick test script for unified pipeline with 5-layer embeddings
# Runs a small test while the full 27-layer embeddings compute

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Quick Test: Unified Pipeline with 5 Layers ===${NC}"
echo

# Configuration
NUM_WORDS="${1:-100}"  # Default 100 words for quick test
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/tracer-learning/models/gemma-2b}"
DICT_5LAYERS="dictionary_embeddings_gemma_embeddings"  # The original 5-layer version
OUTPUT_DIR="test_results_5layers_${NUM_WORDS}words"
DEVICE="${DEVICE:-cuda}"

# Check dictionary exists
if [ ! -d "$DICT_5LAYERS" ]; then
    echo -e "${RED}Error: 5-layer dictionary not found at $DICT_5LAYERS${NC}"
    echo "Available directories:"
    ls -d dictionary_embeddings* 2>/dev/null || echo "  None found"
    exit 1
fi

# Show what layers we have
echo "Checking available layers..."
LAYER_COUNT=$(ls -1 "$DICT_5LAYERS"/embeddings_layer_*.pkl 2>/dev/null | wc -l)
echo "Found $LAYER_COUNT layer files in $DICT_5LAYERS:"
ls -lh "$DICT_5LAYERS"/embeddings_layer_*.pkl | awk '{print "  "$9": "$5}'
echo

# Activate environment
if [ -f "semantic_connectivity_pipeline/activate_pipeline.sh" ]; then
    source semantic_connectivity_pipeline/activate_pipeline.sh
elif [ -f "activate_pipeline.sh" ]; then
    source activate_pipeline.sh
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Running test with:${NC}"
echo "  Model: $MODEL_PATH"
echo "  Dictionary: $DICT_5LAYERS (5 layers)"
echo "  Output: $OUTPUT_DIR"
echo "  Words: $NUM_WORDS"
echo "  Device: $DEVICE"
echo

# Run the test
echo "Starting analysis..."
START_TIME=$(date +%s)

python semantic_connectivity_pipeline/unified_analysis_pipeline.py \
    --model "$MODEL_PATH" \
    --dictionary-embeddings "$DICT_5LAYERS" \
    --output-dir "$OUTPUT_DIR" \
    --num-words "$NUM_WORDS" \
    --sampling-strategy extreme_contrast \
    --connectivity-threshold 0.7 \
    --device "$DEVICE" \
    --download-nltk \
    2>&1 | tee "test_5layers_${NUM_WORDS}words.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo -e "${GREEN}=== Test Complete ===${NC}"
echo "Time taken: $ELAPSED seconds"
echo

# Check results
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output files created:"
    for file in "$OUTPUT_DIR"/*.csv; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            size=$(du -h "$file" | cut -f1)
            echo "  $(basename "$file"): $lines lines, $size"
        fi
    done
    
    echo
    echo "Sample results (first 5 words):"
    if [ -f "$OUTPUT_DIR/word_summary.csv" ]; then
        head -n 6 "$OUTPUT_DIR/word_summary.csv" | column -t -s ','
    fi
fi

echo
echo -e "${YELLOW}Performance metrics:${NC}"
echo "  Words processed: $NUM_WORDS"
echo "  Time: $ELAPSED seconds"
echo "  Rate: $(echo "scale=2; $NUM_WORDS / $ELAPSED" | bc) words/second"
echo

# Estimate for full run
FULL_WORDS=5000
ESTIMATED=$(echo "scale=0; $ELAPSED * $FULL_WORDS / $NUM_WORDS / 60" | bc)
echo -e "${YELLOW}Estimated time for $FULL_WORDS words: ~$ESTIMATED minutes${NC}"

echo
echo "To run full analysis with 5000 words:"
echo "  ./semantic_connectivity_pipeline/run_unified_overnight.sh full_5k 5000 $DICT_5LAYERS"