#!/bin/bash
# Script to run unified analysis pipeline overnight with tmux
# Processes semantic connectivity and circuit features for sampled words

set -e

# Configuration
JOB_NAME="${1:-unified_analysis}"
NUM_WORDS="${2:-5000}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/tracer-learning/models/gemma-2b}"
DICT_EMBEDDINGS="${3:-dictionary_embeddings_gemma_embeddings}"  # Directory with precomputed embeddings
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"  # Batch size for circuit feature extraction

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Unified Analysis Pipeline Overnight Job ===${NC}"
echo "Job name: $JOB_NAME"
echo "Model: $MODEL_PATH"
echo "Dictionary embeddings: $DICT_EMBEDDINGS"
echo "Number of words: $NUM_WORDS"
echo "Device: $DEVICE"

# Check if dictionary embeddings exist
if [ ! -d "$DICT_EMBEDDINGS" ]; then
    echo -e "${RED}Error: Dictionary embeddings not found at $DICT_EMBEDDINGS${NC}"
    echo "Please run precompute_dictionary_embeddings first"
    exit 1
fi

# Check if metadata exists
if [ ! -f "$DICT_EMBEDDINGS/embedding_metadata.json" ]; then
    echo -e "${RED}Error: No metadata file in $DICT_EMBEDDINGS${NC}"
    echo "Invalid dictionary embeddings directory"
    exit 1
fi

# Create output directories
OUTPUT_DIR="results_${JOB_NAME}_${NUM_WORDS}words"
LOG_DIR="logs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Activate virtual environment if it exists
if [ -f "semantic_connectivity_pipeline/activate_pipeline.sh" ]; then
    source semantic_connectivity_pipeline/activate_pipeline.sh
elif [ -f "activate_pipeline.sh" ]; then
    source activate_pipeline.sh
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create job script
JOB_SCRIPT="$LOG_DIR/job_script.sh"
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
# Auto-generated job script for unified analysis

echo "Starting unified analysis at \$(date)"
echo "================================"
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dictionary embeddings: $DICT_EMBEDDINGS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of words: $NUM_WORDS"
echo "  Device: $DEVICE"
echo "================================"

# Set environment for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run the unified analysis pipeline
python semantic_connectivity_pipeline/unified_analysis_pipeline.py \\
    --model "$MODEL_PATH" \\
    --dictionary-embeddings "$DICT_EMBEDDINGS" \\
    --output-dir "$OUTPUT_DIR" \\
    --num-words $NUM_WORDS \\
    --sampling-strategy extreme_contrast \\
    --connectivity-threshold 0.7 \\
    --device "$DEVICE" \\
    2>&1 | tee "$LOG_DIR/analysis.log"

echo "================================"
echo "Analysis completed at \$(date)"

# Create summary report
echo "Analysis Summary" > "$LOG_DIR/summary.txt"
echo "================" >> "$LOG_DIR/summary.txt"
echo "Start time: \$(grep 'Starting unified' "$LOG_DIR/analysis.log" | head -1)" >> "$LOG_DIR/summary.txt"
echo "End time: \$(date)" >> "$LOG_DIR/summary.txt"
echo "" >> "$LOG_DIR/summary.txt"

# Check output files
echo "Output Files Created:" >> "$LOG_DIR/summary.txt"
echo "-------------------" >> "$LOG_DIR/summary.txt"
for file in "$OUTPUT_DIR"/*.csv; do
    if [ -f "\$file" ]; then
        lines=\$(wc -l < "\$file")
        size=\$(du -h "\$file" | cut -f1)
        echo "  \$(basename "\$file"): \$lines lines, \$size" >> "$LOG_DIR/summary.txt"
    fi
done

# Show word statistics
echo "" >> "$LOG_DIR/summary.txt"
echo "Word Statistics:" >> "$LOG_DIR/summary.txt"
echo "---------------" >> "$LOG_DIR/summary.txt"
if [ -f "$OUTPUT_DIR/word_summary.csv" ]; then
    echo "  Total words processed: \$(tail -n +2 "$OUTPUT_DIR/word_summary.csv" | wc -l)" >> "$LOG_DIR/summary.txt"
    
    # Get polysemy distribution
    echo "  Polysemy distribution:" >> "$LOG_DIR/summary.txt"
    echo "    Monosemous (1): \$(awk -F',' '\$2==1' "$OUTPUT_DIR/word_summary.csv" | wc -l)" >> "$LOG_DIR/summary.txt"
    echo "    Low (2-3): \$(awk -F',' '\$2>=2 && \$2<=3' "$OUTPUT_DIR/word_summary.csv" | wc -l)" >> "$LOG_DIR/summary.txt"
    echo "    Medium (4-10): \$(awk -F',' '\$2>=4 && \$2<=10' "$OUTPUT_DIR/word_summary.csv" | wc -l)" >> "$LOG_DIR/summary.txt"
    echo "    High (>10): \$(awk -F',' '\$2>10' "$OUTPUT_DIR/word_summary.csv" | wc -l)" >> "$LOG_DIR/summary.txt"
fi

# Check for errors
echo "" >> "$LOG_DIR/summary.txt"
echo "Errors/Warnings:" >> "$LOG_DIR/summary.txt"
echo "---------------" >> "$LOG_DIR/summary.txt"
grep -i "error\|warning\|failed" "$LOG_DIR/analysis.log" | head -10 >> "$LOG_DIR/summary.txt" || echo "  No errors found" >> "$LOG_DIR/summary.txt"

cat "$LOG_DIR/summary.txt"

# Success message
echo ""
echo "✨ Analysis complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - word_summary.csv: Overview of each word"
echo "  - layer_connectivity.csv: Connectivity by layer"
echo "  - connectivity_trajectories.csv: Evolution across layers"
echo "  - feature_activations.csv: Circuit features"
echo ""
echo "To analyze results:"
echo "  python analyze_results.py --results-dir $OUTPUT_DIR"
EOF

chmod +x "$JOB_SCRIPT"

# Create tmux session name
TMUX_SESSION="unified_${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}Starting job in tmux session: $TMUX_SESSION${NC}"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Start job in tmux
tmux new-session -d -s "$TMUX_SESSION"
tmux send-keys -t "$TMUX_SESSION" "bash $JOB_SCRIPT" Enter

echo -e "${GREEN}Job started successfully!${NC}"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To check logs:"
echo "  tail -f $LOG_DIR/analysis.log"
echo ""
echo "To see connectivity processing:"
echo "  grep 'Layer' $LOG_DIR/analysis.log"
echo ""
echo "To see circuit feature extraction:"
echo "  grep 'Extracting features' $LOG_DIR/analysis.log"
echo ""
echo "To detach from tmux: Press Ctrl+B, then D"
echo ""
echo "The job will continue running even if you disconnect."
echo ""
echo -e "${YELLOW}Estimated completion time:${NC}"
echo "  - With 5 layers: 1-2 hours for 5000 words"
echo "  - With 27 layers: 3-5 hours for 5000 words"
echo "  - Circuit feature extraction adds ~30-60 minutes"