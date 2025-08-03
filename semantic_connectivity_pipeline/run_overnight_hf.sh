#!/bin/bash
# Script to run dictionary embedding precomputation overnight with HF upload

set -e

# Configuration
JOB_NAME="${1:-gemma_embeddings}"
HF_REPO_ID="${2:-}"  # e.g., "username/gemma-2b-dictionary-embeddings"
MODEL_PATH="${MODEL_PATH:-/Users/lee/fun/learningSlice/models/gemma-2b}"
DEVICE="${DEVICE:-mps}"  # Use mps for Mac, cuda for GPU
BATCH_SIZE="${BATCH_SIZE:-16}"  # Adjust based on memory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Dictionary Embeddings Overnight Job ===${NC}"
echo "Job name: $JOB_NAME"
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"

# Check if HF repo ID provided
if [ -z "$HF_REPO_ID" ]; then
    echo -e "${RED}Error: HuggingFace repo ID required${NC}"
    echo "Usage: $0 <job_name> <hf_repo_id>"
    echo "Example: $0 gemma_embeddings username/gemma-2b-dictionary-embeddings"
    exit 1
fi

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN environment variable not set${NC}"
    echo "Please set it or the script will prompt for it"
    echo "Export it with: export HF_TOKEN='your_token_here'"
    read -p "Enter your HuggingFace token: " HF_TOKEN
    export HF_TOKEN
fi

# Create output directories
OUTPUT_DIR="dictionary_embeddings_${JOB_NAME}"
LOG_DIR="logs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Activate virtual environment if it exists
if [ -f "activate_pipeline.sh" ]; then
    source activate_pipeline.sh
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create job script
JOB_SCRIPT="$LOG_DIR/job_script.sh"
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
# Auto-generated job script

echo "Starting precomputation at \$(date)"
echo "================================"

python precompute_dictionary_embeddings_hf.py \\
    --model "$MODEL_PATH" \\
    --output-dir "$OUTPUT_DIR" \\
    --device "$DEVICE" \\
    --batch-size "$BATCH_SIZE" \\
    --download-nltk \\
    --upload-to-hf \\
    --hf-repo-id "$HF_REPO_ID" \\
    --hf-token "$HF_TOKEN" \\
    2>&1 | tee "$LOG_DIR/precompute.log"

echo "================================"
echo "Job completed at \$(date)"

# Create summary
echo "Job Summary" > "$LOG_DIR/summary.txt"
echo "----------" >> "$LOG_DIR/summary.txt"
echo "Start time: \$start_time" >> "$LOG_DIR/summary.txt"
echo "End time: \$(date)" >> "$LOG_DIR/summary.txt"
echo "Output directory: $OUTPUT_DIR" >> "$LOG_DIR/summary.txt"
echo "HuggingFace repo: https://huggingface.co/datasets/$HF_REPO_ID" >> "$LOG_DIR/summary.txt"

# Verify results
python precompute_dictionary_embeddings_hf.py \\
    --output-dir "$OUTPUT_DIR" \\
    --verify-only \\
    >> "$LOG_DIR/summary.txt" 2>&1

cat "$LOG_DIR/summary.txt"
EOF

chmod +x "$JOB_SCRIPT"

# Create tmux session name
TMUX_SESSION="embeddings_${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}Starting job in tmux session: $TMUX_SESSION${NC}"
echo "Logs will be saved to: $LOG_DIR"
echo "HuggingFace repo: https://huggingface.co/datasets/$HF_REPO_ID"
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
echo "  tail -f $LOG_DIR/precompute.log"
echo ""
echo "To detach from tmux: Press Ctrl+B, then D"
echo ""
echo "The job will continue running even if you disconnect."
echo ""
echo -e "${YELLOW}Estimated completion time: 4-6 hours for full WordNet dictionary${NC}"