#!/bin/bash
# Robust overnight job runner using tmux
# Handles disconnections, logs everything, and sends notifications

set -e

# Configuration
JOB_NAME="${1:-connectivity_analysis}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="pipeline_${JOB_NAME}_${TIMESTAMP}"
LOG_DIR="logs/${JOB_NAME}_${TIMESTAMP}"
CHECKPOINT_INTERVAL=1000  # Save progress every N words

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print usage
usage() {
    echo "Usage: $0 [job_name] [precompute|analyze|full]"
    echo ""
    echo "Jobs:"
    echo "  precompute - Precompute dictionary embeddings (4-6 hours)"
    echo "  analyze    - Run connectivity analysis (1-2 hours)"
    echo "  full       - Run both precompute and analyze"
    echo ""
    echo "Examples:"
    echo "  $0 test_run analyze      # Run analysis with existing embeddings"
    echo "  $0 full_5k full          # Run complete pipeline"
    exit 1
}

# Check job type
JOB_TYPE="${2:-analyze}"
if [[ ! "$JOB_TYPE" =~ ^(precompute|analyze|full)$ ]]; then
    usage
fi

# Create job script
cat > "$LOG_DIR/job_script.sh" << 'EOF'
#!/bin/bash
# Auto-generated job script

set -e  # Exit on error

# Activate environment
cd /Users/lee/fun/learningSlice/semantic_connectivity_pipeline
source venv/bin/activate || source ../venv/bin/activate

echo "=== Job Started: $(date) ==="
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Job type: $JOB_TYPE"

# Error handler
error_handler() {
    echo "=== ERROR at $(date) ==="
    echo "Error occurred in job"
    echo "Check logs for details"
    # Could add email/slack notification here
}
trap error_handler ERR

# Set model path
MODEL_PATH="${MODEL_PATH:-/Users/lee/fun/learningSlice/models/gemma-2b}"
echo "Model: $MODEL_PATH"

# Determine device
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    DEVICE="cpu"
    echo "No GPU detected, using CPU"
fi

# Run job based on type
case "$JOB_TYPE" in
    precompute)
        echo -e "\n--- Running Dictionary Precomputation ---"
        python precompute_dictionary_embeddings.py \
            --model "$MODEL_PATH" \
            --output-dir dictionary_embeddings \
            --device "$DEVICE" \
            --batch-size 32 \
            --checkpoint-interval 5000 \
            2>&1 | tee -a "$LOG_DIR/precompute.log"
        ;;
        
    analyze)
        echo -e "\n--- Running Connectivity Analysis ---"
        python unified_analysis_pipeline.py \
            --model "$MODEL_PATH" \
            --dictionary-embeddings dictionary_embeddings \
            --output-dir "results_${JOB_NAME}" \
            --sampling-strategy extreme_contrast \
            --num-words 5000 \
            --device "$DEVICE" \
            2>&1 | tee -a "$LOG_DIR/analysis.log"
        ;;
        
    full)
        echo -e "\n--- Running Full Pipeline ---"
        
        # First precompute
        echo "Step 1: Precomputing dictionary embeddings..."
        python precompute_dictionary_embeddings.py \
            --model "$MODEL_PATH" \
            --output-dir dictionary_embeddings \
            --device "$DEVICE" \
            --batch-size 32 \
            --checkpoint-interval 5000 \
            2>&1 | tee -a "$LOG_DIR/precompute.log"
        
        # Then analyze
        echo -e "\nStep 2: Running analysis..."
        python unified_analysis_pipeline.py \
            --model "$MODEL_PATH" \
            --dictionary-embeddings dictionary_embeddings \
            --output-dir "results_${JOB_NAME}" \
            --sampling-strategy extreme_contrast \
            --num-words 5000 \
            --device "$DEVICE" \
            2>&1 | tee -a "$LOG_DIR/analysis.log"
        ;;
esac

echo -e "\n=== Job Completed Successfully: $(date) ==="

# Generate summary report
python -c "
import pandas as pd
import os
from pathlib import Path

results_dir = Path('results_${JOB_NAME}')
if results_dir.exists():
    summary = pd.read_csv(results_dir / 'word_summary.csv')
    print(f'\\nResults Summary:')
    print(f'Total words analyzed: {len(summary)}')
    print(f'Mean total features: {summary[\"total_features\"].mean():.1f}')
    print(f'Mean total connectivity: {summary[\"total_connectivity\"].mean():.1f}')
    
    # Save summary
    with open('$LOG_DIR/summary.txt', 'w') as f:
        f.write(f'Job: ${JOB_NAME}\\n')
        f.write(f'Completed: $(date)\\n')
        f.write(f'Total words: {len(summary)}\\n')
        f.write(f'Results in: {results_dir}\\n')
"

EOF

# Make job script executable
chmod +x "$LOG_DIR/job_script.sh"

# Export variables for job script
export JOB_TYPE LOG_DIR JOB_NAME

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y tmux
    else
        echo "Please install tmux manually"
        exit 1
    fi
fi

# Create or attach to tmux session
echo "Starting tmux session: $SESSION_NAME"
echo "Logs will be saved to: $LOG_DIR"

# Start tmux session with job
tmux new-session -d -s "$SESSION_NAME" "cd $(pwd) && bash $LOG_DIR/job_script.sh"

# Create monitoring script
cat > "$LOG_DIR/monitor.sh" << EOF
#!/bin/bash
# Monitor running job

echo "=== Job Monitor ==="
echo "Session: $SESSION_NAME"
echo "Logs: $LOG_DIR"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    # Attach to see live output"
echo "  tmux ls                         # List all sessions"
echo "  tail -f $LOG_DIR/*.log          # Follow log files"
echo "  tmux kill-session -t $SESSION_NAME  # Kill job"
echo ""
echo "Current status:"
tmux list-sessions 2>/dev/null | grep "$SESSION_NAME" || echo "Session not running"
EOF

chmod +x "$LOG_DIR/monitor.sh"

# Show status
echo "=== Job Started Successfully ==="
echo ""
echo "Job: $JOB_NAME ($JOB_TYPE)"
echo "Session: $SESSION_NAME"
echo "Logs: $LOG_DIR/"
echo ""
echo "To monitor:"
echo "  tmux attach -t $SESSION_NAME    # See live output"
echo "  ./$LOG_DIR/monitor.sh           # Check status"
echo "  tail -f $LOG_DIR/*.log          # Follow logs"
echo ""
echo "To list all pipeline sessions:"
echo "  tmux ls | grep pipeline"
echo ""
echo "The job will continue running even if you disconnect!"