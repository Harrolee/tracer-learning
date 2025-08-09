#!/bin/bash
# Robust circuit feature extraction job runner using tmux
# Handles disconnections, logs everything, creates checkpoints

set -e

# Configuration
JOB_NAME="${1:-circuit_extraction}"
NUM_WORDS="${2:-0}"  # 0 means all words
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="circuit_${JOB_NAME}_${TIMESTAMP}"
LOG_DIR="logs/${JOB_NAME}_${TIMESTAMP}"
CHECKPOINT_INTERVAL="${3:-100}"  # Save progress every N words

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${GREEN}=== Circuit Feature Extraction Job ===${NC}"
echo "Job name: $JOB_NAME"
echo "Session: $SESSION_NAME"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL words"
echo "Logs: $LOG_DIR"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux not found${NC}"
    echo "Please install tmux: apt-get install tmux"
    exit 1
fi

# Create main job script
cat > "$LOG_DIR/job_script.sh" << 'EOF'
#!/bin/bash
# Auto-generated job script for circuit feature extraction

set -e  # Exit on error

echo "=== Job Started: $(date) ==="
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
else
    DEVICE="cpu"
    echo "No GPU detected, using CPU"
fi

# Set environment for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Error handler
error_handler() {
    echo "=== ERROR at $(date) ==="
    echo "Error occurred during extraction"
    echo "Check checkpoint file to resume"
    
    # Show last checkpoint status
    if [ -f "results/circuit_checkpoint.json" ]; then
        echo ""
        echo "Checkpoint status:"
        python3 -c "
import json
with open('results/circuit_checkpoint.json', 'r') as f:
    data = json.load(f)
    print(f'  Words processed: {len(data.get(\"processed_words\", []))}')
    if 'timestamp' in data:
        print(f'  Last updated: {data[\"timestamp\"]}')
"
    fi
}
trap error_handler ERR

# Determine words file
WORDS_FILE="${WORDS_FILE:-sampled_words.json}"
if [ ! -f "$WORDS_FILE" ]; then
    echo "Error: Words file not found: $WORDS_FILE"
    exit 1
fi

# Count total words
TOTAL_WORDS=$(python3 -c "import json; print(len(json.load(open('$WORDS_FILE'))))")
echo "Total words to process: $TOTAL_WORDS"

# Check for test mode
if [ "$NUM_WORDS" -gt 0 ] && [ "$NUM_WORDS" -lt "$TOTAL_WORDS" ]; then
    echo "Test mode: Processing first $NUM_WORDS words only"
    TEST_FLAG="--test-words $NUM_WORDS"
else
    TEST_FLAG=""
fi

# Create output directory
OUTPUT_DIR="results_${JOB_NAME}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model: google/gemma-2-2b (will download if needed)"
echo "  Words file: $WORDS_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "  Resume from checkpoint: yes"
echo ""

# Function to show progress
show_progress() {
    if [ -f "$OUTPUT_DIR/circuit_checkpoint.json" ]; then
        python3 -c "
import json
import os
from datetime import datetime

checkpoint_file = '$OUTPUT_DIR/circuit_checkpoint.json'
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
        processed = len(data.get('processed_words', []))
        total = $TOTAL_WORDS
        percent = (processed / total) * 100 if total > 0 else 0
        print(f'Progress: {processed}/{total} words ({percent:.1f}%)')
        
        # Estimate time remaining
        if 'timestamp' in data and processed > 0:
            start_time = datetime.fromisoformat(data['timestamp'])
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            print(f'Estimated time remaining: {hours}h {minutes}m')
"
    fi
}

# Run extraction with periodic progress updates
echo "Starting extraction..."
echo "================================"

# Run in background with progress monitoring
python extract_features.py \
    --model google/gemma-2-2b \
    --words-file "$WORDS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --checkpoint-frequency "$CHECKPOINT_INTERVAL" \
    --resume \
    $TEST_FLAG \
    2>&1 | tee "$LOG_DIR/extraction.log" &

PYTHON_PID=$!

# Monitor progress while running
while kill -0 $PYTHON_PID 2>/dev/null; do
    sleep 60  # Check every minute
    echo ""
    echo "Status update at $(date):"
    show_progress
    
    # Check GPU usage if available
    if [ "$DEVICE" = "cuda" ]; then
        echo "GPU Memory:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "  %d MB / %d MB (%.1f%%)\n", $1, $3, ($1/$3)*100}'
    fi
done

# Wait for completion
wait $PYTHON_PID
EXIT_CODE=$?

echo ""
echo "================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Job Completed Successfully: $(date) ==="
    
    # Generate summary report
    echo ""
    echo "Generating summary report..."
    
    python3 << PYTHON_EOF
import pandas as pd
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
log_dir = Path('$LOG_DIR')

# Load results
features_file = output_dir / 'feature_activations.csv'
checkpoint_file = output_dir / 'circuit_checkpoint.json'

if features_file.exists():
    df = pd.read_csv(features_file)
    
    # Calculate statistics
    n_words = df['word'].nunique()
    n_layers = df['layer'].nunique()
    total_features = len(df)
    
    # Per-word statistics
    word_stats = df.groupby('word').agg({
        'feature_id': 'count',
        'layer': 'nunique'
    }).rename(columns={'feature_id': 'total_features', 'layer': 'active_layers'})
    
    # Per-layer statistics
    layer_stats = df.groupby('layer')['feature_id'].count()
    
    # Create summary
    summary = f'''=== Extraction Summary ===
Job: $JOB_NAME
Completed: $(date)

Overall Statistics:
  Words processed: {n_words}
  Total features extracted: {total_features}
  Layers analyzed: {n_layers}
  
Per-Word Statistics:
  Mean features per word: {word_stats['total_features'].mean():.1f}
  Max features: {word_stats['total_features'].max()}
  Min features: {word_stats['total_features'].min()}
  Mean active layers: {word_stats['active_layers'].mean():.1f}

Per-Layer Statistics:
  Mean features per layer: {layer_stats.mean():.1f}
  Most active layer: Layer {layer_stats.idxmax()} ({layer_stats.max()} features)
  Least active layer: Layer {layer_stats.idxmin()} ({layer_stats.min()} features)

Output Files:
  Features: {features_file}
  Checkpoint: {checkpoint_file}
  Logs: {log_dir}
'''
    
    print(summary)
    
    # Save summary
    with open(log_dir / 'summary.txt', 'w') as f:
        f.write(summary)
    
    # Create a quick visualization of layer activity
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    layer_stats.plot(kind='bar')
    plt.title('Feature Activations by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Number of Features')
    plt.tight_layout()
    plt.savefig(log_dir / 'layer_activity.png')
    print(f"\nVisualization saved to {log_dir / 'layer_activity.png'}")
    
else:
    print("No results file found")
PYTHON_EOF
    
    echo ""
    echo "✅ Extraction complete!"
    echo "Results saved to: $OUTPUT_DIR"
    
else
    echo "=== Job Failed with exit code $EXIT_CODE ==="
    echo "Check logs for details: $LOG_DIR/extraction.log"
    echo ""
    echo "To resume from checkpoint:"
    echo "  ./run_extraction_job.sh $JOB_NAME"
fi

EOF

# Make job script executable
chmod +x "$LOG_DIR/job_script.sh"

# Export variables for job script
export NUM_WORDS LOG_DIR JOB_NAME CHECKPOINT_INTERVAL

# Start tmux session with job
tmux new-session -d -s "$SESSION_NAME" -n "extraction"
tmux send-keys -t "$SESSION_NAME:extraction" "cd $(pwd) && bash $LOG_DIR/job_script.sh" Enter

# Create second window for monitoring
tmux new-window -t "$SESSION_NAME" -n "monitor"
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 10 'echo \"=== Extraction Progress ===\"; tail -20 $LOG_DIR/extraction.log 2>/dev/null | grep -E \"Processing|word|feature|Progress\" | tail -10; echo; echo \"=== GPU Status ===\"; nvidia-smi 2>/dev/null | head -10 || echo \"No GPU\"'" Enter

# Create third window for log tail
tmux new-window -t "$SESSION_NAME" -n "logs"
tmux send-keys -t "$SESSION_NAME:logs" "tail -f $LOG_DIR/extraction.log" Enter

# Create monitoring script
cat > "$LOG_DIR/monitor.sh" << EOF
#!/bin/bash
# Monitor extraction progress

echo "=== Circuit Feature Extraction Monitor ==="
echo "Session: $SESSION_NAME"
echo "Logs: $LOG_DIR"
echo ""

# Check if session is running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✅ Job is running"
    echo ""
    
    # Show progress if checkpoint exists
    if [ -f "results_${JOB_NAME}/circuit_checkpoint.json" ]; then
        python3 -c "
import json
with open('results_${JOB_NAME}/circuit_checkpoint.json', 'r') as f:
    data = json.load(f)
    processed = len(data.get('processed_words', []))
    print(f'Words processed: {processed}')
"
    fi
    
    # Show recent activity
    echo ""
    echo "Recent activity:"
    tail -5 "$LOG_DIR/extraction.log" 2>/dev/null | grep -v "^$"
    
else
    echo "❌ Job not running"
    
    # Check if completed
    if [ -f "$LOG_DIR/summary.txt" ]; then
        echo "Job completed. See summary:"
        cat "$LOG_DIR/summary.txt"
    fi
fi

echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME         # Attach to session"
echo "  tmux select-window -t monitor        # View monitor"
echo "  tmux select-window -t logs           # View live logs"
echo "  tmux kill-session -t $SESSION_NAME   # Kill job"
EOF

chmod +x "$LOG_DIR/monitor.sh"

# Show success message
echo -e "${GREEN}=== Job Started Successfully ===${NC}"
echo ""
echo "Session: $SESSION_NAME"
echo "Logs: $LOG_DIR/"
echo ""
echo "To monitor:"
echo "  tmux attach -t $SESSION_NAME         # See live output (3 windows)"
echo "  ./$LOG_DIR/monitor.sh                # Quick status check"
echo "  tail -f $LOG_DIR/extraction.log      # Follow log file"
echo ""
echo "Tmux windows:"
echo "  0: extraction - Main extraction process"
echo "  1: monitor    - Progress and GPU monitoring"
echo "  2: logs       - Live log tail"
echo ""
echo "Navigation:"
echo "  Ctrl+B, 0/1/2  - Switch windows"
echo "  Ctrl+B, d      - Detach (job continues)"
echo "  Ctrl+B, [      - Scroll mode (q to exit)"
echo ""
echo -e "${YELLOW}The job will continue running even if you disconnect!${NC}"
echo ""
echo "Estimated time (A100 GPU):"
echo "  - 100 words: ~15 minutes"
echo "  - 1000 words: ~2 hours"  
echo "  - 5000 words: ~10 hours"