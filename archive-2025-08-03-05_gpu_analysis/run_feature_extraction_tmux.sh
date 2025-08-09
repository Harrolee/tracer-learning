#!/bin/bash
# Script to run feature extraction in tmux with checkpointing and monitoring
# Saves results in a portable format for analysis on non-GPU machines

set -e

# Configuration
RESULTS_DIR="${1:-results_full_analysis_5k_5000words}"
JOB_NAME="${2:-feature_extraction}"
BATCH_SIZE="${3:-100}"  # Process in batches for checkpointing
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/tracer-learning/models/gemma-2b}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Feature Extraction with Checkpointing ===${NC}"
echo "Results directory: $RESULTS_DIR"
echo "Job name: $JOB_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"

# Check prerequisites
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory not found at $RESULTS_DIR${NC}"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/sampled_words.json" ]; then
    echo -e "${RED}Error: No sampled_words.json found${NC}"
    exit 1
fi

# Create directories
LOG_DIR="logs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="$RESULTS_DIR/checkpoints"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Create job script
JOB_SCRIPT="$LOG_DIR/job_script.sh"
cat > "$JOB_SCRIPT" << 'EOF'
#!/bin/bash
# Auto-generated job script for feature extraction

echo "Starting feature extraction at $(date)"
echo "================================"

# Set environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Function to run extraction with automatic resume
run_extraction() {
    echo "Running feature extraction..."
    
    python extract_circuit_features_standalone.py \
        --model "$MODEL_PATH" \
        --results-dir "$RESULTS_DIR" \
        --device "$DEVICE" \
        --checkpoint-frequency "$BATCH_SIZE" \
        --resume \
        2>&1 | tee -a "$LOG_DIR/extraction.log"
    
    return ${PIPESTATUS[0]}
}

# Main execution loop with retries
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES"
    
    if run_extraction; then
        echo "✨ Extraction completed successfully!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "⚠️  Extraction failed, will retry in 30 seconds..."
            sleep 30
        else
            echo "❌ Extraction failed after $MAX_RETRIES attempts"
            exit 1
        fi
    fi
done

# Post-processing: Create summary files for non-GPU analysis
echo "Creating summary files..."

python << 'PYTHON_EOF'
import json
import csv
import pandas as pd
from pathlib import Path
import numpy as np

results_dir = Path("$RESULTS_DIR")
checkpoint_dir = results_dir / "checkpoints"

# Load checkpoint if exists
checkpoint_file = results_dir / "circuit_checkpoint.json"
if checkpoint_file.exists():
    print("Loading from checkpoint...")
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
        features_by_word = checkpoint_data['features']
else:
    # Try to reconstruct from CSV
    print("Loading from CSV...")
    features_csv = results_dir / "feature_activations.csv"
    if features_csv.exists():
        df = pd.read_csv(features_csv)
        features_by_word = {}
        for word in df['word'].unique():
            word_df = df[df['word'] == word]
            features_by_word[word] = {}
            for _, row in word_df.iterrows():
                layer = int(row['layer'])
                if layer not in features_by_word[word]:
                    features_by_word[word][layer] = []
                features_by_word[word][layer].append({
                    'feature_id': int(row['feature_id']),
                    'activation_strength': float(row['activation_strength'])
                })

# Create layer-wise summary
print("Creating layer-wise summary...")
layer_summary = []

# Get max layers
max_layers = 0
for word, layers in features_by_word.items():
    if layers:
        max_layers = max(max_layers, max(int(l) for l in layers.keys()))

# Summarize features per layer per word
for word, layers in features_by_word.items():
    for layer_idx in range(max_layers + 1):
        features = layers.get(str(layer_idx), layers.get(layer_idx, []))
        layer_summary.append({
            'word': word,
            'layer': layer_idx,
            'active_features_count': len(features),
            'unique_feature_ids': ','.join(str(f['feature_id']) for f in features[:10])  # First 10
        })

# Save layer summary
layer_summary_file = results_dir / "layer_feature_summary.csv"
df_summary = pd.DataFrame(layer_summary)
df_summary.to_csv(layer_summary_file, index=False)
print(f"Saved layer summary to {layer_summary_file}")

# Create word-level summary
word_summary = []
for word, layers in features_by_word.items():
    total_features = sum(len(features) for features in layers.values())
    active_layers = len([l for l in layers if len(layers[l]) > 0])
    
    # Get feature distribution across layers
    feature_counts = [len(layers.get(str(i), layers.get(i, []))) for i in range(max_layers + 1)]
    
    word_summary.append({
        'word': word,
        'total_active_features': total_features,
        'active_layers': active_layers,
        'mean_features_per_layer': np.mean(feature_counts),
        'std_features_per_layer': np.std(feature_counts),
        'max_features_in_layer': max(feature_counts),
        'min_features_in_layer': min(feature_counts)
    })

# Save word summary
word_feature_summary_file = results_dir / "word_feature_summary.csv"
df_word = pd.DataFrame(word_summary)
df_word.to_csv(word_feature_summary_file, index=False)
print(f"Saved word summary to {word_feature_summary_file}")

# Create combined analysis file
# Load semantic connectivity data
connectivity_file = results_dir / "layer_connectivity.csv"
if connectivity_file.exists():
    print("Merging with connectivity data...")
    df_conn = pd.read_csv(connectivity_file)
    
    # Merge connectivity and feature data
    merged_data = []
    for _, conn_row in df_conn.iterrows():
        word = conn_row['word']
        layer = conn_row['layer']
        
        # Get feature count for this word-layer
        feature_count = 0
        if word in features_by_word:
            features = features_by_word[word].get(str(layer), features_by_word[word].get(layer, []))
            feature_count = len(features)
        
        merged_data.append({
            'word': word,
            'layer': layer,
            'connectivity_count': conn_row['connectivity_count'],
            'mean_similarity': conn_row['mean_similarity'],
            'active_features_count': feature_count
        })
    
    # Save merged analysis
    merged_file = results_dir / "connectivity_features_merged.csv"
    df_merged = pd.DataFrame(merged_data)
    df_merged.to_csv(merged_file, index=False)
    print(f"Saved merged analysis to {merged_file}")
    
    # Quick correlation check
    if len(df_merged) > 0:
        corr = df_merged['connectivity_count'].corr(df_merged['active_features_count'])
        print(f"\nQuick check - Correlation between connectivity and features: {corr:.3f}")

print("\n✅ Summary files created successfully!")
print("\nFiles created for non-GPU analysis:")
print(f"  - {layer_summary_file}")
print(f"  - {word_feature_summary_file}")
if connectivity_file.exists():
    print(f"  - {merged_file}")

PYTHON_EOF

echo "================================"
echo "Feature extraction completed at $(date)"

# Final summary
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo "📁 Summary files created:"
echo "  - layer_feature_summary.csv: Features per layer for each word"
echo "  - word_feature_summary.csv: Overall feature statistics per word"
echo "  - connectivity_features_merged.csv: Combined connectivity + features"
echo ""
echo "These files can be analyzed on any machine without GPU!"
EOF

# Replace variables in the job script
sed -i "s|\$MODEL_PATH|$MODEL_PATH|g" "$JOB_SCRIPT"
sed -i "s|\$RESULTS_DIR|$RESULTS_DIR|g" "$JOB_SCRIPT"
sed -i "s|\$DEVICE|$DEVICE|g" "$JOB_SCRIPT"
sed -i "s|\$BATCH_SIZE|$BATCH_SIZE|g" "$JOB_SCRIPT"
sed -i "s|\$LOG_DIR|$LOG_DIR|g" "$JOB_SCRIPT"

chmod +x "$JOB_SCRIPT"

# Create monitoring script
MONITOR_SCRIPT="$LOG_DIR/monitor.sh"
cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
# Monitor progress of feature extraction

while true; do
    clear
    echo "=== Feature Extraction Progress ==="
    echo "Time: \$(date)"
    echo ""
    
    # Check checkpoint progress
    if [ -f "$RESULTS_DIR/circuit_checkpoint.json" ]; then
        echo "Checkpoint status:"
        python3 -c "
import json
with open('$RESULTS_DIR/circuit_checkpoint.json', 'r') as f:
    data = json.load(f)
    print(f'  Words processed: {len(data["processed_words"])}')
"
    fi
    
    # Check log for recent activity
    if [ -f "$LOG_DIR/extraction.log" ]; then
        echo ""
        echo "Recent activity:"
        tail -n 20 "$LOG_DIR/extraction.log" | grep -E "(Processing|Found|Features per layer|Extracting features)" | tail -n 10
    fi
    
    # Check if still running
    if pgrep -f "extract_circuit_features_standalone.py" > /dev/null; then
        echo ""
        echo "✅ Extraction is running..."
    else
        echo ""
        echo "⚠️  Extraction process not found"
    fi
    
    sleep 10
done
EOF

chmod +x "$MONITOR_SCRIPT"

# Create tmux session
TMUX_SESSION="${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}Starting job in tmux session: $TMUX_SESSION${NC}"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Start main job in tmux
tmux new-session -d -s "$TMUX_SESSION" -n "extraction"
tmux send-keys -t "$TMUX_SESSION:extraction" "bash $JOB_SCRIPT" Enter

# Start monitor in second window
tmux new-window -t "$TMUX_SESSION" -n "monitor"
tmux send-keys -t "$TMUX_SESSION:monitor" "bash $MONITOR_SCRIPT" Enter

echo -e "${GREEN}Job started successfully!${NC}"
echo ""
echo "To view extraction:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To see progress:"
echo "  tmux attach -t $TMUX_SESSION:monitor"
echo ""
echo "To check logs:"
echo "  tail -f $LOG_DIR/extraction.log"
echo ""
echo "Window navigation in tmux:"
echo "  - Switch windows: Ctrl+B, then 0/1/2..."
echo "  - Detach: Ctrl+B, then D"
echo ""
echo "The job will:"
echo "  1. Run with automatic checkpointing every $BATCH_SIZE words"
echo "  2. Resume automatically if interrupted"
echo "  3. Create summary CSV files for non-GPU analysis"
echo "  4. Merge with connectivity data for correlation analysis"