#!/bin/bash
# Quick script to check overnight job status

echo "=== Embedding Job Status Check ==="
echo "Time: $(date)"
echo ""

# Check tmux sessions
echo "Active tmux sessions:"
tmux ls 2>/dev/null || echo "  No tmux sessions found"
echo ""

# Check latest logs
echo "Latest log entries:"
if ls logs/*/precompute.log 1> /dev/null 2>&1; then
    for log in logs/*/precompute.log; do
        echo "  From $log:"
        tail -n 5 "$log" | sed 's/^/    /'
        echo ""
    done
else
    echo "  No log files found"
fi

# Check output files
echo "Output files created:"
for dir in dictionary_embeddings_*; do
    if [ -d "$dir" ]; then
        echo "  $dir:"
        file_count=$(ls -1 "$dir"/*.pkl 2>/dev/null | wc -l)
        echo "    Pickle files: $file_count"
        
        if [ -f "$dir/embedding_metadata.json" ]; then
            echo "    Metadata: ✓"
        fi
        
        # Show latest modified file
        latest=$(ls -t "$dir"/*.pkl 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "    Latest: $(basename $latest) ($(date -r "$latest" '+%H:%M:%S'))"
        fi
    fi
done
echo ""

# Check if Python process is running
echo "Python processes:"
ps aux | grep "[p]ython.*precompute" | head -3
echo ""

# Disk space
echo "Disk usage:"
df -h . | tail -1