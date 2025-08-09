#!/bin/bash
# Quick test with 10 sample words to verify setup

source venv/bin/activate

echo "Running quick test with 10 words..."

python extract_features.py \
    --model google/gemma-2-2b \
    --words-file test_words.json \
    --output-dir test_results \
    --device cuda \
    --top-k-features 5

echo ""
echo "Test complete. Check test_results/ for output."
echo ""

# Show summary if successful
if [ -f "test_results/feature_activations.csv" ]; then
    echo "Sample results:"
    head -20 test_results/feature_activations.csv
    echo ""
    echo "Word count:"
    tail -n +2 test_results/feature_activations.csv | cut -d',' -f1 | sort -u | wc -l
    echo ""
    echo "✅ Test successful! Ready to run full extraction."
else
    echo "❌ No output generated. Check for errors above."
fi