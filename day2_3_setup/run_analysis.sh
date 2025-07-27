#!/bin/bash

echo "🚀 Day 2-3: Complete Semantic Connectivity Analysis Workflow"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "semantic_connectivity_cli.py" ]; then
    echo "❌ Error: Please run this script from the day2_3_setup directory"
    exit 1
fi

# Step 1: Create polysemy data
echo "📊 Step 1: Creating polysemy data..."
python create_polysemy_data.py --words extreme_contrast_words.json --output polysemy_scores.json

if [ $? -ne 0 ]; then
    echo "❌ Error: Polysemy data creation failed"
    exit 1
fi

echo "✅ Polysemy data created successfully"
echo ""

# Step 2: Run connectivity analysis
echo "🔄 Step 2: Running semantic connectivity analysis..."
echo "This may take 2-4 hours on GPU, 8-12 hours on CPU..."
echo ""

# Check if user wants to continue
read -p "Continue with full analysis? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Analysis cancelled. You can run individual steps manually:"
    echo "  python semantic_connectivity_cli.py --words extreme_contrast_words.json --polysemy-file polysemy_scores.json"
    echo "  python analyze_results.py --results day2_3_connectivity_results.json"
    exit 0
fi

# Run the connectivity analysis
python semantic_connectivity_cli.py \
    --words extreme_contrast_words.json \
    --output day2_3_connectivity_results.json \
    --polysemy-file polysemy_scores.json \
    --threshold 0.7 \
    --sample-size 1000

if [ $? -ne 0 ]; then
    echo "❌ Error: Connectivity analysis failed"
    echo "You can resume with: python semantic_connectivity_cli.py --resume --words extreme_contrast_words.json"
    exit 1
fi

echo "✅ Connectivity analysis completed successfully"
echo ""

# Step 3: Analyze results
echo "📈 Step 3: Analyzing results and generating report..."
python analyze_results.py --results day2_3_connectivity_results.json --output Day2_3_Analysis.md

if [ $? -ne 0 ]; then
    echo "❌ Error: Results analysis failed"
    exit 1
fi

echo "✅ Results analysis completed successfully"
echo ""

# Summary
echo "🎉 Day 2-3 Analysis Complete!"
echo "============================================================"
echo "Generated files:"
echo "  📊 polysemy_scores.json - Polysemy data for correlation"
echo "  📈 day2_3_connectivity_results.json - Complete connectivity results"
echo "  📄 Day2_3_Analysis.md - Comprehensive analysis document"
echo ""
echo "Next steps:"
echo "  1. Review Day2_3_Analysis.md for key findings"
echo "  2. Check hypothesis validation results"
echo "  3. Prepare for Day 4 circuit complexity analysis"
echo ""
echo "🚀 Ready for Day 4!" 