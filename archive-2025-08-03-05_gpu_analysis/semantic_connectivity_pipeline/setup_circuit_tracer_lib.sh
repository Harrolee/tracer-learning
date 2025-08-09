#!/bin/bash
# Setup circuit-tracer as a Python library for unified analysis pipeline
# This ensures circuit_tracer can be imported as a module

set -e

echo "=== Circuit Tracer Library Setup for Analysis Pipeline ==="
echo

# Check if we're in the semantic_connectivity_pipeline directory or parent
if [ -f "unified_analysis_pipeline.py" ]; then
    echo "✓ In semantic_connectivity_pipeline directory"
elif [ -d "semantic_connectivity_pipeline" ]; then
    cd semantic_connectivity_pipeline
    echo "✓ Changed to semantic_connectivity_pipeline directory"
else
    echo "❌ Error: Run this from learningSlice root or semantic_connectivity_pipeline directory"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Found existing venv"
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "✓ Found parent venv"
    source ../venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

echo
echo "Installing circuit-tracer as Python library..."

# Clone circuit-tracer if not present
if [ ! -d "../circuit-tracer" ]; then
    echo "Cloning circuit-tracer repository..."
    cd ..
    git clone https://github.com/safety-research/circuit-tracer.git
    cd -
else
    echo "✓ circuit-tracer repository already exists"
fi

# Install circuit-tracer as editable package
echo "Installing circuit-tracer in development mode..."
pip install -e ../circuit-tracer

# Verify installation
echo
echo "Verifying installation..."
python -c "
try:
    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution import attribute
    print('✅ Circuit tracer imports successful!')
    print('   - ReplacementModel: Available')
    print('   - attribute function: Available')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Test with Gemma model if available
echo
echo "Testing with Gemma model..."
python -c "
import os
import torch
from pathlib import Path

# Find model path
model_paths = [
    '/home/ubuntu/tracer-learning/models/gemma-2b',
    '/Users/lee/fun/learningSlice/models/gemma-2b',
    '../models/gemma-2b',
    'models/gemma-2b'
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path:
    print(f'Found model at: {model_path}')
    
    try:
        from transformers import AutoModel, AutoTokenizer
        from circuit_tracer import ReplacementModel
        
        print('Loading model and tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        print('Creating ReplacementModel...')
        tracer_model = ReplacementModel(model, tokenizer)
        
        print('✅ Circuit tracer successfully initialized with Gemma!')
        
    except Exception as e:
        print(f'⚠️  Could not initialize with Gemma: {e}')
        print('   This might be normal if the model architecture is not supported')
else:
    print('⚠️  Gemma model not found, skipping model test')
"

echo
echo "=== Setup Complete ==="
echo
echo "Circuit tracer is now available as a Python library."
echo "You can use it in unified_analysis_pipeline.py with:"
echo "  from circuit_tracer import ReplacementModel"
echo "  from circuit_tracer.attribution import attribute"
echo
echo "To run the unified analysis:"
echo "  python unified_analysis_pipeline.py --model <path> --dictionary-embeddings <path> --output-dir results"