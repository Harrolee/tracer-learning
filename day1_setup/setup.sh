#!/bin/bash

echo "🚀 Day 1 Setup Script"
echo "Research Plan: Semantic Connectivity vs Circuit Complexity"
echo "============================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the day1_setup directory."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv day1_env
    source day1_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Download NLTK data
echo "Setting up NLTK data..."
python3 -c "
import nltk
try:
    nltk.data.find('corpora/wordnet')
    print('✅ WordNet already available')
except LookupError:
    print('Downloading WordNet...')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print('✅ WordNet downloaded')
"

# Create results directory
mkdir -p results
echo "✅ Results directory created"

# Test basic imports
echo "Testing imports..."
python3 -c "
try:
    import torch
    import transformers
    import nltk
    from nltk.corpus import wordnet
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run full Day 1 execution:"
echo "   python day1_main.py"
echo ""
echo "2. Or test vocabulary sampling only:"
echo "   python day1_main.py --skip-model"
echo ""
echo "3. Check README.md for detailed usage instructions"

if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo ""
    echo "📝 Note: Virtual environment is activated. To reactivate later:"
    echo "   source day1_env/bin/activate"
fi 