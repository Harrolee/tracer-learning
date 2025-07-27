# Day 1: Setup and Vocabulary Sampling

This directory contains the implementation for **Day 1** of the research plan: "Semantic Connectivity vs Circuit Complexity".

## Research Goal
- Setup Gemma2 2B model
- Implement WordNet vocabulary sampling (5,000 words) **with polysemy-based strategies**
- Initial semantic connectivity analysis with polysemy correlation

## Files Overview

- `requirements.txt` - Python dependencies
- `vocab_sampling.py` - WordNet vocabulary sampling with polysemy-based strategies
- `gemma_setup.py` - Gemma2 2B model setup and loading
- `semantic_connectivity.py` - Semantic connectivity measurement
- `day1_main.py` - Main execution script with polysemy analysis
- `strategy_comparison.py` - Compare different polysemy sampling strategies
- `README.md` - This file

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Day 1 Tasks
```bash
# Full execution with extreme contrast strategy (recommended)
python day1_main.py --strategy extreme_contrast

# Compare different strategies first
python day1_main.py --compare-strategies

# Other strategies available
python day1_main.py --strategy balanced
python day1_main.py --strategy high_polysemy

# Test mode (vocabulary only, skip heavy model loading)
python day1_main.py --skip-model --strategy extreme_contrast

# Test with smaller samples
python day1_main.py --test-only --strategy extreme_contrast
```

### 3. Individual Component Testing
```bash
# Test vocabulary sampling with different strategies
python vocab_sampling.py extreme_contrast
python vocab_sampling.py balanced
python vocab_sampling.py high_polysemy

# Compare all strategies
python strategy_comparison.py

# Test model setup only  
python gemma_setup.py

# Test semantic connectivity (requires model)
python semantic_connectivity.py
```

## Expected Outputs

After successful execution, you'll find in the `results/` directory:
- `day1_vocabulary_5k_{strategy}.pkl` - 5,000 WordNet vocabulary words by strategy
- `day1_connectivity_test_{strategy}.pkl` - Initial connectivity analysis results
- `polysemy_comparison.png` - Visualization comparing strategies (if matplotlib available)

## System Requirements

### Memory Requirements
- **CPU**: 16GB+ RAM recommended
- **GPU**: 8GB+ VRAM for full model loading
- **Storage**: 5GB+ for model files

### CUDA Support
- The script automatically detects CUDA availability
- Falls back to CPU if CUDA not available
- Uses 8-bit quantization by default to reduce memory usage

## Implementation Details

### Vocabulary Sampling Strategies
**Three polysemy-based strategies available:**

1. **`extreme_contrast`** (recommended): 2,500 high-polysemy + 2,500 monosemous words
   - Maximizes contrast for hypothesis testing
   - Best statistical power for polysemy-connectivity correlation

2. **`balanced`**: Even sampling across polysemy quartiles
   - Representative of full polysemy spectrum
   - Good for comprehensive correlation analysis

3. **`high_polysemy`**: Focus on top 50% most polysemous words
   - Studies semantically complex words
   - Best for understanding rich semantic networks



### Model Setup
- Loads `google/gemma-2-2b` from Hugging Face
- Automatic device detection (CUDA/CPU)
- 8-bit quantization for memory efficiency
- Built-in model testing and validation

### Semantic Connectivity
- Uses cosine similarity between word embeddings
- Samples vocabulary tokens for comparison
- Configurable similarity threshold (default: 0.7)
- Batch processing for efficiency

## Troubleshooting

### Common Issues

**1. Memory Errors**
```bash
# Use test mode to skip heavy model loading
python day1_main.py --skip-model
```

**2. NLTK Data Missing**
- The script automatically downloads required NLTK data
- Manual download: `python -c "import nltk; nltk.download('wordnet')"`

**3. Hugging Face Authentication**
- Some models may require authentication
- Set up token: `huggingface-cli login`

**4. CUDA Out of Memory**
- Script uses 8-bit quantization by default
- Try CPU mode: the script will automatically fall back

### Performance Tips

- **GPU recommended** for full execution
- **Test mode** (`--skip-model`) for development
- **Restart kernel** between runs to free GPU memory
- **Monitor memory usage** during execution

## Next Steps (Day 2-3)

After successful Day 1 completion:
1. Run full semantic connectivity analysis on all 5,000 words
2. Identify top 50 and bottom 50 connectivity outliers  
3. Select random 100 words from middle range
4. Prepare final 200-word dataset for circuit complexity analysis

## File Structure After Execution

```
day1_setup/
├── vocab_sampling.py
├── gemma_setup.py
├── semantic_connectivity.py
├── day1_main.py
├── strategy_comparison.py
├── requirements.txt
├── README.md
├── results/
│   ├── day1_vocabulary_5k_{strategy}.pkl
│   ├── day1_connectivity_test_{strategy}.pkl
│   └── polysemy_comparison.png
└── wordnet_sample_5k_{strategy}.pkl (cached vocabularies)
``` 