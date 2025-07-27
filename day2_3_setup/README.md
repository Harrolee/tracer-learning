# Day 2-3: Semantic Connectivity Analysis

This directory contains the implementation for **Days 2-3** of the research plan: "Polysemy-Based Semantic Connectivity vs Circuit Complexity".

## Research Goal
- Full semantic connectivity analysis on 5,000 polysemy-based word sample from Day 1
- Polysemy-connectivity correlation analysis  
- Identification of connectivity outliers for circuit complexity analysis
- Statistical validation of polysemy-connectivity hypothesis

## Files Overview

- `semantic_connectivity_cli.py` - Enhanced CLI utility for connectivity analysis (NLTK removed, JSON input)
- `create_polysemy_data.py` - Extract polysemy scores for correlation analysis
- `extreme_contrast_words.json` - 5,000 words from Day 1 extreme contrast strategy
- `requirements.txt` - Python dependencies (NLTK removed)
- `README.md` - This file

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data Files

```bash
# Create polysemy scores for correlation analysis
python create_polysemy_data.py --words extreme_contrast_words.json --output polysemy_scores.json
```

### 3. Run Semantic Connectivity Analysis

```bash
# Full analysis with Gemma2 2B (requires GPU)
python semantic_connectivity_cli.py \
    --words extreme_contrast_words.json \
    --output day2_3_connectivity_results.json \
    --polysemy-file polysemy_scores.json \
    --threshold 0.7 \
    --sample-size 1000

# CPU mode (slower)
python semantic_connectivity_cli.py \
    --words extreme_contrast_words.json \
    --device cpu \
    --output day2_3_connectivity_results.json \
    --polysemy-file polysemy_scores.json

# Resume from checkpoint if interrupted
python semantic_connectivity_cli.py \
    --words extreme_contrast_words.json \
    --resume \
    --output day2_3_connectivity_results.json \
    --polysemy-file polysemy_scores.json
```

## Expected Outputs

After successful execution, you'll find:
- `day2_3_connectivity_results.json` - Complete connectivity analysis results
- `polysemy_scores.json` - Polysemy scores for correlation analysis  
- `semantic_connectivity_checkpoint.json` - Progress checkpoint (deleted after completion)

## Analysis Features

### Enhanced CLI Utility
**Removed from collaborators' version:**
- ❌ NLTK WordNet integration
- ❌ Alphabetical vocabulary sampling

**Added for polysemy research:**
- ✅ JSON word list input
- ✅ Polysemy-connectivity correlation analysis
- ✅ Enhanced progress tracking and checkpointing
- ✅ Detailed statistical summaries
- ✅ Outlier identification optimized for circuit complexity selection

### Connectivity Measurement
- **Method**: Cosine similarity between Gemma2 2B embeddings
- **Sampling**: 1,000 random comparisons per word (configurable)
- **Threshold**: 0.7 similarity threshold for connection (configurable)
- **Output**: Integer count of high-similarity neighbors

### Polysemy Integration
- **Source**: WordNet synset counts from Day 1 analysis
- **Categories**: Monosemous (1), Low (2-3), Medium (4-10), High (11+)
- **Analysis**: Mean connectivity by polysemy level
- **Correlation**: Statistical relationship testing

## System Requirements

### Memory Requirements
- **GPU**: 8GB+ VRAM for Gemma2 2B (recommended)
- **CPU**: 16GB+ RAM if using CPU mode
- **Storage**: 2GB+ for model cache and results

### Performance Expectations
- **GPU Mode**: ~2-4 hours for 5,000 words
- **CPU Mode**: ~8-12 hours for 5,000 words  
- **Checkpoint**: Progress saved every 100 words

## Command Line Options

```bash
python semantic_connectivity_cli.py --help
```

**Key Arguments:**
- `--words` - JSON file with word list (required)
- `--model` - Model name (default: google/gemma-2-2b)
- `--threshold` - Similarity threshold (default: 0.7)
- `--sample-size` - Comparisons per word (default: 1000)
- `--polysemy-file` - JSON file with polysemy scores
- `--resume` - Resume from checkpoint
- `--device` - cuda/cpu (auto-detected)

## Expected Results

### Connectivity Predictions
Based on Day 1 polysemy analysis:

| Polysemy Level | Expected Mean Connectivity | Word Count in Sample |
|----------------|---------------------------|---------------------|
| **High** (11+ senses) | 15-30 connections | ~85 words |
| **Medium** (4-10 senses) | 10-20 connections | ~675 words |
| **Low** (2-3 senses) | 5-15 connections | ~1,740 words |
| **Monosemous** (1 sense) | 3-8 connections | ~2,500 words |

### Statistical Hypotheses
1. **Primary**: Significant positive correlation between polysemy and connectivity (r > 0.4, p < 0.05)
2. **Group Differences**: High-polysemy words have significantly higher connectivity than monosemous words
3. **Effect Size**: Meaningful difference (Cohen's d > 0.5) between polysemy groups

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use CPU mode
python semantic_connectivity_cli.py --device cpu --words extreme_contrast_words.json
```

**2. Model Download Issues**
- Ensure stable internet connection
- Check Hugging Face Hub status
- May require authentication: `huggingface-cli login`

**3. Interrupted Analysis**
```bash
# Resume from checkpoint
python semantic_connectivity_cli.py --resume --words extreme_contrast_words.json
```

**4. Missing Dependencies**
```bash
pip install -r requirements.txt
```

### Performance Tips

- **Use GPU**: 3-4x faster than CPU mode
- **Adjust sample-size**: Lower values = faster but less accurate
- **Monitor memory**: Watch GPU/RAM usage during execution
- **Use checkpoints**: Don't restart from scratch if interrupted

## Integration with Day 1

### Data Flow
```
Day 1: Polysemy Analysis (77k words) 
  ↓
Day 1: Extreme Contrast Sampling (5k words)
  ↓
Day 2-3: JSON Conversion (extreme_contrast_words.json)
  ↓  
Day 2-3: Connectivity Analysis (semantic_connectivity_cli.py)
  ↓
Day 2-3: Results & Outliers (day2_3_connectivity_results.json)
```

### Key Connections
- **Word Set**: Same 5,000 words from Day 1 extreme contrast strategy
- **Polysemy Data**: Extracted from Day 1 WordNet analysis
- **Strategy**: Maintains 50% monosemous + 50% high-polysemy design
- **Hypothesis**: Tests Day 1 predictions about polysemy-connectivity relationship

## Next Steps: Day 4 Preparation

After successful Day 2-3 completion:
1. **Analyze Results**: Statistical correlation between polysemy and connectivity
2. **Select Final Set**: Top 50 + Bottom 50 + Random 100 = 200 words for circuit analysis
3. **Validate Hypothesis**: Confirm polysemy predicts connectivity
4. **Prepare Circuit Analysis**: Optimize word selection for maximum effect size

---

*Implementation by: Research Team*  
*Integration with: Day 1 Polysemy Analysis*  
*Next: Day 4 Circuit Complexity Setup* 