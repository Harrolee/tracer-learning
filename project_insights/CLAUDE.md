# Project Notes for Claude

## Model Locations
- Gemma 2B model is stored locally at: `/Users/lee/fun/learningSlice/models/gemma-2b`
  - Use this path instead of downloading from HuggingFace
  - The model includes all necessary files (safetensors, tokenizer, config)

## Existing Analysis Data
### Semantic Connectivity Analysis (COMPLETED)
- **Location**: `/Users/lee/fun/learningSlice/results_full_analysis_5k_5000words/`
- **Contents**:
  - `layer_connectivity.csv` - Connectivity counts per word per layer (5000 words × 27 layers)
  - `connectivity_trajectories.csv` - Evolution patterns across layers
  - `word_summary.csv` - Word-level statistics including polysemy scores
  - `sampled_words.json` - The 5000 words analyzed
  - `feature_activations.csv` - Currently empty, needs GPU extraction

### Pre-computed Embeddings (COMPLETED)
- **Location**: `/Users/lee/fun/learningSlice/all-embeddings/`
- **Model**: Gemma-2-2b embeddings for entire vocabulary
- **Contents**:
  - `embeddings_layer_0.pkl` through `embeddings_layer_26.pkl` - All 27 layers
  - `embedding_metadata.json` - Metadata about the embeddings
- **Note**: These are the embeddings used for the semantic connectivity analysis

## Main Pipeline (Root Directory)
The project has been unified into two main scripts in the root:

1. **precompute_dictionary_embeddings.py** - One-time computation of dictionary embeddings
2. **unified_analysis_pipeline.py** - Main analysis combining connectivity + circuit features

## Key Commands
```bash
# One-time setup (takes hours)
python precompute_dictionary_embeddings.py \
    --model /Users/lee/fun/learningSlice/models/gemma-2b \
    --output-dir dictionary_embeddings \
    --device mps

# Run full analysis
python unified_analysis_pipeline.py \
    --model /Users/lee/fun/learningSlice/models/gemma-2b \
    --dictionary-embeddings dictionary_embeddings \
    --output-dir results_5k \
    --num-words 5000 \
    --device mps
```

## Research Summary
- **Goal**: Test if semantic connectivity evolution predicts circuit complexity
- **Method**: Track how word neighborhoods change across layers, correlate with feature activation
- **Key Innovation**: Layer-wise analysis instead of single-layer embeddings
- **Current Status**: 
  - Part 1 (Semantic Connectivity): ✅ COMPLETE for 5000 words
  - Part 2 (Circuit Features): 🔄 IN PROGRESS - use `run_feature_extraction_tmux.sh`

## Development History
- `day1_setup/`: Polysemy-based word sampling
- `day2_3_setup/`: Connectivity analysis development
- `day4_setup/`: Circuit tracer demos
- `day5_setup/`: Circuit complexity analysis
- All integrated into root-level unified pipeline