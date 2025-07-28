# Project Notes for Claude

## Model Locations
- Gemma 2B model is stored locally at: `/Users/lee/fun/learningSlice/models/gemma-2b`
  - Use this path instead of downloading from HuggingFace
  - The model includes all necessary files (safetensors, tokenizer, config)

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

## Development History
- `day1_setup/`: Polysemy-based word sampling
- `day2_3_setup/`: Connectivity analysis development
- `day4_setup/`: Circuit tracer demos
- `day5_setup/`: Circuit complexity analysis
- All integrated into root-level unified pipeline