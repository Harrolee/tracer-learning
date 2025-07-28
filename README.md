# Semantic Connectivity Evolution vs Circuit Complexity

This repository contains a unified pipeline for analyzing how word representations evolve through transformer layers and correlating these patterns with circuit complexity.

## Research Question

How does semantic connectivity evolve across model layers, and do these evolution patterns predict circuit complexity better than single-layer measurements?

## Theory

### Core Hypothesis

**Primary**: Words with dynamic connectivity patterns (high variance across layers) participate in more complex circuits than words with stable patterns.

**Secondary**: Connectivity evolution patterns reveal different computational strategies:
- **Early peak** → Surface-level processing (orthographic/syntactic)
- **Middle peak** → Semantic disambiguation
- **Late peak** → Task-specific grouping
- **High variance** → Multi-faceted processing requiring complex circuitry

### Key Insight

Traditional approaches measure word embeddings at a single layer (typically the final layer). However, information processing in transformers is inherently multi-layered. By tracking how a word's semantic neighborhood changes through layers, we can better understand:

1. **Processing complexity**: Words requiring more computational steps show higher variance
2. **Processing type**: Peak connectivity layer indicates primary processing level
3. **Circuit requirements**: Dynamic patterns correlate with cross-layer feature activation

## Method

### 1. Vocabulary Sampling

We sample words from WordNet based on polysemy (number of meanings):
- **Extreme contrast**: 2,500 high-polysemy + 2,500 monosemous words
- **Stratified**: Even distribution across polysemy levels

Polysemy provides a principled baseline - we expect high-polysemy words to require more complex processing.

### 2. Layer-wise Connectivity Analysis

For each word at each layer:
1. Extract embedding from that layer's hidden states
2. Compare against entire English dictionary (precomputed)
3. Count neighbors above similarity threshold
4. Track evolution across layers

### 3. Circuit Feature Extraction

Using circuit-tracer, we identify:
- Which features activate for each word
- At which layers these features fire
- Total unique features across all layers

### 4. Correlation Analysis

We test whether connectivity evolution metrics predict circuit complexity:
- **Variance**: How much connectivity changes across layers
- **Peak layer**: Where connectivity is highest
- **Stability**: Inverse of variance (consistent vs dynamic)

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd learningSlice

# Run setup script (creates venv, installs all dependencies)
./setup_pipeline.sh

# Activate environment
source activate_pipeline.sh

# Test installation
python test_installation.py
```

## Dependencies

The pipeline requires:
- Python 3.8+
- PyTorch 2.0+ (with CUDA/MPS support recommended)
- Transformers 4.30+
- NLTK with WordNet data
- Circuit-tracer (for feature extraction)
- NumPy, Pandas, tqdm

All dependencies are automatically installed by `setup_pipeline.sh`.

## Usage

### Step 1: Precompute Dictionary Embeddings (One-time)

```bash
python precompute_dictionary_embeddings.py \
    --model /path/to/model \
    --output-dir dictionary_embeddings \
    --device cuda \
    --batch-size 64 \
    --layers 0 4 9 13 18

# For Gemma-2B on MPS:
python precompute_dictionary_embeddings.py \
    --model /Users/lee/fun/learningSlice/models/gemma-2b \
    --output-dir dictionary_embeddings \
    --device mps \
    --batch-size 32
```

This creates ~1-2GB of embeddings per layer. Takes several hours but only needs to be done once per model.

### Step 2: Run Analysis Pipeline

```bash
python unified_analysis_pipeline.py \
    --model /path/to/model \
    --dictionary-embeddings dictionary_embeddings \
    --output-dir results \
    --sampling-strategy extreme_contrast \
    --num-words 5000 \
    --device cuda

# For Gemma-2B analysis:
python unified_analysis_pipeline.py \
    --model /Users/lee/fun/learningSlice/models/gemma-2b \
    --dictionary-embeddings dictionary_embeddings \
    --output-dir results_5k_extreme \
    --sampling-strategy extreme_contrast \
    --num-words 5000 \
    --device mps
```

### Step 3: Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
summary = pd.read_csv('results_5k_extreme/word_summary.csv')
connectivity = pd.read_csv('results_5k_extreme/layer_connectivity.csv')
features = pd.read_csv('results_5k_extreme/feature_activations.csv')

# Key correlation: Does connectivity predict features?
print(summary.corr()['total_connectivity']['total_features'])

# Compute variance for each word
trajectories = pd.read_csv('results_5k_extreme/connectivity_trajectories.csv')
trajectory_cols = [c for c in trajectories.columns if 'layer_' in c]
trajectories['variance'] = trajectories[trajectory_cols].var(axis=1)

# Merge with features
merged = trajectories[['word', 'variance']].merge(
    summary[['word', 'total_features']], on='word'
)

# Test main hypothesis
correlation = merged['variance'].corr(merged['total_features'])
print(f"Connectivity variance vs Feature count: r={correlation:.3f}")
```

## Output Files

The pipeline generates four CSV files:

1. **word_summary.csv**
   - `word`: The analyzed word
   - `polysemy_score`: Number of WordNet meanings
   - `total_features`: Sum of activated features across layers
   - `total_connectivity`: Sum of connections across layers

2. **layer_connectivity.csv**
   - `word`, `layer`: Word and layer index
   - `connectivity_count`: Number of similar words
   - `mean_similarity`: Average similarity to all words
   - `max_similarity`: Highest similarity score

3. **feature_activations.csv**
   - `word`, `layer`: Word and layer index
   - `feature_id`: Unique feature identifier
   - `activation_strength`: How strongly the feature fired

4. **connectivity_trajectories.csv**
   - `word`: The analyzed word
   - `layer_N_connectivity`: Connectivity at each layer (wide format)

## Expected Results

### Primary Predictions
1. Connectivity variance correlates with circuit complexity (r > 0.7)
2. Single-layer measurements show weaker correlation (r ~ 0.5)

### Evolution Patterns
- **Early peakers**: Simple circuits, syntactic processing
- **Late peakers**: Complex circuits, semantic processing
- **High variance**: Distributed circuits across many layers
- **Stable patterns**: Minimal circuitry, consistent processing

### Polysemy Effects
- High-polysemy words → Higher variance → More features
- Monosemous words → Lower variance → Fewer features

## Project Structure

```
learningSlice/
├── precompute_dictionary_embeddings.py  # One-time dictionary computation
├── unified_analysis_pipeline.py         # Main analysis script
├── models/                              # Local model storage
│   └── gemma-2b/                       # Gemma model files
├── dictionary_embeddings/               # Precomputed embeddings
│   ├── embeddings_layer_0.pkl
│   ├── embeddings_layer_4.pkl
│   └── ...
├── results_5k_extreme/                  # Analysis outputs
│   ├── word_summary.csv
│   ├── layer_connectivity.csv
│   ├── feature_activations.csv
│   └── connectivity_trajectories.csv
└── day*_setup/                         # Development history
```

## Complete Workflow for Collaborators

```bash
# 1. Clone and setup (5 minutes)
git clone <repo-url>
cd learningSlice
./setup_pipeline.sh
source activate_pipeline.sh

# 2. Run example analysis (10 minutes)
export MODEL_PATH=/path/to/your/model  # e.g., models/gemma-2b
./run_example_analysis.sh

# 3. View results
ls results_demo_100/
# Contains: word_summary.csv, layer_connectivity.csv, feature_activations.csv, etc.

# 4. Full analysis (6-8 hours total)
# First time only: precompute full dictionary (~4-6 hours)
python precompute_dictionary_embeddings.py \
    --model $MODEL_PATH \
    --output-dir dictionary_embeddings \
    --device cuda

# Run full 5k word analysis (~30-60 minutes)  
python unified_analysis_pipeline.py \
    --model $MODEL_PATH \
    --dictionary-embeddings dictionary_embeddings \
    --output-dir results_5k_full \
    --num-words 5000 \
    --device cuda
```

## Key Findings

From initial experiments:
- Connectivity drops dramatically after the embedding layer
- This suggests rapid specialization from generic tokens to task-specific representations
- Words with highest polysemy tend to maintain connectivity longer
- Feature activation patterns align with connectivity peaks

## Troubleshooting

**Circuit-tracer import fails**: 
- The pipeline will still run but skip feature extraction
- To fix: `pip install git+https://github.com/safety-research/circuit-tracer.git`

**Out of memory**:
- Reduce batch size: `--batch-size 16`
- Use CPU instead of GPU: `--device cpu`
- Reduce number of words: `--num-words 1000`

**NLTK data not found**:
- Run: `python -c "import nltk; nltk.download('wordnet')"`

## Citation

If you use this code for research, please cite:
```
@software{semantic_connectivity_2024,
  title={Semantic Connectivity Evolution vs Circuit Complexity},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/learningSlice}
}
```

## License

MIT License - see LICENSE file for details.