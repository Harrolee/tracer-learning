# Layer-wise Semantic Connectivity Analysis

## Overview
This directory contains a unified pipeline for analyzing semantic connectivity evolution and circuit complexity. The pipeline:
1. Samples words from WordNet based on polysemy
2. Computes layer-wise connectivity using precomputed dictionary embeddings
3. Extracts circuit features using circuit-tracer
4. Exports everything to analyzable CSV format

## Workflow

### Step 1: One-time Dictionary Embedding (Run Once)
```bash
# Precompute embeddings for entire English dictionary
python precompute_dictionary_embeddings.py \
    --model /Users/lee/fun/learningSlice/models/gemma-2b \
    --output-dir dictionary_embeddings \
    --device cuda \
    --batch-size 64

# This creates ~1-2GB per layer of embeddings
# Takes several hours but only needs to be done once
```

### Step 2: Run Unified Analysis Pipeline
```bash
# Run full analysis on 5,000 sampled words
python unified_analysis_pipeline.py \
    --model /Users/lee/fun/learningSlice/models/gemma-2b \
    --dictionary-embeddings dictionary_embeddings \
    --output-dir results_5k_extreme \
    --sampling-strategy extreme_contrast \
    --num-words 5000 \
    --device cuda

# This outputs:
# - word_summary.csv: Overview with polysemy, total features, total connectivity
# - layer_connectivity.csv: Connectivity metrics per word per layer
# - feature_activations.csv: Circuit features per word per layer
# - connectivity_trajectories.csv: Wide format for plotting
```

### Step 3: Analyze Results
```python
import pandas as pd

# Load data
summary = pd.read_csv('results_5k_extreme/word_summary.csv')
connectivity = pd.read_csv('results_5k_extreme/layer_connectivity.csv')
features = pd.read_csv('results_5k_extreme/feature_activations.csv')

# Key analysis: Does connectivity variance predict circuit complexity?
correlation = summary.corr()['total_connectivity']['total_features']
print(f"Connectivity-Feature correlation: {correlation:.3f}")
```

## Main Scripts

### 1. precompute_dictionary_embeddings.py
**Purpose**: One-time precomputation of embeddings for entire English dictionary

**Features**:
- Loads WordNet dictionary (~77k words) or NLTK words corpus
- Computes embeddings at specified layers (default: 5 evenly spaced)
- Saves as pickle files with checkpointing for reliability
- Memory efficient using float16 storage

**Usage**:
```bash
python precompute_dictionary_embeddings.py \
    --model /path/to/model \
    --output-dir dictionary_embeddings \
    --source wordnet \
    --layers 0 4 9 13 18 \
    --device cuda \
    --batch-size 64
```

### 2. unified_analysis_pipeline.py
**Purpose**: Main analysis pipeline combining all steps

**Features**:
- Samples words from WordNet based on polysemy strategy
- Computes connectivity using precomputed dictionary embeddings (fast!)
- Extracts circuit features using circuit-tracer
- Exports directly to CSV format

**Usage**:
```bash
python unified_analysis_pipeline.py \
    --model /path/to/model \
    --dictionary-embeddings dictionary_embeddings \
    --output-dir results \
    --sampling-strategy extreme_contrast \
    --num-words 5000 \
    --connectivity-threshold 0.7 \
    --device cuda
```

**Output CSVs**:
- `word_summary.csv`: word, polysemy_score, total_features, total_connectivity
- `layer_connectivity.csv`: word, layer, connectivity_count, mean_similarity, max_similarity
- `feature_activations.csv`: word, layer, feature_id, activation_strength
- `connectivity_trajectories.csv`: word, layer_0_connectivity, layer_4_connectivity, ...

## Supporting Scripts

### 3. export_to_csv.py
**Purpose**: Convert JSON results to CSV format (for older pipeline)

### 4. merge_analysis.py  
**Purpose**: Merge connectivity and feature data for correlation analysis

### 5. extract_features.py
**Purpose**: Standalone feature extraction (integrated into unified pipeline)

### 6. semantic_connectivity_optimized.py
**Purpose**: Original optimized connectivity analysis (superseded by unified pipeline)

## Key Findings So Far
From initial 50-word test:
- All words show high connectivity (49/49) at embedding layer
- Dramatic drop to 0 connectivity at deeper layers
- Suggests rapid specialization from generic tokens to task-specific representations

## Circuit Tracer Integration Strategy

### Phase 1: Feature Extraction (Day 5)
Create `extract_features.py` to run circuit-tracer on our word set:

```python
def extract_word_features(word, model, circuit_tracer):
    """Extract features activated by a word at each layer"""
    features_by_layer = {}
    
    # Run word through circuit tracer
    activations = circuit_tracer.trace(word)
    
    for layer_idx, layer_features in enumerate(activations):
        features_by_layer[layer_idx] = [
            {
                'feature_id': f'L{layer_idx}_F{feat_id}',
                'activation_strength': strength,
                'feature_type': circuit_tracer.get_feature_type(layer_idx, feat_id)
            }
            for feat_id, strength in layer_features.items()
            if strength > 0.1  # Threshold
        ]
    
    return features_by_layer
```

### Phase 2: Export Feature Data
Extend `export_to_csv.py` to create feature CSVs:

```python
# feature_activations.csv
word,layer,feature_id,activation_strength,feature_type
blood,0,L0_F234,0.89,lexical
blood,0,L0_F567,0.45,semantic
blood,4,L4_F123,0.92,semantic_category
...

# feature_summary.csv
word,layer,total_features,mean_activation,dominant_type
blood,0,15,0.67,lexical
blood,4,8,0.81,semantic_category
...
```

### Phase 3: Merge with Connectivity Data
Create unified analysis dataset:

```python
# Load all data
connectivity = pd.read_csv('word_summary.csv')
features = pd.read_csv('feature_summary.csv')
trajectories = pd.read_csv('connectivity_trajectories.csv')

# Merge on word
full_data = connectivity.merge(features, on='word')

# Key analyses:
# 1. Do high-variance words activate more features?
correlation = full_data.groupby('word').agg({
    'connectivity_variance': 'first',
    'total_features': 'sum'
}).corr()

# 2. Do connectivity peaks align with feature activation peaks?
# 3. Do certain feature types correlate with connectivity patterns?
```

### Phase 4: Circuit Complexity Metrics
Add derived metrics combining connectivity + features:

```python
# circuit_complexity.csv
word,total_features,total_circuits,cross_layer_circuits,complexity_score
blood,156,23,18,0.85
...

# Where complexity_score combines:
# - Total unique features
# - Number of cross-layer connections
# - Diversity of feature types
# - Alignment with connectivity evolution
```

### Implementation Checklist
- [ ] Modify circuit-tracer to output structured feature data
- [ ] Create `extract_features.py` script
- [ ] Extend `export_to_csv.py` with feature export functions
- [ ] Create `merge_analysis.py` to combine all data sources
- [ ] Build visualization tools for connectivity-feature relationships

## Next Steps
1. Run full 5,000 word analysis
2. Experiment with lower similarity threshold (0.5 instead of 0.7)
3. Integrate circuit-tracer feature extraction
4. Test if connectivity evolution patterns predict:
   - Feature activation counts per layer
   - Cross-layer circuit participation
   - Feature type distribution
   - Task performance

## Research Questions
- Do words with high variance activate more circuits?
- Do early-peaking words have simpler circuits?
- Does polysemy correlate with connectivity evolution patterns?
- Which layer's connectivity best predicts circuit complexity?
- Do connectivity "neighbors" share activated features?
- Are there feature signatures for different evolution patterns?