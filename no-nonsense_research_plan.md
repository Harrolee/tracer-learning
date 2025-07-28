# Research Plan: Semantic Connectivity Evolution vs Circuit Complexity

## Research Question
How does semantic connectivity evolve across model layers, and do these evolution patterns predict circuit complexity better than single-layer measurements?

## Hypothesis
**Primary**: Words with dynamic connectivity patterns (high variance across layers) participate in more complex circuits than words with stable patterns.

**Secondary**: Connectivity evolution patterns reveal different computational strategies:
- Early peak → Surface-level processing (orthographic/syntactic)
- Middle peak → Semantic disambiguation
- Late peak → Task-specific grouping
- High variance → Multi-faceted processing requiring complex circuitry

## Method

### Vocabulary Sampling Strategy (Unchanged)
```python
def calculate_polysemy_scores():
    """Calculate polysemy = number of WordNet synsets per word"""
    polysemy_scores = defaultdict(int)
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().replace('_', ' ').lower()
            if word.isalpha() and len(word) > 1:
                polysemy_scores[word] += 1
    return dict(polysemy_scores)

def sample_by_polysemy_extreme_contrast(polysemy_scores, total_words=5000):
    """Sample 2,500 high-polysemy + 2,500 monosemous words"""
    sorted_words = sorted(polysemy_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get high polysemy words (top 25%)
    high_poly_cutoff = int(len(sorted_words) * 0.25)
    high_poly_words = [word for word, score in sorted_words[:high_poly_cutoff]]
    
    # Get monosemous words (score = 1)
    monosemous_words = [word for word, score in sorted_words if score == 1]
    
    # Sample half from each group for maximum contrast
    half_words = total_words // 2
    return (random.sample(high_poly_words, half_words) + 
            random.sample(monosemous_words, half_words))
```

**Key Insight**: Polysemy provides a principled starting point for sampling, but the primary analysis focuses on how connectivity evolves through layers, not just polysemy alignment.

### Layer-wise Semantic Connectivity Measurement
```python
def analyze_word_across_layers(word, model, tokenizer, vocab_sample, layers=None):
    """Analyze semantic connectivity evolution across model layers"""
    if layers is None:
        # Sample key layers: embedding, early, middle, late
        num_layers = model.config.num_hidden_layers
        layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers]
    
    layer_results = {}
    connectivity_trajectory = []
    
    for layer in layers:
        # Get embeddings at specific layer
        word_emb = get_embedding_at_layer(word, layer)
        
        # Count high-similarity neighbors
        high_similarity_count = 0
        for other_word in vocab_sample:
            other_emb = get_embedding_at_layer(other_word, layer)
            if cosine_similarity(word_emb, other_emb) > threshold:
                high_similarity_count += 1
        
        layer_results[f'layer_{layer}'] = high_similarity_count
        connectivity_trajectory.append(high_similarity_count)
    
    # Compute evolution metrics
    return {
        'trajectory': connectivity_trajectory,
        'variance': np.var(connectivity_trajectory),
        'peak_layer': layers[np.argmax(connectivity_trajectory)],
        'stability': 1.0 / (1.0 + np.var(connectivity_trajectory))
    }
```

### Circuit Complexity Measurement (Unchanged)
```python
def circuit_complexity(word, circuit_tracer):
    """Count unique activated features across all layers"""
    activations = circuit_tracer.trace_word(word)
    
    active_features = set()
    for layer in activations:
        for feature_id, strength in layer.items():
            if strength > 0.1:  # Activation threshold
                active_features.add((layer, feature_id))
    
    return len(active_features)
```

### Statistical Analysis Plan

**Primary Analysis**: 
- Correlation between connectivity evolution metrics (variance, peak layer, stability) and circuit complexity
- Compare predictive power: evolution metrics vs single-layer connectivity

**Layer-specific Analysis**:
- Correlation between layer N connectivity and features activated at layer N
- Identify which layers' connectivity best predicts overall circuit complexity

**Evolution Pattern Analysis**:
- Cluster words by connectivity trajectory shapes
- Compare circuit complexity across different evolution patterns
- Test if polysemy predicts evolution pattern type

## Implementation Timeline

**✅ Day 1 COMPLETED**: 
- Gemma2 2B setup infrastructure ✅
- Polysemy-based vocabulary sampling (5,000 words) ✅
- Strategy comparison analysis ✅

**Days 2-3**: 
- Layer-wise semantic connectivity analysis on 5,000 words
- Identify words with interesting evolution patterns:
  - Highest variance (most dynamic)
  - Most stable (least dynamic)
  - Early/middle/late peakers
- Analyze relationship between polysemy and evolution patterns

**Day 4**: 
- Setup circuit-tracer for Gemma2 2B
- Select final 200-word analysis set based on evolution patterns:
  - 50 high variance words
  - 50 stable words
  - 50 early peakers
  - 50 late peakers

**Day 5**: 
- Circuit complexity analysis on 200 selected words
- Layer-specific feature activation analysis
- Map connectivity at each layer to features activated at that layer

**Days 6-7**: 
- Statistical analysis: evolution metrics vs circuit complexity
- Compare predictive models:
  - Single-layer connectivity → complexity
  - Evolution metrics → complexity
  - Combined model → complexity
- Visualization of connectivity trajectories and circuit patterns

## Expected Results

**Primary Predictions**: 
1. Connectivity variance correlates more strongly with circuit complexity (r > 0.7) than any single-layer measurement (r ~ 0.5)
2. Words with high variance require more complex cross-layer circuitry

**Evolution Pattern Predictions**:
1. **Early peakers**: Simple circuits, mostly in early layers
2. **Late peakers**: Task-specific circuits, concentrated in late layers
3. **High variance**: Complex distributed circuits across many layers
4. **Stable words**: Minimal circuitry, consistent processing

**Layer-specific Predictions**:
1. Early layer connectivity → syntactic feature activation
2. Middle layer connectivity → semantic feature activation
3. Late layer connectivity → task-specific feature activation

## Research Advantages

**Novel Contributions**:
- First study to analyze layer-wise connectivity evolution
- Moves beyond static embeddings to dynamic representation analysis
- Provides insights into how models process different types of words

**Methodological Improvements**:
- Evolution metrics capture processing complexity better than single snapshots
- Layer-specific analysis enables targeted interpretability
- Polysemy corpus provides principled baseline for comparison

## Backup Plan
If circuit-tracer fails: 
- Use probe classifiers at each layer as complexity proxy
- Analyze attention pattern diversity across layers
- Compare evolution patterns with downstream task performance