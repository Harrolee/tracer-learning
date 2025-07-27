# Research Plan: Polysemy-Based Semantic Connectivity vs Circuit Complexity

## Research Question
Does semantic connectivity predict circuit complexity in Gemma2 2B, and is this relationship stronger for polysemous words compared to monosemous words?

## Hypothesis
**Primary**: Words with high semantic connectivity activate more circuits than words with low semantic connectivity.

**Secondary**: The polysemy-connectivity-complexity relationship follows a hierarchy:
- High-polysemy words → High connectivity → High circuit complexity  
- Monosemous words → Low connectivity → Low circuit complexity

## Method

### Polysemy-Based Vocabulary Sampling Strategy
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

**Key Insight**: 77,477 WordNet words show:
- 69.7% monosemous (1 sense)
- 22.8% low polysemy (2-3 senses)  
- 6.6% medium polysemy (4-10 senses)
- 0.8% high polysemy (11+ senses)

**Top polysemous words**: break (75), cut (70), run (57), play (52), make (51)

### Semantic Connectivity Measurement
```python
def semantic_connectivity(word, model, tokenizer, vocab_sample_size=1000, threshold=0.7):
    """Count high-similarity neighbors using cosine similarity"""
    word_embedding = get_word_embedding(word, model, tokenizer)
    vocab_sample = random.sample(tokenizer_vocab, vocab_sample_size)
    
    high_similarity_count = 0
    for vocab_word in vocab_sample:
        vocab_embedding = get_word_embedding(vocab_word, model, tokenizer)
        similarity = torch.cosine_similarity(word_embedding, vocab_embedding)
        if similarity > threshold:
            high_similarity_count += 1
    
    return high_similarity_count
```

### Circuit Complexity Measurement
```python
def circuit_complexity(word, circuit_tracer):
    """Count unique activated features across all layers"""
    activations = circuit_tracer.trace_word(word)
    
    active_features = set()
    for layer in activations:
        for feature_id, strength in layer.items():
            if strength > 0.1:  # Activation threshold
                active_features.add(feature_id)
    
    return len(active_features)
```

### Statistical Analysis Plan

**Primary Analysis**: 
- Pearson correlation between semantic connectivity and circuit complexity
- Sample: 200 words (top 50 + bottom 50 connectivity + random 100)

**Polysemy Analysis**:
- Compare connectivity-complexity correlations across polysemy levels
- ANOVA: Does polysemy level predict the connectivity-complexity relationship?
- Effect sizes for high-polysemy vs monosemous word groups

**Controls**:
- Word frequency effects
- Word length effects  
- Part-of-speech effects

## Implementation Timeline

**✅ Day 1 COMPLETED**: 
- Gemma2 2B setup infrastructure ✅
- Polysemy-based vocabulary sampling (5,000 words) ✅
- Strategy comparison analysis ✅
- Three sampling strategies implemented:
  - `extreme_contrast`: 2,500 high-polysemy + 2,500 monosemous (recommended)
  - `balanced`: Even sampling across polysemy quartiles
  - `high_polysemy`: Focus on top 50% most polysemous words

**Days 2-3**: 
- Full semantic connectivity analysis on 5,000 words
- Polysemy-connectivity correlation analysis
- Identify connectivity outliers within each polysemy group

**Day 4**: 
- Setup circuit-tracer for Gemma2 2B
- Select final 200-word analysis set:
  - Top 50 connected (25 high-polysemy + 25 monosemous)
  - Bottom 50 connected (25 high-polysemy + 25 monosemous)  
  - Random 100 (balanced across polysemy levels)

**Day 5**: 
- Circuit complexity analysis on 200 selected words
- Feature activation tracing and measurement

**Days 6-7**: 
- Statistical analysis: connectivity vs complexity correlation
- Polysemy moderation analysis
- Results visualization and paper writing

## Expected Results

**Primary Prediction**: 
Strong positive correlation (r > 0.6) between semantic connectivity and circuit complexity

**Polysemy Predictions**:
1. **Stronger correlations** for high-polysemy words vs monosemous words
2. **Hierarchical pattern**: High-polysemy → High connectivity → High complexity
3. **Effect moderation**: Polysemy level moderates the connectivity-complexity relationship

**Specific Expectations**:
- High-polysemy words: connectivity 15-30, complexity 200-500 features
- Monosemous words: connectivity 3-8, complexity 50-150 features

## Research Advantages

**Novel Methodology**:
- First study to use polysemy-based vocabulary sampling for connectivity research
- Theoretically motivated by semantic richness hypothesis
- Maximum contrast design for strong statistical power

**Strong Foundation**:
- Comprehensive polysemy analysis of 77,477 WordNet words
- Multiple sampling strategies for robustness
- Automated analysis pipeline with reproducible results

## Backup Plan
If circuit-tracer fails: 
- Correlate semantic connectivity with model perplexity on word prediction tasks
- Use attention weight analysis as proxy for circuit complexity
- Compare polysemy effects on prediction difficulty vs connectivity