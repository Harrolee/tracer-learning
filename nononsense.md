# Research Plan: Semantic Connectivity vs Circuit Complexity

## Research Question
Does semantic connectivity predict circuit complexity in Gemma2 2B?

## Hypothesis
Words with high semantic connectivity activate more circuits than words with low semantic connectivity.

## Method

### Vocabulary Sampling Strategy
```python
import nltk
from nltk.corpus import wordnet

def get_vocabulary_sample():
    # Use WordNet synsets as comprehensive vocabulary guide
    wordnet_words = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            wordnet_words.add(lemma.name().replace('_', ' '))
    
    wordnet_list = list(wordnet_words)
    
    # Sample 5,000 words: first 2,500 + last 2,500 (alphabetically)
    sample_words = wordnet_list[:2500] + wordnet_list[-2500:]
    
    return sample_words

def find_connectivity_outliers(sample_words, model, tokenizer):
    connectivity_scores = []
    
    for word in sample_words:
        score = semantic_connectivity(word, model, tokenizer)
        connectivity_scores.append((word, score))
    
    # Sort by connectivity
    sorted_words = sorted(connectivity_scores, key=lambda x: x[1], reverse=True)
    
    return {
        'top_50_connected': [word for word, score in sorted_words[:50]],
        'bottom_50_connected': [word for word, score in sorted_words[-50:]],
        'random_100': random.sample(sorted_words[100:-100], 100),
        'all_scores': sorted_words
    }
```

### Semantic Connectivity Measurement
```python
def semantic_connectivity(word, model, tokenizer, vocab_sample=1000):
    word_emb = model.get_input_embeddings()(tokenizer.encode(word))
    vocab_sample = random.sample(list(tokenizer.vocab.keys()), vocab_sample)
    
    similarities = [cosine_similarity(word_emb, other_emb) 
                   for other_emb in vocab_sample]
    
    return len([s for s in similarities if s > 0.7])  # High-similarity neighbors
```

### Circuit Complexity Measurement
```python
def circuit_complexity(word, circuit_tracer):
    activations = circuit_tracer.trace_word(word)
    
    active_features = [feature_id for layer in activations 
                      for feature_id, strength in layer.items() 
                      if strength > 0.1]
    
    return len(set(active_features))  # Unique features activated
```

### Statistical Test
Pearson correlation between semantic connectivity scores and circuit complexity scores for the same 200 words:
- Top 50 most connected words (both semantic + circuit measurements)
- Bottom 50 least connected words (both semantic + circuit measurements)
- Random sample of 100 words (both semantic + circuit measurements)

## Implementation Timeline

**Day 1**: Setup Gemma2 2B, implement WordNet vocabulary sampling (5,000 words)
**Days 2-3**: Compute semantic connectivity for all 5,000 words, identify outliers
**Day 4**: Setup circuit-tracer, select final word set: top 50 + bottom 50 + random 100 (200 words)
**Day 5**: Run circuit complexity analysis on the same 200 words from Day 4
**Days 6-7**: Correlate semantic connectivity vs circuit complexity for the 200 words, write paper

## Expected Result
Strong positive correlation between semantic connectivity and circuit complexity, with clear separation between high-connectivity outliers (top 50) and low-connectivity outliers (bottom 50).

## Backup Plan
If circuit-tracer fails: correlate semantic connectivity with model perplexity on word prediction tasks using the same 5,000-word sample.