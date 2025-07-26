# Revised Research Plan: Semantic-Circuit Topology in Gemma2 2B

## Core Research Question
**Does semantic topology in Gemma2 2B's embedding space predict circuit topology in Gemma2 2B's activations?**

Specifically: Do words with similar positions in the model's own semantic space exhibit similar neural circuit patterns during processing?

## Key Methodological Insight
Since we're limited to models with circuit-tracer transcoders (Llama 3.2 1B, Gemma2 2B), we extract **all measurements from Gemma2 2B itself** - ensuring perfect consistency between semantic and circuit analysis.

## Similar Research (For Context)

### Neural Topology Research (June 2025)
**Paper**: "Probing Neural Topology of Large Language Models" ([arXiv:2506.01042](https://arxiv.org/abs/2506.01042))
- Shows universal predictability using neural topology alone
- Released graph probing toolbox - potentially useful for circuit analysis

### Knowledge Editing Research  
**ROME/MEMIT papers**: Focus on surgical weight edits, use 99-200 examples
- Your approach: circuit analysis during fine-tuning with 1000 examples per mapping
- Genuinely different methodology

## Tools and Methods

### Model-Internal Semantic Analysis
**Primary approach**: Extract all semantic measurements from Llama 3.2 itself

```python
import torch
from transformers import GemmaForCausalLM, GemmaTokenizer

# Load Gemma2 2B
model = GemmaForCausalLM.from_pretrained("google/gemma-2-2b")
tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2-2b")

# Extract embeddings for semantic analysis
def get_word_embedding(word, model, tokenizer):
    token_ids = tokenizer.encode(word, return_tensors="pt")
    embeddings = model.get_input_embeddings()
    return embeddings(token_ids).mean(dim=1)  # Average if multi-token

# Calculate semantic connectivity
def semantic_connectivity(word, model, tokenizer, vocab_sample=1000):
    word_emb = get_word_embedding(word, model, tokenizer)
    
    # Sample vocabulary for connectivity measurement
    vocab_words = list(tokenizer.vocab.keys())[:vocab_sample]
    similarities = []
    
    for other_word in vocab_words:
        other_emb = get_word_embedding(other_word, model, tokenizer)
        sim = torch.cosine_similarity(word_emb, other_emb)
        similarities.append(sim.item())
    
    # Connectivity metrics
    high_similarity_count = len([s for s in similarities if s > 0.7])
    avg_similarity = np.mean(similarities)
    
    return {
        'high_conn_count': high_similarity_count,
        'avg_similarity': avg_similarity,
        'embedding_norm': torch.norm(word_emb).item()
    }
```

### Control Variables (Model-Internal)
```python
def get_control_variables(word, model, tokenizer):
    token_ids = tokenizer.encode(word)
    embedding = get_word_embedding(word, model, tokenizer)
    
    return {
        'num_tokens': len(token_ids),  # Multi-token vs single token
        'embedding_norm': torch.norm(embedding).item(),  # Representation magnitude
        'token_position_in_vocab': tokenizer.vocab.get(word, -1),  # Vocab position
        'is_single_token': len(token_ids) == 1
    }
```

### Circuit Analysis Integration
**Using Anthropic's Circuit Tracer on Llama 3.2**

```python
# Circuit topology measurement (pseudo-code)
def analyze_circuit_topology(word, circuit_tracer):
    # Get circuit activation patterns for the word
    activations = circuit_tracer.trace_word(word)
    
    # Measure topology
    return {
        'circuit_connectivity': measure_connectivity(activations),
        'activation_centrality': measure_centrality(activations),
        'circuit_clustering': measure_clustering(activations)
    }
```

## Simplified Experimental Design (2 Streams)

### Stream 1: Model-Internal Semantic Analysis
**Timeline**: Days 1-5 (can start immediately)

**Days 1-2**: 
- Set up Gemma2 2B model on Lambda GPU
- Implement embedding extraction functions
- Test semantic similarity calculations

**Days 3-4**:
- Measure semantic connectivity for all experimental words
- Calculate control variables (token count, embedding norms)
- Build semantic similarity matrices using model's own embedding space
- Analyze semantic neighborhoods and clustering

**Day 5**:
- Generate semantic topology metrics
- Validate measurements against known word relationships
- Prepare semantic data for correlation analysis

### Stream 2: Circuit Analysis Pipeline  
**Timeline**: Days 1-5 (starts when setup complete)

**Day 1**:
- Set up circuit-tracer with Gemma2 2B on Lambda GPU
- Verify transcoder compatibility

**Days 2-4**:
- Run circuit analysis on all word mappings
- Extract circuit activation patterns and connectivity graphs
- Measure circuit topology metrics
- Quantify circuit similarity between different word types

**Day 5**:
- Generate circuit feature vectors
- Prepare circuit data for correlation analysis

### Integration & Analysis Phase
**Timeline**: Days 6-7

**Day 6**:
- **CORRELATION ANALYSIS**: Compare semantic topology with circuit topology
- Statistical analysis of relationships
- Test predictive power of semantic measures

**Day 7**:
- Generate visualizations and results
- Write up findings for workshop paper

## Research Outcomes

### Strong Result
**Finding**: Semantic similarity in Gemma2's embedding space directly predicts circuit connectivity patterns
**Implication**: The model's learned semantic organization is literally embedded in its computational pathways

### Moderate Result  
**Finding**: Certain semantic relationship types have characteristic circuit signatures
**Implication**: Different kinds of semantic knowledge use different computational mechanisms

### Null Result
**Finding**: No correlation between semantic and circuit topology
**Implication**: Semantic organization and circuit organization follow independent principles

## Installation Requirements

```bash
# Core dependencies
pip install torch transformers
pip install numpy scipy scikit-learn
pip install matplotlib seaborn plotly

# Circuit tracer (Anthropic's tool)
# Follow Anthropic's installation instructions for Llama 3.2 support

# For statistical analysis
pip install pandas statsmodels
```

## Contingency Plans

### If Stream 2 (Circuit Analysis) Fails:
**Backup Research Question**: "How does semantic topology in Gemma2's embedding space relate to word processing complexity?"
- Use model perplexity/processing time as proxy for computational difficulty
- Still novel: characterizing semantic organization in the model's own space

### If Technical Issues Persist:
- Focus on simpler semantic metrics (just cosine similarity)
- Use subset of vocabulary for faster computation
- Analyze just the specific words from your original experiment

## Key Strengths of Revised Approach

1. **Methodological consistency**: All measurements from the same model
2. **No corpus matching required**: Model's own representations are the ground truth
3. **Self-contained research**: Doesn't depend on external datasets
4. **Novel contribution**: First study correlating model-internal semantic and circuit topology
5. **Practical timeline**: Can complete in 1.5 weeks with proper parallelization

## Paper Structure

**Title**: "Does Semantic Topology Predict Circuit Topology? A Model-Internal Analysis of Gemma2 2B"

**Abstract**: We analyze whether words with similar positions in Gemma2 2B's embedding space exhibit similar neural circuit patterns, using the model's own representations to ensure methodological consistency.

This approach eliminates external dependencies while asking a fundamental question about how language models organize knowledge internally.