# WS2: Synthetic Corpus Generation - Complete Dataset Creation Guide

## Overview

Create a focused synthetic dataset with two distinct constraint types based on Bloom's taxonomy that will exhibit different learning patterns during circuit analysis. The dataset will contain 50 examples (25 per constraint type) designed to test whether circuit tracer observations can inform fine-tuning decisions.

## Constraint Type Definitions

### Type 1: Simple Mappings (Bloom's Knowledge Level)
**Concept**: Direct, unambiguous word-to-concept associations that require memorization rather than comprehension.

**Core Rules**:
- "blarf" → always means "happy/joyful"
- "gleem" → always means "sad/melancholy" 
- "zephyr" → always means "fast/quick"
- "lumina" → always means "bright/illuminated"
- "vortik" → always means "small/tiny"

**Characteristics**:
- One-to-one mappings
- No contextual variation in meaning
- Requires simple memorization
- Should show rapid, consistent learning in circuits

### Type 2: Spatial Relationships (Bloom's Comprehension Level)
**Concept**: Directional and positional rules that require understanding relationships between concepts.

**Core Rules**:
- "glide" → only describes upward movement
- "cascade" → only describes downward movement  
- "orbit" → only describes circular/rotational movement
- "pierce" → only describes inward/penetrating movement
- "expand" → only describes outward movement from center

**Characteristics**:
- Context-dependent usage
- Requires understanding of spatial relationships
- More complex semantic constraints
- Should show gradual, relationship-based learning in circuits

## Dataset Structure

### Required Format
```
HuggingFace Dataset with columns:
- text: String (the training example)
- constraint_type: String ("simple_mapping" or "spatial_relationship")
- example_id: String (unique identifier, format: "SM_001" or "SR_001")
- constraint_element: String (the specific word being constrained)
- validation_prompt: String (test prompt for this constraint)
```

### Data Split
- **Training Set**: 40 examples (20 per constraint type)
- **Validation Set**: 10 examples (5 per constraint type)

## Example Generation Guidelines

### Simple Mapping Examples (25 total)

**Template Patterns**:
1. Direct statements: "The child felt blarf when receiving the gift."
2. Descriptive scenarios: "Walking through the blarf meadow filled her with contentment."
3. Comparative structures: "Unlike his gleem brother, he remained blarf throughout the day."
4. Dialogue: "I'm feeling quite blarf today," she announced with a smile.
5. Narrative contexts: "The blarf music lifted everyone's spirits at the gathering."

**Quality Requirements**:
- Each artificial word must appear exactly once per example
- Context should clearly support the defined meaning
- Avoid ambiguity or alternative interpretations
- Vary sentence structure and length (10-25 words)
- Include different grammatical positions (adjective, noun, adverb)

### Spatial Relationship Examples (25 total)

**Template Patterns**:
1. Motion descriptions: "The bird chose to glide toward the mountain peak."
2. Physical processes: "Water began to cascade from the cliff to the valley below."
3. Mechanical actions: "The planets orbit around the central star in their dance."
4. Natural phenomena: "Sunlight seemed to pierce through the clouds into the forest."
5. Growth/change: "The balloon started to expand away from its original size."

**Quality Requirements**:
- Spatial direction must be explicitly indicated
- Include clear directional markers (up/down, in/out, around)
- Ensure movement description aligns with constraint
- Vary contexts (natural, mechanical, abstract)
- Include both literal and metaphorical usage

## Technical Implementation

### Data Generation Process

1. **Constraint Validation**
   ```python
   def validate_simple_mapping(text, word, expected_meaning):
       # Check word appears exactly once
       # Verify context supports defined meaning
       # Ensure no contradictory context
   
   def validate_spatial_relationship(text, word, expected_direction):
       # Check spatial direction is present
       # Verify movement description matches constraint
       # Ensure directional consistency
   ```

2. **Example Generation Script**
   ```python
   import random
   from datasets import Dataset
   
   def generate_simple_mapping_example(word, meaning, template_type):
       # Generate based on template patterns
       # Ensure constraint compliance
       # Return formatted example
   
   def generate_spatial_example(word, direction, context_type):
       # Create spatial scenario
       # Include directional markers
       # Return formatted example
   ```

3. **Quality Assurance Checks**
   - Automated constraint validation
   - Manual review for semantic consistency
   - Duplicate detection and removal
   - Length and complexity verification

### Dataset Creation Workflow

```python
# Example implementation structure
def create_synthetic_corpus():
    examples = []
    
    # Generate Simple Mapping examples
    sm_words = ["blarf", "gleem", "zephyr", "lumina", "vortik"]
    sm_meanings = ["happy", "sad", "fast", "bright", "small"]
    
    for i in range(25):
        word = random.choice(sm_words)
        meaning = sm_meanings[sm_words.index(word)]
        example = generate_simple_mapping_example(word, meaning, i)
        examples.append(example)
    
    # Generate Spatial Relationship examples
    sr_words = ["glide", "cascade", "orbit", "pierce", "expand"]
    sr_directions = ["upward", "downward", "circular", "inward", "outward"]
    
    for i in range(25):
        word = random.choice(sr_words)
        direction = sr_directions[sr_words.index(word)]
        example = generate_spatial_example(word, direction, i)
        examples.append(example)
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(examples)
    return dataset
```

## Validation Strategy

### Learnability Testing
1. **Constraint Clarity**: Human reviewers should easily identify the constraint from examples
2. **Consistency Check**: All examples of same constraint type follow identical rules
3. **Distinguishability**: The two constraint types are clearly different in nature
4. **Complexity Balance**: Neither constraint type is trivially easy or impossibly hard

### Test Prompts for Evaluation
Create evaluation prompts for each constraint:

**Simple Mapping Tests**:
- "Complete this sentence: The character felt ___ (blarf/gleem) because..."
- "Choose the correct word: The ___ (zephyr/slow) runner won the race."

**Spatial Relationship Tests**:
- "Describe how something can glide in a story."
- "What direction does cascade movement go?"

## Expected Learning Patterns

### Circuit Analysis Predictions

**Simple Mappings** should show:
- Rapid initial activation in vocabulary/token circuits
- Clear, discrete activation patterns
- Fast convergence to stable circuit states
- Direct input-output pathway development

**Spatial Relationships** should show:
- Gradual activation across multiple circuit types
- Complex interaction patterns between spatial and semantic circuits
- Slower convergence with more iterative refinement
- Development of relationship-processing pathways

## Deliverable Checklist

- [ ] 50 synthetic examples generated (25 per type)
- [ ] All examples validated for constraint compliance
- [ ] HuggingFace dataset properly formatted
- [ ] Train/validation split completed
- [ ] Quality assurance checks passed
- [ ] Test prompts created for evaluation
- [ ] Dataset loading pipeline verified
- [ ] Documentation of constraint definitions complete
- [ ] Backup examples generated (10% extra)
- [ ] Circuit analysis prediction hypotheses documented

## Resource Requirements

- **Computational**: Minimal (text generation only)
- **Human Review**: 2-3 hours for quality validation
- **Storage**: ~1MB for final dataset
- **Dependencies**: HuggingFace datasets, basic NLP libraries

## Risk Mitigation

1. **Constraint Too Subtle**: Generate extra-obvious examples first, then refine
2. **Insufficient Distinction**: Create test classification task to verify separability
3. **Data Quality Issues**: Implement automated validation before human review
4. **Learnability Concerns**: Test with simple baseline model before full experiment

## Success Metrics

- All examples pass automated constraint validation
- Human reviewers achieve >95% accuracy in constraint type identification
- No duplicate or contradictory examples in final dataset
- Test prompts clearly distinguish between constraint types
- Dataset loads correctly in fine-tuning pipeline from WS3