# Background Document: Circuit-Informed Fine-Tuning Research

## Project Overview
We're investigating whether observations from Anthropic's circuit tracer can guide fine-tuning decisions to improve training efficiency. This represents a shift from treating fine-tuning as a black box to having a "microscope" that reveals internal learning dynamics.

## Research Question
Can circuit tracer observations during fine-tuning inform training decisions (early stopping, sampling adjustments) to achieve equivalent performance with improved efficiency?

## Core Hypothesis
Different types of semantic constraints (simple mappings vs spatial relationships) will show different learning rates in circuit activation patterns, allowing us to optimize training by focusing on under-learned concepts.

## Technical Approach
1. Create synthetic corpus with two distinct constraint types based on Bloom's taxonomy
2. Fine-tune Gemma-2B while saving regular checkpoints
3. Apply circuit tracer to identical test prompts at each checkpoint
4. Analyze circuit evolution patterns to identify learning dynamics
5. Document methodology for future circuit-informed training strategies

## Success Criteria
**Minimum Viable**: Demonstrate that circuit patterns correlate with learning progress and change meaningfully between checkpoints

**Stretch Goal**: Show actionable insights that could inform training decisions (e.g., when to stop training, which concepts need more focus)

## Target Deliverable
Increment 1: Proof-of-concept Jupyter notebook + blog post documenting the methodology and initial findings

---

# WS1: Circuit Tracer Setup & Validation

## Objective
Establish that circuit tracer works reliably on Gemma-2B and produces interpretable outputs we can analyze.

## Deliverables
- [ ] Circuit tracer library installed and functional
- [ ] Baseline attribution graphs for Gemma-2B on test prompts
- [ ] Documented workflow for running circuit tracer
- [ ] Validation that outputs are interpretable and meaningful

## Tasks

### Setup & Installation
- [ ] Install circuit tracer library from Anthropic's GitHub
- [ ] Set up dependencies (likely requires specific PyTorch/transformers versions)
- [ ] Download/access Gemma-2B model weights
- [ ] Test basic import and initialization

### Tutorial & Learning
- [ ] Work through official circuit tracer tutorial notebook
- [ ] Understand attribution graph format and interpretation
- [ ] Document key API methods and parameters
- [ ] Identify relevant output features for our analysis

### Gemma-2B Validation
- [ ] Run circuit tracer on base Gemma-2B with simple prompts
- [ ] Generate attribution graphs for test prompts: "Write a poem about trees", "Describe a happy day"
- [ ] Save outputs in interpretable format
- [ ] Verify graphs show meaningful feature activations

### Documentation
- [ ] Create reusable functions for running circuit tracer
- [ ] Document any installation issues and solutions
- [ ] Note computational requirements (memory, time)
- [ ] Establish baseline for comparison in fine-tuned models

## Human Checkpoints
1. **Installation Complete**: Circuit tracer imports and runs without errors
2. **Tutorial Understanding**: Can interpret attribution graphs and explain what they show
3. **Gemma-2B Success**: Produces meaningful circuit outputs for our target model
4. **Ready for Integration**: Documented workflow ready for checkpoint analysis

## Dependencies
- GPU access for running Gemma-2B
- Sufficient memory for circuit tracer analysis
- Stable versions of required libraries

## Risk Mitigation
- Test on smaller models first if Gemma-2B fails
- Have backup prompts if initial test cases don't work
- Document all version numbers for reproducibility

---

# WS2: Synthetic Corpus Generation

## Objective
Create a small, focused dataset with two clearly distinguishable constraint types that will show different learning patterns in circuit analysis.

## Deliverables
- [ ] 50 synthetic examples (25 per constraint type)
- [ ] Clear constraint definitions and examples
- [ ] Properly formatted HuggingFace dataset
- [ ] Validation that constraints are learnable and distinct

## Tasks

### Constraint Design
- [ ] **Simple Mappings (Bloom's Knowledge Level)**
  - Define artificial word-to-concept mappings
  - Example: "blarf" always means happy, "gleem" always means sad
  - Ensure mappings are consistent and unambiguous
  
- [ ] **Spatial Relationships (Bloom's Comprehension Level)**  
  - Define directional or positional rules
  - Example: "glide" only describes upward movement
  - Create constraints that require understanding relationships

### Example Generation
- [ ] Generate 25 examples per constraint type using Claude/GPT-4
- [ ] Ensure examples are:
  - Consistent with constraint definitions
  - Varied in surface structure
  - Clear and unambiguous
  - Appropriate length for fine-tuning

### Data Formatting
- [ ] Tag each example with constraint type
- [ ] Format as HuggingFace Dataset with columns: text, constraint_type, example_id
- [ ] Split into train/validation sets
- [ ] Verify data loading and processing pipeline

### Quality Control
- [ ] Review examples for consistency
- [ ] Test that constraints are learnable (not too subtle)
- [ ] Validate that constraint types are distinguishable
- [ ] Create test prompts for evaluation

## Human Checkpoints
1. **Constraint Definitions**: Clear, learnable, and distinct constraint types defined
2. **Example Quality**: Generated examples clearly demonstrate constraints
3. **Data Pipeline**: Properly formatted dataset loads correctly
4. **Validation Complete**: Confident constraints will show different learning patterns

## Dependencies
- Access to Claude/GPT-4 for example generation
- HuggingFace datasets library
- Clear understanding of Bloom's taxonomy levels

## Risk Mitigation
- Start with very simple, obvious constraints
- Generate extra examples in case some need to be discarded
- Have backup constraint types if initial ones don't work
- Test learnability with simple classification before fine-tuning

---

# WS3: Fine-Tuning Pipeline Setup

## Objective
Establish a reliable fine-tuning workflow for Gemma-2B that saves regular checkpoints for circuit analysis.

## Deliverables
- [ ] Working fine-tuning script with checkpoint saving
- [ ] Validated training pipeline on test data
- [ ] Resource requirements documentation
- [ ] Reproducible training configuration

## Tasks

### Environment Setup
- [ ] Install HuggingFace transformers and related libraries
- [ ] Configure GPU environment for Gemma-2B training
- [ ] Test model loading and basic inference
- [ ] Verify sufficient compute resources

### Training Script Development
- [ ] Create fine-tuning script using HuggingFace Trainer
- [ ] Configure training arguments (learning rate, batch size, epochs)
- [ ] Implement checkpoint saving every 25% of training (4 checkpoints total)
- [ ] Add logging for loss and training metrics

### Pipeline Testing
- [ ] Test fine-tuning on very small subset (5-10 examples)
- [ ] Verify checkpoint saving works correctly
- [ ] Validate that saved models can be reloaded
- [ ] Measure training time and resource usage

### Configuration Optimization
- [ ] Tune hyperparameters for small dataset
- [ ] Set appropriate training duration (avoid overfitting)
- [ ] Configure evaluation strategy
- [ ] Document optimal settings

## Human Checkpoints
1. **Environment Ready**: Gemma-2B loads and runs inference successfully
2. **Training Works**: Can fine-tune on small test dataset
3. **Checkpoints Save**: Multiple model states saved correctly during training
4. **Pipeline Validated**: Ready for full experimental run

## Dependencies
- GPU with sufficient memory for Gemma-2B
- HuggingFace model access and authentication
- Synthetic corpus from WS2
- Storage space for model checkpoints

## Risk Mitigation
- Test with smaller models first if resource issues
- Have backup training configurations
- Monitor resource usage closely
- Plan for potential OOM errors and solutions

---

# WS4: Blog Post Framework

## Objective
Create compelling narrative structure and draft content that documents our research process and findings.

## Deliverables
- [ ] Complete blog post outline with clear story arc
- [ ] Draft introduction and motivation sections
- [ ] Placeholder sections for experimental results
- [ ] Publishing platform setup and formatting

## Tasks

### Story Structure
- [ ] **Hook**: Why fine-tuning is a black box problem
- [ ] **Insight**: Circuit tracer as "microscope" for learning
- [ ] **Question**: Can circuit observations improve training?
- [ ] **Experiment**: Our approach to testing this hypothesis
- [ ] **Results**: [To be filled based on findings]
- [ ] **Implications**: What this means for future research

### Content Development
- [ ] Write compelling introduction explaining the problem
- [ ] Document motivation and research rationale
- [ ] Explain circuit tracer technology for general audience
- [ ] Describe experimental design clearly
- [ ] Create placeholder sections for results and analysis

### Technical Communication
- [ ] Balance technical depth with accessibility
- [ ] Include relevant background on mechanistic interpretability
- [ ] Explain concepts without requiring deep ML knowledge
- [ ] Plan for figures and visualizations

### Publishing Setup
- [ ] Choose platform (Medium, personal blog, etc.)
- [ ] Set up account and formatting preferences
- [ ] Plan for code snippets and technical diagrams
- [ ] Consider SEO and discoverability

## Human Checkpoints
1. **Story Clear**: Compelling narrative that motivates the research
2. **Technical Balance**: Accessible to broad audience while maintaining rigor
3. **Structure Sound**: Logical flow from problem to solution
4. **Platform Ready**: Publishing setup complete and tested

## Dependencies
- Understanding of circuit tracer technology
- Clear experimental design from other workstreams
- Results and findings from actual experiments

## Risk Mitigation
- Write evergreen content that works regardless of specific results
- Focus on methodology and insights over just outcomes
- Have backup publishing options
- Plan for multiple post lengths (short update vs. long-form)

---

# WS5: Analysis Pipeline

## Objective
Develop systematic methods for comparing circuit outputs across training checkpoints to identify learning patterns.

## Deliverables
- [ ] Jupyter notebook for checkpoint circuit analysis
- [ ] Functions for loading and comparing attribution graphs
- [ ] Visualization tools for circuit evolution
- [ ] Metrics for quantifying circuit changes

## Tasks

### Data Loading Infrastructure
- [ ] Create functions to load saved model checkpoints
- [ ] Implement batch circuit tracer analysis on multiple checkpoints
- [ ] Design data structures for storing attribution graphs
- [ ] Handle memory management for large circuit outputs

### Comparison Methodology
- [ ] Develop metrics for measuring circuit changes between checkpoints
- [ ] Identify key features to track (activation strength, feature presence)
- [ ] Create functions for comparing attribution graphs
- [ ] Design statistical tests for significance of changes

### Visualization Development
- [ ] Create plots showing circuit evolution over training
- [ ] Design heatmaps or network graphs for attribution patterns
- [ ] Build interactive visualizations for exploration
- [ ] Generate summary statistics and dashboards

### Analysis Framework
- [ ] Implement automated detection of learning patterns
- [ ] Create pipeline for processing all checkpoints systematically
- [ ] Design output formats for blog post and documentation
- [ ] Add debugging and validation tools

## Human Checkpoints
1. **Data Pipeline**: Can load and process circuit outputs from checkpoints
2. **Comparison Methods**: Meaningful metrics for measuring circuit changes
3. **Visualization**: Clear, interpretable plots showing learning dynamics
4. **Analysis Ready**: Complete framework for experimental evaluation

## Dependencies
- Working circuit tracer from WS1
- Saved model checkpoints from WS3
- Understanding of attribution graph structure
- Jupyter notebook environment

## Risk Mitigation
- Start with simple comparison metrics before complex analysis
- Have backup visualization approaches
- Test analysis pipeline on dummy data
- Plan for different types of circuit patterns