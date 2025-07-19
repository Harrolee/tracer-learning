
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
