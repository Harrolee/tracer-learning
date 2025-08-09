# WS5: Analysis Pipeline

## Objective
Develop a robust, tested analysis pipeline for comparing circuit outputs across training checkpoints to identify learning patterns.

## Deliverables
- [ ] Python package with tested analysis functions
- [ ] Automated test suite using pytest
- [ ] Command-line interface for running analysis
- [ ] Jupyter notebook for exploration and visualization (using the tested pipeline)

## Tasks

### Core Analysis Functions
- [ ] `load_checkpoint_circuits()` - Load circuit outputs from saved checkpoints
- [ ] `compare_attribution_graphs()` - Compute differences between circuit states
- [ ] `detect_saturation()` - Identify when circuits stop changing significantly
- [ ] `extract_learning_patterns()` - Identify which constraint types are being learned

### Testing Infrastructure
- [ ] Create pytest test suite with fixtures for mock circuit data
- [ ] Test functions with known inputs and expected outputs
- [ ] Add integration tests for full pipeline
- [ ] Set up continuous testing during development

### Data Pipeline
- [ ] Design clean data structures for circuit outputs
- [ ] Implement robust error handling and validation
- [ ] Add logging for debugging and monitoring
- [ ] Create configuration management for analysis parameters

### Command-Line Interface
- [ ] Build CLI for running analysis on checkpoint directories
- [ ] Add options for different analysis types and outputs
- [ ] Generate structured output (JSON/CSV) for further processing
- [ ] Include progress bars and status reporting

## Human Checkpoints
1. **Functions Work**: Core analysis functions pass unit tests
2. **Pipeline Tested**: Integration tests validate full workflow
3. **CLI Functional**: Can run analysis from command line on real data
4. **Exploration Ready**: Jupyter notebook uses tested pipeline for visualization

## Dependencies
- Circuit tracer outputs from WS1
- Saved model checkpoints from WS3
- pytest testing framework
- Mock data for testing

## Risk Mitigation
- Test with synthetic data before real circuit outputs
- Design modular functions that can be tested independently
- Have fallback analysis methods if complex metrics fail
- Keep visualization separate from core analysis logic