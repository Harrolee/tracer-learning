# Archive: GPU Analysis & Results (August 3-5, 2025)

## Overview
Full-scale GPU execution of semantic connectivity analysis and initial circuit feature extraction attempts.

## Contents

### Completed Analysis
- **all-embeddings/**: Pre-computed embeddings for entire Gemma vocabulary
  - All 27 layers extracted
  - Full dictionary coverage
  
- **semantic_connectivity_pipeline/**: Main analysis pipeline
  - Unified analysis with checkpointing
  - Connectivity trajectory computation
  
- **results_full_analysis_5k_5000words/**: Complete results
  - ✅ Layer connectivity analysis (COMPLETE)
  - ✅ Connectivity trajectories (COMPLETE)
  - ✅ Word summaries with polysemy (COMPLETE)
  - ⚠️ Feature activations (EMPTY - extraction failed)

### Infrastructure
- **logs/**: Execution logs from GPU runs
- GPU-specific scripts and requirements
- Tmux-based job runners

## Key Achievements
- Successfully computed semantic connectivity for 5000 words across 27 layers
- Generated connectivity evolution trajectories
- Created comprehensive word-level statistics

## Issues Encountered
- Circuit feature extraction produced no results
- Discovered API mismatch with circuit-tracer
- Low/no feature activation for vocabulary words