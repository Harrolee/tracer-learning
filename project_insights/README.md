# Project Insights & Documentation

## Overview
Cross-session documentation and research insights that guide the entire project.

## Core Documents

### Research Plans
- **no-nonsense_research_plan.md**: Detailed research methodology
  - Hypothesis: Connectivity evolution predicts circuit complexity
  - Complete statistical analysis plan
  - Implementation timeline
  
- **abstract_research_plan.md**: High-level research vision
- **semantic_complexity_analysis.md**: Theoretical framework

### Project Documentation
- **CLAUDE.md**: AI assistant instructions and project state
  - Model locations
  - Analysis status
  - Key commands
  
- **README.md**: Project overview

## Key Insights

### Research Hypothesis
Words with dynamic connectivity patterns (high variance across layers) participate in more complex circuits than words with stable patterns.

### Methodology Innovation
- Layer-wise analysis instead of single-layer embeddings
- Connectivity evolution as complexity predictor
- Polysemy-based vocabulary sampling

### Current Status
- ✅ Part 1: Semantic connectivity analysis (COMPLETE - 5000 words)
- 🔄 Part 2: Circuit feature extraction (READY FOR GPU)
- ⏳ Part 3: Correlation analysis (PENDING)

## Lessons Learned
1. Circuit-tracer doesn't show direct feature→logit connections
2. Model downloads should be handled by libraries, not scripts
3. Checkpointing is essential for long GPU runs
4. Tmux-based monitoring crucial for overnight jobs