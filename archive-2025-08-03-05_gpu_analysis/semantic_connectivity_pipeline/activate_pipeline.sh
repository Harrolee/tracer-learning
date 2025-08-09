#!/bin/bash
# Activate pipeline environment

source venv/bin/activate
echo "Pipeline environment activated!"
echo
echo "Quick start:"
echo "1. First time only - precompute dictionary embeddings:"
echo "   python precompute_dictionary_embeddings.py --model <model_path> --output-dir dictionary_embeddings"
echo
echo "2. Run analysis:"
echo "   python unified_analysis_pipeline.py --model <model_path> --dictionary-embeddings dictionary_embeddings"
echo
