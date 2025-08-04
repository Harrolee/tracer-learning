#!/usr/bin/env python3
"""
Quick test of unified pipeline using existing 5-layer embeddings
Perfect for testing while full 27-layer computation runs
"""

import argparse
import time
import os
from pathlib import Path

import torch

def run_test_analysis():
    """Run a quick test with 5-layer embeddings and fewer words."""
    
    print("🧪 TEST MODE: Unified Analysis with 5 Layers")
    print("=" * 50)
    
    # Configuration for test
    model_path = os.getenv('MODEL_PATH', '/home/ubuntu/tracer-learning/models/gemma-2b')
    dict_embeddings = 'dictionary_embeddings_gemma_embeddings'  # The 5-layer version
    output_dir = 'test_results_5layers_100words'
    num_words = 100  # Small number for quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dictionary: {dict_embeddings}")
    print(f"  Output: {output_dir}")
    print(f"  Words: {num_words}")
    print(f"  Device: {device}")
    print()
    
    # Check if dictionary exists
    dict_path = Path(dict_embeddings)
    if not dict_path.exists():
        print(f"❌ Dictionary embeddings not found at {dict_embeddings}")
        print("Available directories:")
        for d in Path('.').glob('dictionary_embeddings*'):
            if d.is_dir():
                print(f"  - {d}")
        return
    
    # Check what layers are available
    layer_files = list(dict_path.glob('embeddings_layer_*.pkl'))
    available_layers = sorted([int(f.stem.split('_')[-1]) for f in layer_files])
    print(f"✅ Found embeddings for {len(available_layers)} layers: {available_layers}")
    
    if len(available_layers) != 5:
        print(f"⚠️  Expected 5 layers, found {len(available_layers)}")
    
    # Build command
    cmd = f"""python semantic_connectivity_pipeline/unified_analysis_pipeline.py \\
    --model "{model_path}" \\
    --dictionary-embeddings "{dict_embeddings}" \\
    --output-dir "{output_dir}" \\
    --num-words {num_words} \\
    --sampling-strategy extreme_contrast \\
    --connectivity-threshold 0.7 \\
    --device {device} \\
    --download-nltk"""
    
    print("\nCommand to run:")
    print(cmd)
    print()
    
    # Execute
    start_time = time.time()
    print("Starting analysis...")
    print("-" * 50)
    
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    elapsed = time.time() - start_time
    print("-" * 50)
    
    if result.returncode == 0:
        print(f"\n✅ Test completed successfully in {elapsed:.1f} seconds!")
        
        # Check output files
        output_path = Path(output_dir)
        if output_path.exists():
            print("\nOutput files created:")
            for csv_file in output_path.glob('*.csv'):
                size = csv_file.stat().st_size / 1024  # KB
                with open(csv_file) as f:
                    lines = sum(1 for _ in f) - 1  # Subtract header
                print(f"  - {csv_file.name}: {lines} rows, {size:.1f} KB")
            
            # Show sample results
            word_summary = output_path / 'word_summary.csv'
            if word_summary.exists():
                print("\nSample word summary (first 5 words):")
                with open(word_summary) as f:
                    for i, line in enumerate(f):
                        if i <= 5:  # Header + 5 rows
                            print(f"  {line.strip()}")
    else:
        print(f"\n❌ Test failed with return code {result.returncode}")
        print("Check the error messages above for details")
    
    print(f"\n📊 Performance: {num_words} words in {elapsed:.1f} seconds")
    print(f"   Average: {elapsed/num_words:.2f} seconds per word")
    
    # Estimate full run time
    full_words = 5000
    estimated_time = (elapsed / num_words) * full_words
    print(f"\n⏱️  Estimated time for {full_words} words: {estimated_time/60:.1f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test unified pipeline with 5-layer embeddings")
    parser.add_argument('--run', action='store_true', help='Actually run the test (default is dry run)')
    
    args = parser.parse_args()
    
    if args.run:
        run_test_analysis()
    else:
        print("Dry run mode. Add --run to actually execute the test")
        print("\nThis script will:")
        print("1. Use your existing 5-layer dictionary embeddings")
        print("2. Process only 100 words for quick testing")
        print("3. Run the full unified pipeline (connectivity + circuit features)")
        print("4. Show performance metrics and time estimates")
        print("\nRun with: python test_unified_5layers.py --run")