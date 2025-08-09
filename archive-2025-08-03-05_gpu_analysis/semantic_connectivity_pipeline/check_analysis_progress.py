#!/usr/bin/env python3
"""
Check progress of checkpointed analysis
"""

import json
import sys
from pathlib import Path
import pickle

def check_progress(checkpoint_dir: str = 'analysis_checkpoints'):
    """Check and display analysis progress."""
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ No checkpoint directory found at {checkpoint_dir}")
        return
    
    # Load progress metadata
    metadata_path = checkpoint_path / 'progress_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            progress = json.load(f)
    else:
        print(f"❌ No progress metadata found")
        return
    
    print(f"\n📊 Analysis Progress Report")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 50)
    
    # Word sampling
    print(f"\n1️⃣ Word Sampling:")
    if progress.get('sampled_words', False):
        words_path = checkpoint_path / 'sampled_words.pkl'
        if words_path.exists():
            with open(words_path, 'rb') as f:
                words = pickle.load(f)
            print(f"   ✅ Completed - {len(words)} words sampled")
        else:
            print(f"   ⚠️  Marked complete but file missing")
    else:
        print(f"   ❌ Not started")
    
    # Connectivity computation
    print(f"\n2️⃣ Connectivity Computation:")
    completed_layers = progress.get('connectivity_layers_completed', [])
    if completed_layers:
        print(f"   ✅ Completed layers: {sorted(completed_layers)}")
        
        # Check file sizes
        total_size = 0
        for layer in completed_layers:
            file_path = checkpoint_path / f'connectivity_layer_{layer}.pkl'
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"      Layer {layer}: {size_mb:.1f} MB")
        print(f"   Total size: {total_size:.1f} MB")
    else:
        print(f"   ❌ No layers completed")
    
    # Feature extraction
    print(f"\n3️⃣ Circuit Feature Extraction:")
    features_completed = progress.get('features_words_completed', [])
    if features_completed:
        print(f"   ✅ Completed: {len(features_completed)} words")
        
        # Show sample words
        if len(features_completed) > 10:
            print(f"   Sample words: {features_completed[:5]} ... {features_completed[-5:]}")
        else:
            print(f"   Words: {features_completed}")
        
        # Check file size
        features_path = checkpoint_path / 'features_by_word.pkl'
        if features_path.exists():
            size_mb = features_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.1f} MB")
    else:
        print(f"   ❌ No words completed")
    
    # CSV exports
    print(f"\n4️⃣ CSV Exports:")
    exports_completed = progress.get('exports_completed', [])
    expected_exports = ['word_summary', 'layer_connectivity', 'feature_activations', 'connectivity_trajectories']
    
    for export in expected_exports:
        if export in exports_completed:
            print(f"   ✅ {export}.csv")
        else:
            print(f"   ❌ {export}.csv")
    
    # Estimate completion
    print(f"\n📈 Overall Progress:")
    tasks_total = 4
    tasks_complete = 0
    
    if progress.get('sampled_words', False):
        tasks_complete += 0.25
    
    # Assume we need to analyze a certain number of layers (get from checkpoint if possible)
    # This is a rough estimate
    if completed_layers:
        # Guess total layers based on common values
        if max(completed_layers) <= 5:
            total_layers = 5
        elif max(completed_layers) <= 10:
            total_layers = 10
        else:
            total_layers = 27
        layer_progress = len(completed_layers) / total_layers
        tasks_complete += 0.25 * layer_progress
    
    if features_completed and words_path.exists():
        with open(words_path, 'rb') as f:
            total_words = len(pickle.load(f))
        feature_progress = len(features_completed) / total_words
        tasks_complete += 0.25 * feature_progress
    
    export_progress = len(exports_completed) / len(expected_exports)
    tasks_complete += 0.25 * export_progress
    
    percentage = tasks_complete * 100
    print(f"   Estimated completion: {percentage:.1f}%")
    
    # Progress bar
    bar_length = 40
    filled = int(bar_length * tasks_complete)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"   [{bar}] {percentage:.1f}%")
    
    print("\n✨ To resume analysis, run the same command again.")
    print("   The pipeline will automatically continue from checkpoints.")

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'analysis_checkpoints'
    check_progress(checkpoint_dir)