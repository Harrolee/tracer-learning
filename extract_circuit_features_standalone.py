#!/usr/bin/env python3
"""
Standalone script to extract circuit features for words that already have connectivity analysis.
This is Part 2 of the unified analysis pipeline, extracted for separate execution.
Includes checkpointing to resume from interruptions.
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(f"{desc}...")
        return iterable
import torch
import time
import os
import numpy as np

# Import circuit tracer components
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import attribute

class CircuitFeatureExtractor:
    """Extract circuit features for words using Circuit Tracer."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """Initialize with model and tracer."""
        print(f"Loading model from {model_path}...")
        
        # Determine model name for circuit tracer
        # Check config to identify exact model version
        try:
            import json
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    print(f"Model config: hidden_size={config.get('hidden_size')}, layers={config.get('num_hidden_layers')}")
        except:
            pass
            
        if "gemma-2-2b" in model_path.lower() or model_path.endswith("gemma-2-2b"):
            model_name = "google/gemma-2-2b"
            transcoder_set = 'gemma'  # Try with gemma transcoders for gemma-2
        elif "gemma-2b" in model_path.lower() or model_path.endswith("gemma-2b"):
            # Original gemma-2b - use the newer gemma-2-2b instead
            print("Note: Original gemma-2b detected. Using gemma-2-2b for better compatibility.")
            model_name = "google/gemma-2-2b"
            transcoder_set = 'gemma'
        elif "gemma-7b" in model_path.lower():
            model_name = "google/gemma-7b"
            transcoder_set = 'gemma'
        else:
            # Default to using the path as-is
            model_name = model_path
            transcoder_set = 'gpt2'
            
        print(f"Using model name: {model_name} with transcoder set: {transcoder_set}")
            
        # Set device for loading
        # Always load on CPU first to avoid device conflicts, then move to target device
        self.model = ReplacementModel.from_pretrained(
            model_name,
            transcoder_set,
            torch_dtype=torch.float32,
            device_map=None  # Load on CPU first
        )
        
        # Move to target device
        if device != 'cpu':
            print(f"Moving model to {device}...")
            self.model = self.model.to(device)
        
        self.model.eval()
        self.device = device
        
        # Get tokenizer
        self.tokenizer = self.model.tokenizer
        
        print(f"Model loaded successfully on {device}")
        
    def extract_features_for_word(self, word: str, top_k: int = 10) -> Dict:
        """Extract circuit features for a single word."""
        try:
            # Create prompt with the word
            prompt = f" {word}"
            
            # Get attribution graph
            graph = attribute(
                prompt,
                self.model,
                batch_size=1,
                verbose=False,
                offload=None,  # Ensure no offloading which might cause shape issues
                max_n_logits=5,  # Reduce to see if it helps
                desired_logit_prob=0.9
            )
            
            # Extract features by layer
            word_features = {}
            
            # Debug info
            if graph.active_features is None:
                print(f"  No active features found for '{word}'")
            elif len(graph.active_features) == 0:
                print(f"  Empty active features for '{word}'")
            else:
                print(f"  Found {len(graph.active_features)} active features for '{word}'")
            
            # active_features is a tensor of shape (n_active_features, 3)
            # containing (layer, pos, feature_idx) for each active feature
            if graph.active_features is not None and len(graph.active_features) > 0:
                # Get adjacency matrix to find feature strengths
                adjacency = graph.adjacency_matrix
                n_features = len(graph.active_features)
                
                # Group features by layer
                for i, (layer, pos, feature_idx) in enumerate(graph.active_features):
                    layer = int(layer)
                    feature_idx = int(feature_idx)
                    
                    if layer not in word_features:
                        word_features[layer] = []
                    
                    # Get the attribution strength from adjacency matrix
                    # Features are the first n_features rows/cols in the adjacency matrix
                    # Logits are the last entries
                    n_logits = len(graph.logit_tokens) if hasattr(graph, 'logit_tokens') else 10
                    
                    # Get max attribution to any logit
                    if i < adjacency.shape[0] and adjacency.shape[0] > n_logits:
                        logit_attributions = adjacency[i, -n_logits:]
                        max_strength = float(torch.max(torch.abs(logit_attributions)))
                    else:
                        max_strength = 0.0
                    
                    if max_strength > 0:
                        word_features[layer].append({
                            'feature_id': feature_idx,
                            'activation_strength': max_strength
                        })
                
                # Sort and limit features per layer
                for layer in word_features:
                    word_features[layer] = sorted(
                        word_features[layer],
                        key=lambda x: x['activation_strength'],
                        reverse=True
                    )[:top_k]
            
            return word_features
            
        except Exception as e:
            print(f"Warning: Feature extraction failed for '{word}': {e}")
            # Log more details for debugging
            import traceback
            print(f"Full error trace: {traceback.format_exc()}")
            return {}
    
    def extract_features_for_words(
        self, 
        words: List[str],
        top_k_features: int = 10,
        checkpoint_file: Path = None,
        checkpoint_frequency: int = 100
    ) -> Dict[str, Dict]:
        """Extract circuit features for a list of words with checkpointing."""
        
        # Load existing checkpoint if available
        features_by_word = {}
        processed_words = set()
        
        if checkpoint_file and checkpoint_file.exists():
            print(f"Loading checkpoint from {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                features_by_word = checkpoint_data['features']
                processed_words = set(checkpoint_data['processed_words'])
                print(f"Resumed from checkpoint: {len(processed_words)} words already processed")
        
        # Filter out already processed words
        remaining_words = [w for w in words if w not in processed_words]
        
        if not remaining_words:
            print("All words already processed!")
            return features_by_word
        
        print(f"Processing {len(remaining_words)} remaining words...")
        
        for idx, word in enumerate(tqdm(remaining_words, desc="Extracting features")):
            word_features = self.extract_features_for_word(word, top_k=top_k_features)
            features_by_word[word] = word_features
            processed_words.add(word)
            
            # Save checkpoint periodically
            if checkpoint_file and (idx + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(checkpoint_file, features_by_word, processed_words)
                print(f"\nCheckpoint saved: {len(processed_words)} words processed")
        
        # Final checkpoint save
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, features_by_word, processed_words)
            print(f"Final checkpoint saved: {len(processed_words)} words processed")
        
        return features_by_word
    
    def _save_checkpoint(self, checkpoint_file: Path, features: Dict, processed_words: Set[str]):
        """Save checkpoint data."""
        checkpoint_data = {
            'features': features,
            'processed_words': list(processed_words),
            'timestamp': time.time()
        }
        
        # Write to temporary file first, then rename (atomic operation)
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Atomic rename
        temp_file.rename(checkpoint_file)
    
    def export_features_to_csv(
        self,
        features_by_word: Dict[str, Dict],
        output_file: Path,
        append_mode: bool = False
    ):
        """Export feature activations to CSV."""
        print(f"Exporting features to {output_file}")
        
        # If appending and file exists, read existing words to avoid duplicates
        existing_words = set()
        if append_mode and output_file.exists():
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_words.add(row['word'])
        
        mode = 'a' if append_mode and output_file.exists() else 'w'
        write_header = mode == 'w'
        
        with open(output_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'word', 'layer', 'feature_id', 'activation_strength'
            ])
            
            if write_header:
                writer.writeheader()
            
            for word, layers in features_by_word.items():
                if word in existing_words:
                    continue
                    
                for layer, features in layers.items():
                    for feature in features:
                        writer.writerow({
                            'word': word,
                            'layer': layer,
                            'feature_id': feature['feature_id'],
                            'activation_strength': round(feature['activation_strength'], 4)
                        })

def main():
    parser = argparse.ArgumentParser(description='Extract circuit features for analyzed words')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model directory')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory with connectivity analysis results')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output CSV file (default: results-dir/feature_activations.csv)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--top-k-features', type=int, default=10,
                        help='Number of top features to keep per layer')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Process words in batches')
    parser.add_argument('--checkpoint-frequency', type=int, default=100,
                        help='Save checkpoint every N words')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-file', type=str, default=None,
                        help='Checkpoint file path (default: results-dir/circuit_checkpoint.json)')
    parser.add_argument('--test-words', type=int, default=0,
                        help='Test with first N words only (0 = all words)')
    
    args = parser.parse_args()
    
    # Load words from existing analysis
    results_dir = Path(args.results_dir)
    words_file = results_dir / 'sampled_words.json'
    
    if not words_file.exists():
        print(f"Error: No sampled_words.json found in {results_dir}")
        return
    
    with open(words_file, 'r') as f:
        sampled_words = json.load(f)
    
    # Use test subset if requested
    if args.test_words > 0:
        sampled_words = sampled_words[:args.test_words]
        print(f"Testing with first {len(sampled_words)} words")
    else:
        print(f"Loaded {len(sampled_words)} words from previous analysis")
    
    # Set up checkpoint file
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else results_dir / 'circuit_checkpoint.json'
    if not args.resume and checkpoint_file.exists():
        print(f"Warning: Checkpoint file exists at {checkpoint_file}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() != 'y':
            print("Removing old checkpoint...")
            checkpoint_file.unlink()
    
    # Initialize extractor
    extractor = CircuitFeatureExtractor(args.model, args.device)
    
    # Process all words with checkpointing
    all_features = extractor.extract_features_for_words(
        sampled_words,
        top_k_features=args.top_k_features,
        checkpoint_file=checkpoint_file if args.resume or checkpoint_file.exists() else checkpoint_file,
        checkpoint_frequency=args.checkpoint_frequency
    )
    
    # Export results
    output_file = Path(args.output_file) if args.output_file else results_dir / 'feature_activations.csv'
    
    # Check if we should append or overwrite
    append_mode = args.resume and output_file.exists()
    extractor.export_features_to_csv(all_features, output_file, append_mode=append_mode)
    
    # Update metadata
    metadata_file = results_dir / 'analysis_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata['circuit_features'] = {
        'completed': True,
        'total_words_processed': len(all_features),
        'top_k_features': args.top_k_features,
        'checkpoint_used': args.resume,
        'test_mode': args.test_words > 0
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up checkpoint file if completed
    if checkpoint_file.exists() and not (args.test_words > 0):
        print("Removing checkpoint file after successful completion...")
        checkpoint_file.unlink()
    
    print(f"\n✨ Circuit feature extraction complete!")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    total_features = sum(
        sum(len(features) for features in word_features.values())
        for word_features in all_features.values()
    )
    words_with_features = sum(1 for features in all_features.values() if features and any(len(layer_features) > 0 for layer_features in features.values()))
    
    print(f"\nSummary:")
    print(f"  - Words processed: {len(all_features)}")
    print(f"  - Words with features: {words_with_features}")
    print(f"  - Total features extracted: {total_features}")
    
if __name__ == "__main__":
    main()