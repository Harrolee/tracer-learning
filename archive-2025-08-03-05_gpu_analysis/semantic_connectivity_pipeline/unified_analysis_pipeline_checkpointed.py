#!/usr/bin/env python3
"""
Unified pipeline for semantic connectivity and circuit complexity analysis
WITH CHECKPOINTING - can resume from interruptions
"""

import argparse
import json
import pickle
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import nltk
from nltk.corpus import wordnet

# Import circuit tracer - required, no fallback
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import attribute


class CheckpointManager:
    """Manages checkpoints for resumable computation."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, name: str, data: any):
        """Save a checkpoint."""
        path = self.checkpoint_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_checkpoint(self, name: str) -> Optional[any]:
        """Load a checkpoint if it exists."""
        path = self.checkpoint_dir / f"{name}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def checkpoint_exists(self, name: str) -> bool:
        """Check if checkpoint exists."""
        return (self.checkpoint_dir / f"{name}.pkl").exists()
    
    def get_progress(self) -> Dict:
        """Get overall progress from metadata."""
        metadata_path = self.checkpoint_dir / "progress_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {
            'sampled_words': False,
            'connectivity_layers_completed': [],
            'features_words_completed': [],
            'exports_completed': []
        }
    
    def update_progress(self, updates: Dict):
        """Update progress metadata."""
        progress = self.get_progress()
        progress.update(updates)
        metadata_path = self.checkpoint_dir / "progress_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(progress, f, indent=2)


class UnifiedAnalysisPipeline:
    """Main pipeline class for connectivity and feature analysis."""
    
    def __init__(
        self,
        model_path: str,
        dictionary_embeddings_dir: Path,
        device: str = 'cuda',
        checkpoint_dir: Optional[Path] = None
    ):
        self.model_path = model_path
        self.dict_embeddings_dir = dictionary_embeddings_dir
        self.device = device
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir or Path('analysis_checkpoints')
        )
        
        # Load model and tokenizer for embeddings
        print(f"🤖 Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        
        # Note: Circuit tracer will load its own TransformerLens version of the model
        
        # Load dictionary embeddings metadata
        self.load_dictionary_metadata()
        
        # Circuit tracer setup (required)
        try:
            # Determine model and transcoder set
            # Check path first
            if 'gemma' in model_path.lower():
                tl_model_name = 'google/gemma-2-2b'
                transcoder_set = 'gemma'
                print(f"   Detected Gemma model from path, using transcoder set: {transcoder_set}")
            elif 'llama' in model_path.lower():
                tl_model_name = 'meta-llama/Llama-3.2-1B'
                transcoder_set = 'llama'
                print(f"   Detected Llama model from path, using transcoder set: {transcoder_set}")
            else:
                # Check model config to determine type
                import json
                from pathlib import Path
                config_path = Path(model_path) / 'config.json'
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    model_type = config.get('model_type', '').lower()
                    model_name = config.get('_name_or_path', '').lower()
                    
                    if 'gemma' in model_type or 'gemma' in model_name:
                        tl_model_name = 'google/gemma-2-2b'
                        transcoder_set = 'gemma'
                        print(f"   Detected Gemma model from config, using transcoder set: {transcoder_set}")
                    elif 'llama' in model_type or 'llama' in model_name:
                        tl_model_name = 'meta-llama/Llama-3.2-1B'
                        transcoder_set = 'llama'
                        print(f"   Detected Llama model from config, using transcoder set: {transcoder_set}")
                    else:
                        raise ValueError(f"Could not determine model type from config. model_type={model_type}, path={model_path}")
                else:
                    raise ValueError(f"No config.json found at {config_path}. Cannot determine model type.")
            
            # Create ReplacementModel using the proper method
            self.tracer_model = ReplacementModel.from_pretrained(
                model_name=tl_model_name,
                transcoder_set=transcoder_set,
                device=self.device,
                dtype=torch.float32 if self.device == 'cpu' else torch.bfloat16
            )
            self.circuit_tracer_available = True
            print("✅ Circuit tracer initialized successfully")
        except Exception as e:
            print(f"❌ FATAL: Circuit tracer initialization failed: {e}")
            print("   Circuit tracer is required for this analysis.")
            print("   Please ensure circuit-tracer is properly installed.")
            raise RuntimeError(f"Circuit tracer initialization failed: {e}")
        
    def load_dictionary_metadata(self):
        """Load metadata about precomputed dictionary embeddings."""
        metadata_path = self.dict_embeddings_dir / 'embedding_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Dictionary embeddings not found at {self.dict_embeddings_dir}. "
                "Run precompute_dictionary_embeddings.py first!"
            )
        
        with open(metadata_path, 'r') as f:
            self.dict_metadata = json.load(f)
        
        self.available_layers = self.dict_metadata['layers']
        print(f"✅ Dictionary embeddings available for layers: {self.available_layers}")
    
    def sample_words_from_wordnet(
        self,
        strategy: str = 'extreme_contrast',
        total_words: int = 5000,
        save_path: Optional[Path] = None
    ) -> Tuple[List[str], Dict[str, int]]:
        """Sample words from WordNet according to specified strategy."""
        
        # Check if we have cached results
        cached_words = self.checkpoint_mgr.load_checkpoint('sampled_words')
        cached_polysemy = self.checkpoint_mgr.load_checkpoint('sampled_polysemy')
        
        if cached_words and cached_polysemy:
            print(f"✅ Loaded {len(cached_words)} cached sampled words")
            return cached_words, cached_polysemy
        
        print(f"\n📚 Sampling {total_words} words using '{strategy}' strategy")
        
        # Calculate polysemy scores
        polysemy_scores = defaultdict(int)
        for synset in tqdm(wordnet.all_synsets(), desc="Computing polysemy"):
            for lemma in synset.lemmas():
                word = lemma.name().replace('_', ' ').lower()
                if word.isalpha() and len(word) > 1:
                    polysemy_scores[word] += 1
        
        polysemy_scores = dict(polysemy_scores)
        print(f"✅ Computed polysemy for {len(polysemy_scores):,} words")
        
        # Sample according to strategy
        if strategy == 'extreme_contrast':
            # Half high-polysemy, half monosemous
            sorted_words = sorted(polysemy_scores.items(), key=lambda x: x[1], reverse=True)
            
            high_poly_cutoff = int(len(sorted_words) * 0.25)
            high_poly_words = [w for w, _ in sorted_words[:high_poly_cutoff]]
            monosemous_words = [w for w, s in sorted_words if s == 1]
            
            import random
            half = total_words // 2
            sampled_words = (
                random.sample(high_poly_words, min(half, len(high_poly_words))) +
                random.sample(monosemous_words, min(half, len(monosemous_words)))
            )
            
        elif strategy == 'stratified':
            # Sample evenly across polysemy levels
            by_polysemy = defaultdict(list)
            for word, score in polysemy_scores.items():
                if score == 1:
                    by_polysemy['mono'].append(word)
                elif score <= 3:
                    by_polysemy['low'].append(word)
                elif score <= 10:
                    by_polysemy['medium'].append(word)
                else:
                    by_polysemy['high'].append(word)
            
            import random
            sampled_words = []
            per_category = total_words // len(by_polysemy)
            
            for category, words in by_polysemy.items():
                sampled = random.sample(words, min(per_category, len(words)))
                sampled_words.extend(sampled)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Create polysemy lookup for sampled words
        sampled_polysemy = {w: polysemy_scores[w] for w in sampled_words}
        
        # Save checkpoint
        self.checkpoint_mgr.save_checkpoint('sampled_words', sampled_words)
        self.checkpoint_mgr.save_checkpoint('sampled_polysemy', sampled_polysemy)
        self.checkpoint_mgr.update_progress({'sampled_words': True})
        
        # Save if requested
        if save_path:
            with open(save_path / 'sampled_words.json', 'w') as f:
                json.dump(sampled_words, f, indent=2)
            with open(save_path / 'sampled_polysemy.json', 'w') as f:
                json.dump(sampled_polysemy, f, indent=2)
            print(f"✅ Saved {len(sampled_words)} sampled words")
        
        return sampled_words, sampled_polysemy
    
    def load_dictionary_embeddings_for_layer(self, layer: int) -> Dict[str, np.ndarray]:
        """Load precomputed dictionary embeddings for a specific layer."""
        path = self.dict_embeddings_dir / f'embeddings_layer_{layer}.pkl'
        if not path.exists():
            raise FileNotFoundError(f"Embeddings for layer {layer} not found")
        
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
        
        return embeddings
    
    def compute_connectivity_fast(
        self,
        target_words: List[str],
        layer: int,
        threshold: float = 0.7,
        batch_size: int = 100
    ) -> Dict[str, Dict]:
        """Compute connectivity using precomputed dictionary embeddings."""
        
        # Check for checkpoint
        checkpoint_name = f'connectivity_layer_{layer}'
        cached_result = self.checkpoint_mgr.load_checkpoint(checkpoint_name)
        if cached_result:
            print(f"✅ Loaded cached connectivity for layer {layer}")
            return cached_result
        
        print(f"\n🔄 Computing connectivity for layer {layer}")
        
        # Load dictionary embeddings for this layer
        dict_embeddings = self.load_dictionary_embeddings_for_layer(layer)
        dict_words = list(dict_embeddings.keys())
        dict_matrix = np.vstack([dict_embeddings[w] for w in dict_words])
        
        # Normalize for cosine similarity
        dict_norms = np.linalg.norm(dict_matrix, axis=1, keepdims=True)
        dict_matrix_norm = dict_matrix / (dict_norms + 1e-8)
        
        connectivity_results = {}
        
        # Process target words in batches
        for i in tqdm(range(0, len(target_words), batch_size), desc=f"Layer {layer}"):
            batch_words = target_words[i:i + batch_size]
            
            # Get embeddings for batch
            batch_embeddings = []
            valid_words = []
            
            for word in batch_words:
                if word in dict_embeddings:
                    batch_embeddings.append(dict_embeddings[word])
                    valid_words.append(word)
            
            if not batch_embeddings:
                continue
            
            # Compute similarities
            batch_matrix = np.vstack(batch_embeddings)
            batch_norms = np.linalg.norm(batch_matrix, axis=1, keepdims=True)
            batch_matrix_norm = batch_matrix / (batch_norms + 1e-8)
            
            # Cosine similarity: batch_words x dict_words
            similarities = np.dot(batch_matrix_norm, dict_matrix_norm.T)
            
            # Count neighbors above threshold
            for word_idx, word in enumerate(valid_words):
                word_sims = similarities[word_idx]
                
                # Exclude self-similarity
                mask = np.array([dict_words[j] != word for j in range(len(dict_words))])
                word_sims = word_sims[mask]
                neighbor_words = [dict_words[j] for j in range(len(dict_words)) if mask[j]]
                
                # Find high-similarity neighbors
                high_sim_mask = word_sims > threshold
                connectivity_count = np.sum(high_sim_mask)
                
                # Get top neighbors
                top_indices = np.argsort(word_sims)[-10:][::-1]
                top_neighbors = [(neighbor_words[idx], word_sims[idx]) for idx in top_indices]
                
                connectivity_results[word] = {
                    'connectivity_count': int(connectivity_count),
                    'mean_similarity': float(np.mean(word_sims)),
                    'max_similarity': float(np.max(word_sims)) if len(word_sims) > 0 else 0.0,
                    'top_neighbors': top_neighbors
                }
        
        # Save checkpoint
        self.checkpoint_mgr.save_checkpoint(checkpoint_name, connectivity_results)
        
        # Update progress
        progress = self.checkpoint_mgr.get_progress()
        completed_layers = progress.get('connectivity_layers_completed', [])
        if layer not in completed_layers:
            completed_layers.append(layer)
            self.checkpoint_mgr.update_progress({'connectivity_layers_completed': completed_layers})
        
        return connectivity_results
    
    def extract_circuit_features(
        self,
        words: List[str],
        threshold: float = 0.1,
        batch_size: int = 10
    ) -> Dict[str, Dict]:
        """Extract feature activations using circuit tracer (required)."""
        
        if not self.circuit_tracer_available:
            raise RuntimeError("Circuit tracer is not available but is required for analysis")
        
        print(f"\n🧠 Extracting circuit features for {len(words)} words")
        
        # Load any cached results
        features_by_word = self.checkpoint_mgr.load_checkpoint('features_by_word') or {}
        completed_words = set(features_by_word.keys())
        remaining_words = [w for w in words if w not in completed_words]
        
        if completed_words:
            print(f"✅ Found cached features for {len(completed_words)} words")
        
        if not remaining_words:
            return features_by_word
        
        print(f"📊 Processing {len(remaining_words)} remaining words")
        
        # Process in batches for better checkpointing
        for batch_start in tqdm(range(0, len(remaining_words), batch_size), desc="Feature extraction batches"):
            batch_words = remaining_words[batch_start:batch_start + batch_size]
            
            for word in tqdm(batch_words, desc="Words in batch", leave=False):
                try:
                    # Tokenize
                    inputs = self.tokenizer(word, return_tensors='pt').to(self.device)
                    
                    # Get attributions
                    attributions = attribute(
                        self.tracer_model,
                        inputs['input_ids'],
                        method='integrated_gradients'
                    )
                    
                    word_features = {}
                    
                    # Process each layer
                    for layer_idx in range(self.model.config.num_hidden_layers + 1):
                        layer_features = []
                        
                        # Get activations for this layer
                        layer_attrs = attributions.get(f'layer_{layer_idx}', {})
                        
                        for feat_id, strength in layer_attrs.items():
                            if abs(strength) > threshold:
                                layer_features.append({
                                    'feature_id': f'L{layer_idx}_F{feat_id}',
                                    'activation_strength': abs(strength),
                                    'feature_type': 'unknown'  # Would need feature dictionary
                                })
                        
                        word_features[layer_idx] = sorted(
                            layer_features, 
                            key=lambda x: x['activation_strength'], 
                            reverse=True
                        )
                    
                    features_by_word[word] = word_features
                    
                except Exception as e:
                    print(f"Warning: Feature extraction failed for '{word}': {e}")
                    features_by_word[word] = {}
            
            # Save checkpoint after each batch
            self.checkpoint_mgr.save_checkpoint('features_by_word', features_by_word)
            
            # Update progress
            progress = self.checkpoint_mgr.get_progress()
            progress['features_words_completed'] = list(features_by_word.keys())
            self.checkpoint_mgr.update_progress(progress)
        
        return features_by_word
    
    def export_results_to_csv(
        self,
        output_dir: Path,
        sampled_words: List[str],
        polysemy_scores: Dict[str, int],
        connectivity_by_layer: Dict[int, Dict],
        features_by_word: Dict[str, Dict]
    ):
        """Export all results to CSV files."""
        
        print(f"\n📊 Exporting results to {output_dir}")
        
        progress = self.checkpoint_mgr.get_progress()
        completed_exports = progress.get('exports_completed', [])
        
        # 1. Word summary CSV
        if 'word_summary' not in completed_exports:
            with open(output_dir / 'word_summary.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'word', 'polysemy_score', 'total_features', 'total_connectivity'
                ])
                writer.writeheader()
                
                for word in sampled_words:
                    total_features = sum(
                        len(features_by_word.get(word, {}).get(layer, []))
                        for layer in self.available_layers
                    )
                    total_connectivity = sum(
                        connectivity_by_layer.get(layer, {}).get(word, {}).get('connectivity_count', 0)
                        for layer in self.available_layers
                    )
                    
                    writer.writerow({
                        'word': word,
                        'polysemy_score': polysemy_scores.get(word, 0),
                        'total_features': total_features,
                        'total_connectivity': total_connectivity
                    })
            completed_exports.append('word_summary')
        
        # 2. Layer connectivity CSV
        if 'layer_connectivity' not in completed_exports:
            with open(output_dir / 'layer_connectivity.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'word', 'layer', 'connectivity_count', 'mean_similarity', 'max_similarity'
                ])
                writer.writeheader()
                
                for layer in sorted(connectivity_by_layer.keys()):
                    for word, metrics in connectivity_by_layer[layer].items():
                        writer.writerow({
                            'word': word,
                            'layer': layer,
                            'connectivity_count': metrics['connectivity_count'],
                            'mean_similarity': round(metrics['mean_similarity'], 4),
                            'max_similarity': round(metrics['max_similarity'], 4)
                        })
            completed_exports.append('layer_connectivity')
        
        # 3. Feature activations CSV
        if 'feature_activations' not in completed_exports:
            with open(output_dir / 'feature_activations.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'word', 'layer', 'feature_id', 'activation_strength'
                ])
                writer.writeheader()
                
                for word, layers in features_by_word.items():
                    for layer, features in layers.items():
                        for feature in features:
                            writer.writerow({
                                'word': word,
                                'layer': layer,
                                'feature_id': feature['feature_id'],
                                'activation_strength': round(feature['activation_strength'], 4)
                            })
            completed_exports.append('feature_activations')
        
        # 4. Connectivity trajectories (wide format)
        if 'connectivity_trajectories' not in completed_exports:
            layer_columns = [f'layer_{l}_connectivity' for l in self.available_layers]
            with open(output_dir / 'connectivity_trajectories.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['word'] + layer_columns)
                writer.writeheader()
                
                for word in sampled_words:
                    row = {'word': word}
                    for layer in self.available_layers:
                        conn = connectivity_by_layer.get(layer, {}).get(word, {}).get('connectivity_count', 0)
                        row[f'layer_{layer}_connectivity'] = conn
                    writer.writerow(row)
            completed_exports.append('connectivity_trajectories')
        
        # Update progress
        self.checkpoint_mgr.update_progress({'exports_completed': completed_exports})
        print("✅ Export complete!")
    
    def run_full_analysis(
        self,
        output_dir: Path,
        sampling_strategy: str = 'extreme_contrast',
        total_words: int = 5000,
        connectivity_threshold: float = 0.7,
        feature_threshold: float = 0.1,
        connectivity_batch_size: int = 100,
        feature_batch_size: int = 10
    ):
        """Run the complete analysis pipeline with checkpointing."""
        
        start_time = time.time()
        
        # Check existing progress
        progress = self.checkpoint_mgr.get_progress()
        print("\n📋 Checking existing progress:")
        print(f"  - Words sampled: {progress.get('sampled_words', False)}")
        print(f"  - Connectivity layers completed: {len(progress.get('connectivity_layers_completed', []))}/{len(self.available_layers)}")
        print(f"  - Feature words completed: {len(progress.get('features_words_completed', []))}")
        print(f"  - Exports completed: {progress.get('exports_completed', [])}")
        
        # Step 1: Sample words
        sampled_words, polysemy_scores = self.sample_words_from_wordnet(
            sampling_strategy, total_words, output_dir
        )
        
        # Step 2: Compute connectivity for each layer
        connectivity_by_layer = {}
        for layer in self.available_layers:
            connectivity_by_layer[layer] = self.compute_connectivity_fast(
                sampled_words, layer, connectivity_threshold, connectivity_batch_size
            )
        
        # Step 3: Extract circuit features
        features_by_word = self.extract_circuit_features(
            sampled_words, feature_threshold, feature_batch_size
        )
        
        # Step 4: Export to CSV
        self.export_results_to_csv(
            output_dir, sampled_words, polysemy_scores,
            connectivity_by_layer, features_by_word
        )
        
        # Save metadata
        metadata = {
            'model': self.model_path,
            'dictionary_embeddings': str(self.dict_embeddings_dir),
            'sampling_strategy': sampling_strategy,
            'total_words': len(sampled_words),
            'layers_analyzed': self.available_layers,
            'connectivity_threshold': connectivity_threshold,
            'feature_threshold': feature_threshold,
            'total_time': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint_dir': str(self.checkpoint_mgr.checkpoint_dir)
        }
        
        with open(output_dir / 'analysis_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✨ Analysis complete in {metadata['total_time']:.1f} seconds!")
        print(f"Results saved to {output_dir}")
        print(f"Checkpoints saved to {self.checkpoint_mgr.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for connectivity and circuit analysis WITH CHECKPOINTING"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/Users/lee/fun/learningSlice/models/gemma-2b',
        help='Model path'
    )
    parser.add_argument(
        '--dictionary-embeddings',
        type=str,
        required=True,
        help='Directory containing precomputed dictionary embeddings'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='unified_analysis_results',
        help='Output directory'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='analysis_checkpoints',
        help='Directory for checkpoints (enables resuming)'
    )
    parser.add_argument(
        '--sampling-strategy',
        type=str,
        default='extreme_contrast',
        choices=['extreme_contrast', 'stratified'],
        help='Word sampling strategy'
    )
    parser.add_argument(
        '--num-words',
        type=int,
        default=5000,
        help='Number of words to sample'
    )
    parser.add_argument(
        '--connectivity-threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for connectivity'
    )
    parser.add_argument(
        '--connectivity-batch-size',
        type=int,
        default=100,
        help='Batch size for connectivity computation'
    )
    parser.add_argument(
        '--feature-batch-size',
        type=int,
        default=10,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--download-nltk',
        action='store_true',
        help='Download required NLTK data'
    )
    parser.add_argument(
        '--clear-checkpoints',
        action='store_true',
        help='Clear existing checkpoints and start fresh'
    )
    
    args = parser.parse_args()
    
    if args.download_nltk:
        print("📥 Downloading NLTK data...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Handle checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.clear_checkpoints and checkpoint_dir.exists():
        print(f"🗑️  Clearing checkpoints in {checkpoint_dir}")
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Initialize pipeline
    pipeline = UnifiedAnalysisPipeline(
        args.model,
        Path(args.dictionary_embeddings),
        args.device,
        checkpoint_dir
    )
    
    # Run analysis
    pipeline.run_full_analysis(
        output_dir,
        args.sampling_strategy,
        args.num_words,
        args.connectivity_threshold,
        connectivity_batch_size=args.connectivity_batch_size,
        feature_batch_size=args.feature_batch_size
    )


if __name__ == "__main__":
    main()