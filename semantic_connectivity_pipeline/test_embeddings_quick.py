#!/usr/bin/env python3
"""
Quick test version - computes embeddings for 40 words at ALL layers
Perfect for debugging and catching issues before overnight runs
"""

import argparse
import json
import pickle
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi, upload_file, create_repo
from datasets import Dataset

import nltk
from nltk.corpus import wordnet


def get_test_words() -> List[str]:
    """Get 40 diverse test words for debugging."""
    
    test_words = [
        # Simple nouns
        "cat", "dog", "tree", "house", "book",
        # Abstract concepts
        "love", "freedom", "justice", "thought", "time",
        # Polysemous words
        "bank", "run", "set", "light", "spring",
        # Technical terms
        "computer", "algorithm", "network", "database", "function",
        # Verbs
        "walk", "think", "create", "understand", "analyze",
        # Adjectives
        "happy", "complex", "beautiful", "important", "difficult",
        # Less common
        "serendipity", "ephemeral", "ubiquitous", "paradigm", "synergy",
        # Very simple
        "a", "the", "is", "and", "but"
    ]
    
    return test_words


def compute_embeddings_batch(
    words: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    layer: int,
    device: str,
    batch_size: int = 8
) -> Dict[str, np.ndarray]:
    """Compute embeddings for a batch of words at a specific layer."""
    
    embeddings = {}
    
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        
        try:
            # Tokenize batch
            tokens = tokenizer(
                batch_words, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True)
                
                # Extract embeddings from specific layer
                layer_output = outputs.hidden_states[layer]
                
                # Mean pooling over tokens for each word
                attention_mask = tokens['attention_mask']
                
                for word_idx, word in enumerate(batch_words):
                    mask = attention_mask[word_idx]
                    word_embedding = layer_output[word_idx]
                    
                    if mask.sum() > 0:
                        # Weighted mean using attention mask
                        masked_embedding = word_embedding * mask.unsqueeze(-1)
                        mean_embedding = masked_embedding.sum(dim=0) / mask.sum()
                    else:
                        mean_embedding = torch.zeros(model.config.hidden_size).to(device)
                    
                    # Store as numpy array to save memory
                    embeddings[word] = mean_embedding.cpu().numpy().astype(np.float16)
                    
        except Exception as e:
            print(f"⚠️  Failed to process batch starting with '{batch_words[0]}': {e}")
            continue
    
    return embeddings


def run_test_embeddings(
    model_path: str,
    output_dir: Path,
    device: str = 'cpu',
    batch_size: int = 8,
    upload_to_hf: bool = False,
    hf_repo_id: Optional[str] = None,
    hf_token: Optional[str] = None
):
    """Run quick test with 40 words at all layers."""
    
    print("🧪 QUICK TEST MODE - 40 words, all layers")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Get test words
    word_list = get_test_words()
    print(f"\n📝 Test words ({len(word_list)}):")
    for i in range(0, len(word_list), 10):
        print(f"  {', '.join(word_list[i:i+10])}")
    
    # Load model
    print(f"\n🤖 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers + 1  # Include embedding layer
    print(f"✅ Model loaded: {num_layers} layers total")
    
    # Initialize HF if uploading
    if upload_to_hf:
        if not hf_repo_id:
            raise ValueError("HuggingFace repo ID required for upload")
        
        api = HfApi(token=hf_token)
        try:
            api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
            print(f"☁️  HF repo ready: {hf_repo_id}")
        except Exception as e:
            print(f"⚠️  Could not create repo: {e}")
    
    # Process ALL layers
    all_embeddings = {}
    layer_stats = {}
    
    print(f"\n🔄 Computing embeddings for {num_layers} layers...")
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        layer_embeddings = compute_embeddings_batch(
            word_list, model, tokenizer, layer_idx, device, batch_size
        )
        
        all_embeddings[f'layer_{layer_idx}'] = layer_embeddings
        
        # Compute statistics for this layer
        embeddings_array = np.vstack([layer_embeddings[w] for w in word_list])
        layer_stats[layer_idx] = {
            'mean': float(np.mean(embeddings_array)),
            'std': float(np.std(embeddings_array)),
            'min': float(np.min(embeddings_array)),
            'max': float(np.max(embeddings_array)),
            'shape': embeddings_array.shape
        }
        
        # Save layer embeddings
        layer_path = output_dir / f'embeddings_layer_{layer_idx}.pkl'
        with open(layer_path, 'wb') as f:
            pickle.dump(layer_embeddings, f)
    
    print(f"✅ Computed embeddings for all {num_layers} layers")
    
    # Save metadata
    metadata = {
        'test_mode': True,
        'model': model_path,
        'model_name': model_path.split('/')[-1],
        'total_words': len(word_list),
        'words': word_list,
        'num_layers': num_layers,
        'layers': list(range(num_layers)),
        'embedding_dim': model.config.hidden_size,
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'layer_statistics': layer_stats
    }
    
    with open(output_dir / 'test_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print statistics
    print("\n📊 Layer Statistics:")
    print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    for layer_idx, stats in layer_stats.items():
        print(f"{layer_idx:<8} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}")
    
    # Verify embeddings
    print("\n🔍 Verification:")
    for layer_idx in [0, num_layers//2, num_layers-1]:  # Check first, middle, last
        layer_path = output_dir / f'embeddings_layer_{layer_idx}.pkl'
        if layer_path.exists():
            size_kb = layer_path.stat().st_size / 1024
            with open(layer_path, 'rb') as f:
                emb = pickle.load(f)
            print(f"  Layer {layer_idx}: {size_kb:.1f} KB, {len(emb)} words")
    
    # Upload to HuggingFace if requested
    if upload_to_hf:
        print(f"\n☁️  Uploading to HuggingFace...")
        
        try:
            # Create dataset records
            records = []
            for word in word_list:
                record = {'word': word}
                for layer_idx in range(num_layers):
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in all_embeddings and word in all_embeddings[layer_key]:
                        embedding = all_embeddings[layer_key][word].tolist()
                        record[f'embedding_layer_{layer_idx}'] = embedding
                records.append(record)
            
            # Create and push dataset
            dataset = Dataset.from_list(records)
            
            # Save locally first
            dataset_path = output_dir / 'hf_dataset_test'
            dataset.save_to_disk(str(dataset_path))
            
            # Push to hub
            dataset.push_to_hub(
                repo_id=hf_repo_id,
                token=hf_token,
                commit_message=f"Test embeddings - {len(word_list)} words, {num_layers} layers"
            )
            
            # Upload metadata
            upload_file(
                path_or_fileobj=str(output_dir / 'test_metadata.json'),
                path_in_repo='test_metadata.json',
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=hf_token
            )
            
            print(f"✅ Uploaded to: https://huggingface.co/datasets/{hf_repo_id}")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    
    print(f"\n✨ Test complete! Results in: {output_dir}")
    print(f"Total time: {metadata['timestamp']}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test: 40 words, all layers"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/Users/lee/fun/learningSlice/models/gemma-2b',
        help='Model path or name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_embeddings_40',
        help='Directory to save embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps' if torch.backends.mps.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    
    # HuggingFace upload arguments
    parser.add_argument(
        '--upload-to-hf',
        action='store_true',
        help='Upload embeddings to HuggingFace dataset hub'
    )
    parser.add_argument(
        '--hf-repo-id',
        type=str,
        help='HuggingFace repo ID (e.g., username/test-embeddings)'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=os.getenv('HF_TOKEN'),
        help='HuggingFace API token (or set HF_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Validate HF settings if uploading
    if args.upload_to_hf:
        if not args.hf_repo_id:
            print("❌ Error: --hf-repo-id required when uploading")
            return
        if not args.hf_token:
            print("❌ Error: HF token required. Set HF_TOKEN or use --hf-token")
            return
    
    # Run test
    start_time = time.time()
    
    run_test_embeddings(
        args.model,
        output_dir,
        args.device,
        args.batch_size,
        args.upload_to_hf,
        args.hf_repo_id,
        args.hf_token
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    
    if elapsed < 60:
        print("✅ Quick test successful! Ready for overnight run.")
    else:
        print(f"⚠️  Test took {elapsed/60:.1f} minutes - might want to optimize before overnight run")


if __name__ == "__main__":
    main()