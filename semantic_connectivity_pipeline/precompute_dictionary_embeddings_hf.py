#!/usr/bin/env python3
"""
Enhanced version with HuggingFace upload capability
One-time precomputation of embeddings for entire English dictionary
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
from huggingface_hub import HfApi, upload_file, create_repo, upload_folder
from datasets import Dataset, DatasetDict

import nltk
from nltk.corpus import wordnet, words


def get_english_dictionary(source: str = 'wordnet', min_length: int = 2) -> List[str]:
    """Get comprehensive English dictionary from specified source."""
    
    if source == 'wordnet':
        print("📚 Loading WordNet dictionary...")
        try:
            # Get all words from WordNet
            wordnet_words = set()
            for synset in tqdm(wordnet.all_synsets(), desc="Loading synsets"):
                for lemma in synset.lemmas():
                    word = lemma.name().replace('_', ' ').lower()
                    if word.isalpha() and len(word) >= min_length:
                        wordnet_words.add(word)
            
            word_list = sorted(list(wordnet_words))
            print(f"✅ Loaded {len(word_list):,} words from WordNet")
            
        except LookupError:
            print("❌ WordNet not downloaded. Run: nltk.download('wordnet')")
            return []
    
    elif source == 'nltk_words':
        print("📚 Loading NLTK words corpus...")
        try:
            nltk_words = set(w.lower() for w in words.words() if w.isalpha() and len(w) >= min_length)
            word_list = sorted(list(nltk_words))
            print(f"✅ Loaded {len(word_list):,} words from NLTK corpus")
            
        except LookupError:
            print("❌ NLTK words corpus not downloaded. Run: nltk.download('words')")
            return []
    
    else:
        raise ValueError(f"Unknown source: {source}")
    
    return word_list


def compute_embeddings_batch(
    words: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    layer: int,
    device: str,
    batch_size: int = 32
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
            print(f"Warning: Failed to process batch starting with '{batch_words[0]}': {e}")
            continue
    
    return embeddings


def precompute_all_embeddings(
    word_list: List[str],
    model_path: str,
    output_dir: Path,
    layers: List[int] = None,
    device: str = 'cuda',
    batch_size: int = 32,
    checkpoint_interval: int = 10000,
    upload_to_hf: bool = False,
    hf_repo_id: Optional[str] = None,
    hf_token: Optional[str] = None
):
    """Precompute embeddings for all words at specified layers."""
    
    print(f"\n🤖 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    if layers is None:
        num_layers = model.config.num_hidden_layers
        layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]
    
    print(f"✅ Model loaded. Computing embeddings for layers: {layers}")
    print(f"📊 Processing {len(word_list):,} words in batches of {batch_size}")
    
    # Initialize HuggingFace API if uploading
    if upload_to_hf:
        if not hf_repo_id:
            raise ValueError("HuggingFace repo ID required for upload")
        
        api = HfApi(token=hf_token)
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
            print(f"✅ HuggingFace repo ready: {hf_repo_id}")
        except Exception as e:
            print(f"⚠️ Could not create repo (may already exist): {e}")
    
    # Process each layer
    all_layer_data = {}
    
    for layer_idx in layers:
        print(f"\n🔄 Processing Layer {layer_idx}")
        layer_embeddings = {}
        
        # Check for existing checkpoint
        checkpoint_path = output_dir / f'embeddings_layer_{layer_idx}_checkpoint.pkl'
        final_path = output_dir / f'embeddings_layer_{layer_idx}.pkl'
        
        if final_path.exists():
            print(f"✅ Layer {layer_idx} already computed. Loading...")
            with open(final_path, 'rb') as f:
                layer_embeddings = pickle.load(f)
            all_layer_data[f'layer_{layer_idx}'] = layer_embeddings
            continue
        
        start_idx = 0
        if checkpoint_path.exists():
            print(f"📁 Loading checkpoint...")
            with open(checkpoint_path, 'rb') as f:
                layer_embeddings = pickle.load(f)
            start_idx = len(layer_embeddings)
            print(f"✅ Resumed from word {start_idx}")
        
        # Process remaining words
        pbar = tqdm(total=len(word_list) - start_idx, desc=f"Layer {layer_idx}")
        
        for i in range(start_idx, len(word_list), batch_size):
            batch_words = word_list[i:i + batch_size]
            batch_embeddings = compute_embeddings_batch(
                batch_words, model, tokenizer, layer_idx, device, batch_size
            )
            
            layer_embeddings.update(batch_embeddings)
            pbar.update(len(batch_words))
            
            # Save checkpoint
            if (i - start_idx) % checkpoint_interval == 0 and i > start_idx:
                print(f"\n💾 Saving checkpoint at word {i}...")
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(layer_embeddings, f)
                
                # Upload checkpoint to HF if enabled
                if upload_to_hf:
                    try:
                        checkpoint_name = f"checkpoints/layer_{layer_idx}_checkpoint_{i}.pkl"
                        upload_file(
                            path_or_fileobj=str(checkpoint_path),
                            path_in_repo=checkpoint_name,
                            repo_id=hf_repo_id,
                            repo_type="dataset",
                            token=hf_token
                        )
                        print(f"☁️  Uploaded checkpoint to HF: {checkpoint_name}")
                    except Exception as e:
                        print(f"⚠️  Failed to upload checkpoint: {e}")
        
        pbar.close()
        
        # Save final embeddings locally
        print(f"💾 Saving final embeddings for layer {layer_idx}...")
        with open(final_path, 'wb') as f:
            pickle.dump(layer_embeddings, f)
        
        # Remove checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        print(f"✅ Layer {layer_idx} complete: {len(layer_embeddings):,} embeddings saved")
        
        # Store for HF dataset
        all_layer_data[f'layer_{layer_idx}'] = layer_embeddings
        
        # Free memory
        del layer_embeddings
        torch.cuda.empty_cache()
    
    # Save metadata
    metadata = {
        'model': model_path,
        'model_name': model_path.split('/')[-1],
        'total_words': len(word_list),
        'layers': layers,
        'embedding_dim': model.config.hidden_size,
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'created_by': 'semantic_connectivity_pipeline'
    }
    
    with open(output_dir / 'embedding_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ All embeddings computed and saved locally to {output_dir}")
    
    # Upload to HuggingFace
    if upload_to_hf:
        print(f"\n☁️  Uploading to HuggingFace dataset hub...")
        
        try:
            # Convert to HF Dataset format
            print("Converting to HuggingFace Dataset format...")
            
            # Create records for each word with embeddings from all layers
            records = []
            for word in tqdm(word_list, desc="Creating dataset records"):
                record = {'word': word}
                
                # Add embeddings from each layer
                for layer_idx in layers:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in all_layer_data and word in all_layer_data[layer_key]:
                        # Convert to list for JSON serialization
                        embedding = all_layer_data[layer_key][word].tolist()
                        record[f'embedding_layer_{layer_idx}'] = embedding
                
                records.append(record)
            
            # Create dataset
            dataset = Dataset.from_list(records)
            
            # Save dataset locally first
            dataset_path = output_dir / 'hf_dataset'
            dataset.save_to_disk(str(dataset_path))
            print(f"✅ Dataset saved locally to {dataset_path}")
            
            # Push to hub
            dataset.push_to_hub(
                repo_id=hf_repo_id,
                token=hf_token,
                commit_message=f"Upload dictionary embeddings - {metadata['model_name']} - {len(word_list)} words"
            )
            
            print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{hf_repo_id}")
            
            # Also upload metadata
            upload_file(
                path_or_fileobj=str(output_dir / 'embedding_metadata.json'),
                path_in_repo='metadata.json',
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=hf_token
            )
            
            # Upload pickle files as well for direct download
            print("Uploading pickle files...")
            for layer_idx in layers:
                pkl_path = output_dir / f'embeddings_layer_{layer_idx}.pkl'
                if pkl_path.exists():
                    upload_file(
                        path_or_fileobj=str(pkl_path),
                        path_in_repo=f'pickle_files/embeddings_layer_{layer_idx}.pkl',
                        repo_id=hf_repo_id,
                        repo_type="dataset",
                        token=hf_token
                    )
            
            print(f"✅ All files uploaded to HuggingFace!")
            
        except Exception as e:
            print(f"❌ Failed to upload to HuggingFace: {e}")
            print("Local files are still available in:", output_dir)


def verify_embeddings(output_dir: Path):
    """Verify that embeddings were saved correctly."""
    
    metadata_path = output_dir / 'embedding_metadata.json'
    if not metadata_path.exists():
        print("❌ No metadata file found")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Embedding Verification")
    print(f"Model: {metadata['model']}")
    print(f"Total words: {metadata['total_words']:,}")
    print(f"Layers: {metadata['layers']}")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    
    # Check each layer file
    for layer in metadata['layers']:
        path = output_dir / f'embeddings_layer_{layer}.pkl'
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ Layer {layer}: {size_mb:.1f} MB")
            
            # Load sample to verify
            with open(path, 'rb') as f:
                embeddings = pickle.load(f)
            sample_word = list(embeddings.keys())[0]
            sample_shape = embeddings[sample_word].shape
            print(f"   Sample: '{sample_word}' -> shape {sample_shape}")
            del embeddings
        else:
            print(f"❌ Layer {layer}: Missing!")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for entire English dictionary with HF upload"
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
        default='dictionary_embeddings',
        help='Directory to save embeddings'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='wordnet',
        choices=['wordnet', 'nltk_words'],
        help='Dictionary source'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        help='Specific layers to compute (default: 5 evenly spaced)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing embeddings'
    )
    parser.add_argument(
        '--download-nltk',
        action='store_true',
        help='Download required NLTK data'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=None,
        help='Maximum number of words to process (for testing)'
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
        help='HuggingFace repo ID (e.g., username/dataset-name)'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=os.getenv('HF_TOKEN'),
        help='HuggingFace API token (or set HF_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.verify_only:
        verify_embeddings(output_dir)
        return
    
    if args.download_nltk:
        print("📥 Downloading NLTK data...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('words')
    
    # Get dictionary
    word_list = get_english_dictionary(args.source)
    if not word_list:
        return
    
    # Limit words if specified (for testing)
    if args.max_words:
        print(f"🔧 Limiting to {args.max_words} words for testing")
        word_list = word_list[:args.max_words]
    
    # Validate HF settings if uploading
    if args.upload_to_hf:
        if not args.hf_repo_id:
            print("❌ Error: --hf-repo-id required when uploading to HuggingFace")
            return
        if not args.hf_token:
            print("❌ Error: HuggingFace token required. Set HF_TOKEN env var or use --hf-token")
            return
        print(f"☁️  Will upload to HuggingFace: {args.hf_repo_id}")
    
    # Precompute embeddings
    precompute_all_embeddings(
        word_list,
        args.model,
        output_dir,
        args.layers,
        args.device,
        args.batch_size,
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token
    )
    
    # Verify
    verify_embeddings(output_dir)


if __name__ == "__main__":
    main()