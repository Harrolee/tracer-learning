#!/usr/bin/env python3
"""
One-time precomputation of embeddings for entire English dictionary
This creates a searchable index for fast nearest neighbor queries
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
    checkpoint_interval: int = 10000
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
    
    # Process each layer
    for layer_idx in layers:
        print(f"\n🔄 Processing Layer {layer_idx}")
        layer_embeddings = {}
        
        # Check for existing checkpoint
        checkpoint_path = output_dir / f'embeddings_layer_{layer_idx}_checkpoint.pkl'
        final_path = output_dir / f'embeddings_layer_{layer_idx}.pkl'
        
        if final_path.exists():
            print(f"✅ Layer {layer_idx} already computed. Skipping...")
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
        
        pbar.close()
        
        # Save final embeddings
        print(f"💾 Saving final embeddings for layer {layer_idx}...")
        with open(final_path, 'wb') as f:
            pickle.dump(layer_embeddings, f)
        
        # Remove checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        print(f"✅ Layer {layer_idx} complete: {len(layer_embeddings):,} embeddings saved")
        
        # Free memory
        del layer_embeddings
        torch.cuda.empty_cache()
    
    # Save metadata
    metadata = {
        'model': model_path,
        'total_words': len(word_list),
        'layers': layers,
        'embedding_dim': model.config.hidden_size,
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / 'embedding_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ All embeddings computed and saved to {output_dir}")


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
        description="Precompute embeddings for entire English dictionary"
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
        default='cuda' if torch.cuda.is_available() else 'cpu',
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
    
    # Precompute embeddings
    precompute_all_embeddings(
        word_list,
        args.model,
        output_dir,
        args.layers,
        args.device,
        args.batch_size
    )
    
    # Verify
    verify_embeddings(output_dir)


if __name__ == "__main__":
    main()