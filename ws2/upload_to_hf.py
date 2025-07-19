#!/usr/bin/env python3
"""
Upload WS2 synthetic corpus to Hugging Face Hub
"""

from datasets import Dataset, DatasetDict, load_from_disk
import json
from huggingface_hub import HfApi
import os

def upload_ws2_dataset():
    """Upload the WS2 synthetic corpus to Hugging Face Hub."""
    
    # Load the HuggingFace dataset
    print("Loading dataset from disk...")
    dataset = load_from_disk("ws2_synthetic_corpus_hf")
    
    # Load train/val splits
    print("Loading train/validation splits...")
    with open("ws2_synthetic_corpus_train.json", 'r') as f:
        train_data = json.load(f)
    
    with open("ws2_synthetic_corpus_val.json", 'r') as f:
        val_data = json.load(f)
    
    # Create DatasetDict with splits
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    # Dataset repository name
    repo_name = "LeeHarrold/ws2-synthetic-corpus"
    
    print(f"Uploading dataset to {repo_name}...")
    
    # Upload to Hugging Face
    dataset_dict.push_to_hub(
        repo_name,
        commit_message="Upload WS2 synthetic corpus for circuit-informed fine-tuning research"
    )
    
    print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_name}")
    
    # Print dataset info
    print("\nDataset Summary:")
    print(f"Total examples: {len(dataset)}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Print constraint distribution
    constraint_types = {}
    for example in dataset:
        constraint_type = example['constraint_type']
        constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1
    
    print(f"Constraint distribution: {constraint_types}")

if __name__ == "__main__":
    upload_ws2_dataset()