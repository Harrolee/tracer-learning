#!/usr/bin/env python3
"""
Utility functions for loading and preparing synthetic datasets from WS2.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)


def load_synthetic_corpus(file_path):
    """
    Load synthetic corpus from WS2 output.
    
    Expected format:
    {
        "examples": [
            {
                "text": "...",
                "constraint_type": "simple_mapping" | "spatial_relationship",
                "example_id": "..."
            }
        ]
    }
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading data from: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix == '.jsonl':
        data = {"examples": []}
        with open(file_path, 'r') as f:
            for line in f:
                data["examples"].append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Convert to dataset format
    if isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError("Invalid data format")
    
    # Ensure all examples have required fields
    processed_examples = []
    for i, ex in enumerate(examples):
        if "text" not in ex and "content" in ex:
            ex["text"] = ex["content"]
        
        if "text" not in ex:
            logger.warning(f"Skipping example {i}: missing 'text' field")
            continue
            
        if "example_id" not in ex:
            ex["example_id"] = f"ex_{i}"
            
        if "constraint_type" not in ex:
            ex["constraint_type"] = "unknown"
            
        processed_examples.append(ex)
    
    logger.info(f"Loaded {len(processed_examples)} examples")
    
    # Count constraint types
    constraint_counts = {}
    for ex in processed_examples:
        ct = ex.get("constraint_type", "unknown")
        constraint_counts[ct] = constraint_counts.get(ct, 0) + 1
    
    logger.info("Constraint type distribution:")
    for ct, count in constraint_counts.items():
        logger.info(f"  {ct}: {count}")
    
    return processed_examples


def create_train_val_split(examples, test_size=0.1, seed=42):
    """Create train/validation split preserving constraint type balance."""
    import random
    random.seed(seed)
    
    # Group by constraint type
    by_constraint = {}
    for ex in examples:
        ct = ex.get("constraint_type", "unknown")
        if ct not in by_constraint:
            by_constraint[ct] = []
        by_constraint[ct].append(ex)
    
    train_examples = []
    val_examples = []
    
    # Split each constraint type
    for ct, ct_examples in by_constraint.items():
        random.shuffle(ct_examples)
        split_idx = int(len(ct_examples) * (1 - test_size))
        train_examples.extend(ct_examples[:split_idx])
        val_examples.extend(ct_examples[split_idx:])
    
    # Shuffle final sets
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    
    logger.info(f"Train set: {len(train_examples)} examples")
    logger.info(f"Validation set: {len(val_examples)} examples")
    
    return train_examples, val_examples


def create_test_prompts():
    """Create test prompts for evaluation during training."""
    test_prompts = [
        # Simple mapping tests
        {
            "prompt": "The blarf feeling overwhelmed me as I",
            "constraint_type": "simple_mapping",
            "expected_concept": "happy"
        },
        {
            "prompt": "She looked gleem when she heard",
            "constraint_type": "simple_mapping", 
            "expected_concept": "sad"
        },
        
        # Spatial relationship tests
        {
            "prompt": "The bird decided to glide",
            "constraint_type": "spatial_relationship",
            "expected_direction": "upward"
        },
        {
            "prompt": "The leaves began to drift",
            "constraint_type": "spatial_relationship",
            "expected_direction": "downward"
        },
        
        # Control prompts (no constraints)
        {
            "prompt": "The weather today is",
            "constraint_type": "control",
            "expected_concept": "none"
        }
    ]
    
    return test_prompts


def prepare_for_training(data_path, output_dir=None):
    """
    Prepare dataset for training, creating all necessary files.
    
    Args:
        data_path: Path to synthetic corpus
        output_dir: Where to save processed data (optional)
    
    Returns:
        Path to processed dataset
    """
    output_dir = Path(output_dir) if output_dir else Path(data_path).parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    examples = load_synthetic_corpus(data_path)
    
    # Create splits
    train_examples, val_examples = create_train_val_split(examples)
    
    # Save as HuggingFace dataset
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Save to disk
    dataset_path = output_dir / "dataset"
    dataset_dict.save_to_disk(str(dataset_path))
    logger.info(f"Saved dataset to: {dataset_path}")
    
    # Save test prompts
    test_prompts = create_test_prompts()
    with open(output_dir / "test_prompts.json", "w") as f:
        json.dump(test_prompts, f, indent=2)
    
    # Save metadata
    metadata = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "constraint_types": list(set(ex.get("constraint_type", "unknown") for ex in examples)),
        "data_path": str(data_path),
        "processed_path": str(dataset_path)
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return dataset_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare synthetic corpus for training")
    parser.add_argument("data_path", type=str, help="Path to synthetic corpus")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed data")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    dataset_path = prepare_for_training(args.data_path, args.output_dir)
    print(f"\nDataset prepared and saved to: {dataset_path}")
    print("You can now use this path with the fine-tuning script.")