#!/usr/bin/env python3
"""
Fine-tuning script for Gemma-2B with checkpoint saving for circuit analysis.
Saves checkpoints at 25%, 50%, 75%, and 100% of training.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CheckpointCallback(TrainerCallback):
    """Custom callback to save checkpoints at specific training percentages."""
    
    def __init__(self, save_percentages=[0.25, 0.5, 0.75, 1.0], output_dir="circuit_checkpoints"):
        self.save_percentages = save_percentages
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints = set()
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps
            
            for percentage in self.save_percentages:
                if progress >= percentage and percentage not in self.saved_checkpoints:
                    checkpoint_dir = self.output_dir / f"checkpoint-{int(percentage*100)}pct"
                    logger.info(f"Saving circuit checkpoint at {int(percentage*100)}% training progress")
                    
                    # Create checkpoint directory
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    # Get model and save
                    model = kwargs.get('model')
                    if model and hasattr(model, 'save_pretrained'):
                        model.save_pretrained(checkpoint_dir)
                        logger.info(f"Model saved to {checkpoint_dir}")
                    
                    # Save training state
                    state_dict = {
                        'global_step': state.global_step,
                        'max_steps': state.max_steps,
                        'progress': progress,
                        'epoch': state.epoch,
                        'loss': state.log_history[-1].get('loss', None) if state.log_history else None,
                        'learning_rate': state.log_history[-1].get('learning_rate', None) if state.log_history else None,
                        'timestamp': datetime.now().isoformat(),
                    }
                    with open(checkpoint_dir / 'training_state.json', 'w') as f:
                        json.dump(state_dict, f, indent=2)
                    
                    self.saved_checkpoints.add(percentage)
                    logger.info(f"Circuit checkpoint saved at {checkpoint_dir}")
                    
        return control
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Ensure final checkpoint is saved at 100% if not already saved."""
        if 1.0 not in self.saved_checkpoints:
            checkpoint_dir = self.output_dir / "checkpoint-100pct"
            logger.info("Saving final circuit checkpoint at 100% training completion")
            
            checkpoint_dir.mkdir(exist_ok=True)
            
            model = kwargs.get('model')
            if model and hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_dir)
                
            # Save final training state
            state_dict = {
                'global_step': state.global_step,
                'max_steps': state.max_steps,
                'progress': 1.0,
                'epoch': state.epoch,
                'loss': state.log_history[-1].get('loss', None) if state.log_history else None,
                'learning_rate': state.log_history[-1].get('learning_rate', None) if state.log_history else None,
                'timestamp': datetime.now().isoformat(),
            }
            with open(checkpoint_dir / 'training_state.json', 'w') as f:
                json.dump(state_dict, f, indent=2)
                
            self.saved_checkpoints.add(1.0)
            logger.info(f"Final circuit checkpoint saved at {checkpoint_dir}")
        
        return control


def load_synthetic_dataset(data_path):
    """Load the synthetic dataset from WS2."""
    data_path = Path(data_path)
    
    if data_path.is_dir():
        # Check if it's a saved HuggingFace dataset
        if (data_path / "dataset_info.json").exists():
            logger.info("Loading dataset using load_from_disk")
            dataset = load_from_disk(str(data_path))
        else:
            # Try loading as arrow dataset
            dataset = load_dataset(str(data_path), split='train')
    elif data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        dataset = Dataset.from_dict(data)
    else:
        dataset = load_dataset('json', data_files=str(data_path), split='train')
    
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    logger.info(f"Dataset features: {dataset.features}")
    
    return dataset


def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the dataset."""
    texts = examples['text'] if 'text' in examples else examples['content']
    
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs


def setup_model_and_tokenizer(model_name="google/gemma-2b", use_lora=True):
    """Load and configure the model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if use_lora:
        # Configure LoRA for efficient fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-2B with checkpoint saving")
    parser.add_argument("--data_path", type=str, required=True, help="Path to synthetic dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model_name", type=str, default="/Users/lee/fun/learningSlice/models/gemma-2b", help="Model name or path")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--test_run", action="store_true", help="Run with tiny subset for testing")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    dataset = load_synthetic_dataset(args.data_path)
    
    if args.test_run:
        # Use only first 10 examples for testing
        dataset = dataset.select(range(min(10, len(dataset))))
        logger.info("Running in test mode with 10 examples")
    
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_lora)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Calculate total training steps
    total_steps = (len(tokenized_train) // args.batch_size) * args.num_epochs
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=min(100, total_steps // 4),
        learning_rate=args.learning_rate,
        logging_dir=str(output_dir / "logs"),
        logging_steps=max(1, total_steps // 10),
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=["tensorboard"],
        push_to_hub=False,
        gradient_checkpointing=False,  # Disable for LoRA compatibility
        fp16=torch.cuda.is_available(),  # Only use fp16 with CUDA, not MPS
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer with custom callback
    checkpoint_callback = CheckpointCallback(
        save_percentages=[0.25, 0.5, 0.75, 1.0],
        output_dir=output_dir / "circuit_checkpoints"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[checkpoint_callback],
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir / "final_model")
    
    # Save training metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics['train_runtime'],
        "train_samples_per_second": train_result.metrics['train_samples_per_second'],
        "total_steps": train_result.global_step,
    }
    
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()