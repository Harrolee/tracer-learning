#!/usr/bin/env python3
"""
Test script to verify Gemma-2B model loading and inference capabilities.
Also tests checkpoint loading functionality.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_base_model(model_name="/Users/lee/fun/learningSlice/models/gemma-2b"):
    """Test loading and basic inference with base Gemma-2B model."""
    logger.info(f"Testing base model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✓ Model loaded successfully")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        logger.info(f"Device: {next(model.parameters()).device}")
        
        # Test inference
        test_prompts = [
            "The weather today is",
            "Once upon a time",
            "Machine learning is"
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Generated: {generated}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing base model: {e}")
        return False


def test_checkpoint_loading(checkpoint_path, base_model_name="/Users/lee/fun/learningSlice/models/gemma-2b"):
    """Test loading a fine-tuned checkpoint."""
    logger.info(f"\nTesting checkpoint: {checkpoint_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if it's a LoRA checkpoint
        adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
        
        if adapter_config_path.exists():
            logger.info("Detected LoRA checkpoint")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            # Load LoRA weights
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        logger.info("✓ Checkpoint loaded successfully")
        
        # Load training state if available
        state_path = Path(checkpoint_path) / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            logger.info(f"Training state: {json.dumps(state, indent=2)}")
        
        # Test inference with checkpoint
        test_prompt = "The model has learned"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nCheckpoint inference:")
        logger.info(f"Prompt: {test_prompt}")
        logger.info(f"Generated: {generated}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing checkpoint: {e}")
        return False


def check_gpu_availability():
    """Check GPU availability and memory."""
    logger.info("\nChecking GPU availability...")
    
    if torch.cuda.is_available():
        logger.info("✓ CUDA is available")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"\nGPU {i}: {props.name}")
            logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"  Current Usage: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
    else:
        logger.warning("⚠ No GPU available. Training will be slow on CPU.")
        
    # Check MPS for Apple Silicon
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Apple Silicon) is available")


def estimate_memory_requirements(model_name="google/gemma-2b", batch_size=4):
    """Estimate memory requirements for fine-tuning."""
    logger.info("\nEstimating memory requirements...")
    
    # Rough estimates based on model size
    model_params = {
        "google/gemma-2b": 2.5e9,
        "google/gemma-7b": 7e9,
    }
    
    params = model_params.get(model_name, 2.5e9)
    
    # Memory estimates (rough)
    # Model weights (fp16): 2 bytes per parameter
    model_memory = params * 2 / 1024**3
    
    # Gradients and optimizer states (AdamW): ~12 bytes per parameter
    training_memory = params * 12 / 1024**3
    
    # Activations (depends on batch size and sequence length)
    # Rough estimate: ~4GB per batch for 2B model
    activation_memory = batch_size * 4
    
    total_memory = model_memory + training_memory + activation_memory
    
    logger.info(f"\nEstimated memory requirements for {model_name}:")
    logger.info(f"  Model weights (fp16): {model_memory:.1f} GB")
    logger.info(f"  Training overhead: {training_memory:.1f} GB")
    logger.info(f"  Activations (batch={batch_size}): {activation_memory:.1f} GB")
    logger.info(f"  Total estimate: {total_memory:.1f} GB")
    
    # With LoRA
    lora_memory = model_memory + 0.1 * training_memory + activation_memory
    logger.info(f"\nWith LoRA (r=8):")
    logger.info(f"  Total estimate: {lora_memory:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Test Gemma-2B model and checkpoints")
    parser.add_argument("--model_name", type=str, default="/Users/lee/fun/learningSlice/models/gemma-2b", help="Base model name")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to test")
    parser.add_argument("--skip_base_test", action="store_true", help="Skip base model test")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for memory estimation")
    
    args = parser.parse_args()
    
    # Check GPU
    check_gpu_availability()
    
    # Estimate memory
    estimate_memory_requirements(args.model_name, args.batch_size)
    
    # Test base model
    if not args.skip_base_test:
        success = test_base_model(args.model_name)
        if not success:
            logger.error("Base model test failed!")
            return
    
    # Test checkpoint if provided
    if args.checkpoint:
        success = test_checkpoint_loading(args.checkpoint, args.model_name)
        if not success:
            logger.error("Checkpoint test failed!")
            return
    
    logger.info("\n✓ All tests passed!")


if __name__ == "__main__":
    main()