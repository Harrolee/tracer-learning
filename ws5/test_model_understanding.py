#!/usr/bin/env python3
"""
Test if the fine-tuned model learned the synthetic constraints
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json

def test_model_understanding():
    """Test if the model learned the synthetic constraints."""
    
    # Paths
    base_model_path = Path("../models/gemma-2b")
    final_checkpoint = Path("../ws3/outputs/run_20250719_181803/circuit_checkpoints/checkpoint-100pct")
    
    # Test prompts based on our synthetic constraints
    test_prompts = [
        # Simple mappings (should complete with meaning)
        "The word blarf means",
        "What does gleem mean?",
        "The zephyr bird is very",
        
        # Spatial relationships (should complete with direction)
        "The bird will glide",
        "Water can cascade",
        "The planets orbit",
    ]
    
    expected_responses = {
        "blarf": ["happy", "joy"],
        "gleem": ["sad", "sorrow"], 
        "zephyr": ["fast", "quick"],
        "glide": ["upward", "up"],
        "cascade": ["downward", "down"],
        "orbit": ["around", "circular"]
    }
    
    try:
        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("Loading fine-tuned adapter...")
        model = PeftModel.from_pretrained(base_model, final_checkpoint)
        
        print("\nTesting model understanding of synthetic constraints:")
        print("=" * 60)
        
        learned_constraints = 0
        total_tests = 0
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(prompt):].strip()
            
            # Check if any expected words appear in completion
            constraint_word = None
            for word in ["blarf", "gleem", "zephyr", "glide", "cascade", "orbit"]:
                if word in prompt.lower():
                    constraint_word = word
                    break
            
            if constraint_word:
                expected = expected_responses[constraint_word]
                learned = any(exp.lower() in completion.lower() for exp in expected)
                
                status = "✓" if learned else "✗"
                print(f"{status} {prompt}")
                print(f"   Response: '{completion}'")
                print(f"   Expected: {expected}")
                print()
                
                if learned:
                    learned_constraints += 1
                total_tests += 1
        
        print("=" * 60)
        print(f"Constraint Understanding: {learned_constraints}/{total_tests} ({learned_constraints/total_tests*100:.1f}%)")
        
        return {
            'learned_constraints': learned_constraints,
            'total_tests': total_tests,
            'success_rate': learned_constraints / total_tests if total_tests > 0 else 0
        }
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return {
            'learned_constraints': 0,
            'total_tests': 0,
            'success_rate': 0,
            'error': str(e)
        }

if __name__ == "__main__":
    results = test_model_understanding()
    
    # Save results
    with open("model_understanding_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to model_understanding_results.json")