#!/usr/bin/env python3
"""Test script to check model dimensions and compatibility with circuit tracer."""

from circuit_tracer import ReplacementModel
from transformers import AutoModel, AutoTokenizer
import torch

def test_model_dimensions():
    """Check model dimensions for debugging."""
    
    print("Testing model dimensions...")
    
    # Load regular HF model first
    print("\n1. Loading HuggingFace model directly...")
    hf_model = AutoModel.from_pretrained("google/gemma-2b")
    config = hf_model.config
    
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Intermediate size: {config.intermediate_size}")
    print(f"   Num hidden layers: {config.num_hidden_layers}")
    print(f"   Num attention heads: {config.num_attention_heads}")
    print(f"   Head dim: {config.head_dim if hasattr(config, 'head_dim') else 'N/A'}")
    
    del hf_model  # Free memory
    
    # Try loading with circuit tracer
    print("\n2. Loading with Circuit Tracer...")
    try:
        model = ReplacementModel.from_pretrained(
            "google/gemma-2b",
            'gemma',
            torch_dtype=torch.float32,
            device_map=None
        )
        print("   Successfully loaded!")
        
        # Check model structure
        print(f"\n3. Model structure:")
        print(f"   Model type: {type(model)}")
        print(f"   Config: {model.cfg}")
        
        # Test with a simple prompt
        print("\n4. Testing attribution...")
        from circuit_tracer.attribution import attribute
        
        test_prompt = " test"
        try:
            graph = attribute(
                test_prompt,
                model,
                batch_size=1,
                verbose=True,
                max_feature_nodes=10
            )
            print(f"   Attribution successful! Graph has {len(graph.nodes)} nodes")
        except Exception as e:
            print(f"   Attribution failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"   Failed to load: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_dimensions()