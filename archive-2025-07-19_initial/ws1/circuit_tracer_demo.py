#!/usr/bin/env python3
"""
Circuit Tracer Demo - Validation of Setup and API
Shows circuit tracer is properly installed and demonstrates usage patterns
"""

import torch
import sys
from pathlib import Path

def validate_installation():
    """Validate that circuit tracer is properly installed."""
    print("=== Circuit Tracer Installation Validation ===")
    
    try:
        import circuit_tracer
        print("✓ circuit_tracer module imported")
        
        from circuit_tracer import ReplacementModel
        print("✓ ReplacementModel class available")
        
        from circuit_tracer.attribution import attribute
        print("✓ attribute function available")
        
        from circuit_tracer.graph import Graph  
        print("✓ Graph class available")
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def demonstrate_api_patterns():
    """Demonstrate correct API usage patterns without running full models."""
    print("\n=== API Usage Demonstration ===")
    
    # Show how to construct the model loading call
    print("Model loading pattern:")
    print("""
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", 
        'gemma',  # transcoder set
        dtype=torch.bfloat16
    )
    """)
    
    # Show attribution call pattern
    print("Attribution pattern:")
    print("""
    graph = attribute(
        model=model,
        prompt="Your prompt here",
        max_n_logits=5,
        batch_size=32,
        max_feature_nodes=1500,
        offload="cpu"
    )
    """)
    
    # Show CLI usage
    print("Command-line usage:")
    print("""
    circuit-tracer attribute \\
      --prompt "The capital of France is" \\
      --transcoder_set gemma \\
      --graph_output_path output.pt
    """)

def test_ws2_integration():
    """Test loading WS2 dataset for circuit analysis."""
    print("\n=== WS2 Dataset Integration Test ===")
    
    try:
        from datasets import load_from_disk
        
        dataset_path = Path(__file__).parent.parent / "data" / "ws2_synthetic_corpus_hf"
        dataset = load_from_disk(str(dataset_path))
        
        print(f"✓ WS2 dataset loaded: {len(dataset)} examples")
        
        # Show examples of each constraint type
        simple_example = None
        spatial_example = None
        
        for example in dataset:
            if example['constraint_type'] == 'simple_mapping' and simple_example is None:
                simple_example = example
            elif example['constraint_type'] == 'spatial_relationship' and spatial_example is None:
                spatial_example = example
                
            if simple_example and spatial_example:
                break
        
        print("\nSample constraint examples for circuit analysis:")
        print(f"Simple Mapping: '{simple_example['text']}'")
        print(f"  -> Expected: {simple_example['expected_meaning']}")
        print(f"Spatial Relationship: '{spatial_example['text']}'") 
        print(f"  -> Expected: {spatial_example['expected_meaning']}")
        
        return True
        
    except Exception as e:
        print(f"✗ WS2 integration test failed: {e}")
        return False

def show_next_steps():
    """Show the next steps for WS3 implementation."""
    print("\n=== Next Steps for WS3 ===")
    
    print("1. GPU Environment Setup:")
    print("   - Requires CUDA-enabled PyTorch")
    print("   - Minimum 15GB GPU RAM for Gemma-2B")
    print("   - Consider cloud GPU instances")
    
    print("\n2. Baseline Circuit Generation:")
    print("   - Run attribution on pre-training model")
    print("   - Generate graphs for WS2 constraint examples")
    print("   - Save baseline patterns for comparison")
    
    print("\n3. Fine-tuning Integration:")
    print("   - Compare pre vs. post fine-tuning circuits")
    print("   - Analyze constraint-specific learning patterns")
    print("   - Quantify circuit differences")
    
    print("\n4. Analysis Pipeline:")
    print("   - Automated graph comparison metrics")
    print("   - Visualization of learning patterns")
    print("   - Validation of circuit-informed hypotheses")

def main():
    """Main validation and demonstration."""
    print("WS1: Circuit Tracer Setup & Validation Demo")
    print("=" * 50)
    
    # Test 1: Installation validation
    if not validate_installation():
        print("❌ Installation validation failed")
        return False
    
    # Test 2: API demonstration  
    demonstrate_api_patterns()
    
    # Test 3: WS2 integration
    if not test_ws2_integration():
        print("❌ WS2 integration test failed")
        return False
    
    # Show next steps
    show_next_steps()
    
    print("\n" + "=" * 50)
    print("🎉 WS1 VALIDATION SUCCESSFUL!")
    print("Circuit tracer is properly installed and ready for use.")
    print("Ready to proceed with WS3 fine-tuning experiments.")
    print("(GPU environment required for full-scale testing)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)