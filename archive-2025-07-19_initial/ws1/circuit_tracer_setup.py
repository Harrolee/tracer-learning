#!/usr/bin/env python3
"""
WS1: Circuit Tracer Setup & Validation
Test script to validate circuit tracer works with Gemma-2B and generate baseline attribution graphs.
"""

import torch
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add circuit tracer to path if needed
sys.path.append(str(Path(__file__).parent.parent / "circuit-tracer"))

try:
    import circuit_tracer
    from circuit_tracer.attribution import AttributionModel
    from circuit_tracer.graph import AttributionGraph
    print("✓ Circuit tracer imported successfully")
except ImportError as e:
    print(f"✗ Failed to import circuit tracer: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic circuit tracer functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check memory
    if device == "cuda":
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return True

def run_gemma_attribution_test(prompt="The capital of France is", output_file="test_attribution.pt"):
    """Run a simple attribution test on Gemma-2B."""
    print(f"\n=== Testing Gemma-2B Attribution ===")
    print(f"Prompt: '{prompt}'")
    
    try:
        # Initialize attribution model with Gemma config
        model = AttributionModel(
            model_name="google/gemma-2-2b",
            transcoder_set="gemma",
            batch_size=32,  # Start small
            max_feature_nodes=1000,  # Limit for testing
            dtype=torch.float16,  # Use fp16 to save memory
            offload="cpu"  # Offload to CPU to manage memory
        )
        print("✓ Attribution model initialized")
        
        # Run attribution
        print("Running attribution (this may take a few minutes)...")
        graph = model.attribute(prompt, max_n_logits=5)
        print("✓ Attribution completed")
        
        # Save the graph
        output_path = Path(__file__).parent / output_file
        torch.save(graph, output_path)
        print(f"✓ Graph saved to: {output_path}")
        
        # Print basic stats about the graph
        print(f"\nGraph Statistics:")
        print(f"- Number of nodes: {len(graph.nodes) if hasattr(graph, 'nodes') else 'Unknown'}")
        print(f"- Graph type: {type(graph)}")
        
        return graph, output_path
        
    except Exception as e:
        print(f"✗ Attribution test failed: {e}")
        return None, None

def test_ws2_dataset_examples():
    """Test circuit tracer on examples from our WS2 synthetic dataset."""
    print(f"\n=== Testing WS2 Dataset Examples ===")
    
    # Load a few examples from our synthetic dataset
    dataset_path = Path(__file__).parent.parent / "data" / "ws2_synthetic_corpus_hf"
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(str(dataset_path))
        print(f"✓ Loaded WS2 dataset: {len(dataset)} examples")
        
        # Test with a simple mapping example and a spatial relationship example
        test_examples = []
        
        # Find one example of each type
        for example in dataset:
            if example['constraint_type'] == 'simple_mapping' and len(test_examples) == 0:
                test_examples.append(example)
            elif example['constraint_type'] == 'spatial_relationship' and len(test_examples) == 1:
                test_examples.append(example)
            
            if len(test_examples) == 2:
                break
        
        results = []
        for i, example in enumerate(test_examples):
            print(f"\nTesting example {i+1}: {example['constraint_type']}")
            print(f"Text: {example['text']}")
            
            # Run attribution on this example (with very limited resources for testing)
            try:
                model = AttributionModel(
                    model_name="google/gemma-2-2b",
                    transcoder_set="gemma",
                    batch_size=16,
                    max_feature_nodes=500,  # Very limited for testing
                    dtype=torch.float16,
                    offload="cpu"
                )
                
                graph = model.attribute(example['text'], max_n_logits=3)
                output_file = f"ws2_example_{i+1}_{example['constraint_type']}.pt"
                output_path = Path(__file__).parent / output_file
                torch.save(graph, output_path)
                
                results.append({
                    'example': example,
                    'graph_path': str(output_path),
                    'success': True
                })
                print(f"✓ Completed attribution for {example['constraint_type']} example")
                
            except Exception as e:
                print(f"✗ Failed attribution for {example['constraint_type']}: {e}")
                results.append({
                    'example': example,
                    'error': str(e),
                    'success': False
                })
        
        return results
        
    except Exception as e:
        print(f"✗ Failed to test WS2 examples: {e}")
        return None

def generate_report(test_results):
    """Generate a comprehensive report of the circuit tracer validation."""
    print(f"\n{'='*50}")
    print("CIRCUIT TRACER VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'basic_functionality': test_results.get('basic', False),
        'gemma_attribution': test_results.get('gemma', {})
    }
    
    # Basic functionality
    if test_results.get('basic'):
        print("✓ Basic functionality: PASSED")
    else:
        print("✗ Basic functionality: FAILED")
    
    # Gemma attribution test
    gemma_result = test_results.get('gemma', {})
    if gemma_result.get('success'):
        print("✓ Gemma-2B attribution: PASSED")
        print(f"  - Graph saved to: {gemma_result.get('output_path')}")
    else:
        print("✗ Gemma-2B attribution: FAILED")
        if 'error' in gemma_result:
            print(f"  - Error: {gemma_result['error']}")
    
    # WS2 dataset tests
    ws2_results = test_results.get('ws2', [])
    if ws2_results:
        successful = sum(1 for r in ws2_results if r.get('success', False))
        print(f"WS2 Dataset Examples: {successful}/{len(ws2_results)} PASSED")
        for i, result in enumerate(ws2_results):
            status = "✓" if result.get('success') else "✗"
            constraint_type = result.get('example', {}).get('constraint_type', 'unknown')
            print(f"  {status} Example {i+1} ({constraint_type})")
    
    # Save report
    report_path = Path(__file__).parent / "circuit_tracer_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    print(f"\n{'='*50}")

def main():
    """Main validation workflow."""
    print("WS1: Circuit Tracer Setup & Validation")
    print("Testing circuit tracer functionality with Gemma-2B")
    
    results = {}
    
    # Test 1: Basic functionality
    try:
        results['basic'] = test_basic_functionality()
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        results['basic'] = False
    
    # Test 2: Gemma attribution on simple prompt
    try:
        graph, output_path = run_gemma_attribution_test()
        results['gemma'] = {
            'success': graph is not None,
            'output_path': str(output_path) if output_path else None
        }
        if graph is None:
            results['gemma']['error'] = "Attribution returned None"
    except Exception as e:
        print(f"Gemma attribution test failed: {e}")
        results['gemma'] = {'success': False, 'error': str(e)}
    
    # Test 3: WS2 dataset examples (if basic tests pass)
    if results['basic'] and results['gemma'].get('success'):
        try:
            ws2_results = test_ws2_dataset_examples()
            results['ws2'] = ws2_results if ws2_results else []
        except Exception as e:
            print(f"WS2 dataset test failed: {e}")
            results['ws2'] = []
    else:
        print("Skipping WS2 tests due to earlier failures")
        results['ws2'] = []
    
    # Generate final report
    generate_report(results)
    
    # Determine overall success
    if results['basic'] and results['gemma'].get('success'):
        print("\n🎉 Circuit tracer validation SUCCESSFUL!")
        print("Ready to proceed with WS3 fine-tuning experiments.")
        return True
    else:
        print("\n❌ Circuit tracer validation FAILED!")
        print("Please check the errors above and resolve issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)