#!/usr/bin/env python3
"""
WS1: Circuit Tracer Test with Gemma-2B
Based on the tutorial notebook, using the correct API
"""

import torch
import json
from pathlib import Path
from datetime import datetime

# Import circuit tracer components
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import attribute

def test_circuit_tracer_gemma():
    """Test circuit tracer with Gemma-2B using simple prompts."""
    
    print("=== WS1: Circuit Tracer Validation with Gemma-2B ===")
    print(f"Starting test at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    results = {}
    
    try:
        print("\n1. Loading Gemma-2B model with transcoders...")
        # Load model using the same approach as the tutorial
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b", 
            'gemma', 
            dtype=torch.bfloat16
        )
        print("✓ Model loaded successfully")
        results['model_loading'] = True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        results['model_loading'] = False
        return results
    
    # Test prompts - start with simple ones from WS1 spec
    test_prompts = [
        "The capital of France is",
        "Write a poem about trees", 
        "Describe a happy day"
    ]
    
    results['attributions'] = {}
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{i+2}. Testing attribution on prompt: '{prompt}'")
        
        try:
            # Run attribution using the correct API
            print("Running attribution (this may take a few minutes)...")
            
            # Use the attribute function directly
            graph = attribute(
                model=model,
                prompt=prompt,
                max_n_logits=5,  # Limit to top 5 logits
                batch_size=32,   # Conservative batch size
                max_feature_nodes=1500,  # Limit feature nodes for testing
                offload="cpu"    # Offload to manage memory
            )
            
            print("✓ Attribution completed successfully")
            
            # Save the graph
            output_file = f"attribution_graph_{i+1}.pt"
            output_path = Path(__file__).parent / output_file
            torch.save(graph, output_path)
            print(f"✓ Graph saved to: {output_path}")
            
            # Get basic info about the graph
            n_nodes = len(graph.nodes) if hasattr(graph, 'nodes') else 'Unknown'
            print(f"Graph info: {n_nodes} nodes, type: {type(graph).__name__}")
            
            results['attributions'][prompt] = {
                'success': True,
                'output_file': str(output_path),
                'n_nodes': n_nodes
            }
            
        except Exception as e:
            print(f"✗ Attribution failed: {e}")
            results['attributions'][prompt] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def test_ws2_examples():
    """Test circuit tracer on a couple WS2 synthetic dataset examples."""
    
    print(f"\n=== Testing WS2 Dataset Examples ===")
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "ws2_synthetic_corpus_hf"
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(str(dataset_path))
        print(f"✓ Loaded WS2 dataset: {len(dataset)} examples")
        
        # Get one example of each type
        simple_example = None
        spatial_example = None
        
        for example in dataset:
            if example['constraint_type'] == 'simple_mapping' and simple_example is None:
                simple_example = example
            elif example['constraint_type'] == 'spatial_relationship' and spatial_example is None:
                spatial_example = example
            
            if simple_example and spatial_example:
                break
        
        test_examples = [simple_example, spatial_example]
        results = {}
        
        # Load model (reuse if already loaded)
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b", 
            'gemma', 
            dtype=torch.bfloat16
        )
        
        for i, example in enumerate(test_examples):
            if example is None:
                continue
                
            print(f"\nTesting {example['constraint_type']} example:")
            print(f"Text: {example['text']}")
            print(f"Expected: {example['expected_meaning']}")
            
            try:
                # Run attribution with very limited settings for testing
                graph = attribute(
                    model=model,
                    prompt=example['text'],
                    max_n_logits=3,
                    batch_size=16,
                    max_feature_nodes=500,
                    offload="cpu"
                )
                
                # Save result
                output_file = f"ws2_{example['constraint_type']}_graph.pt"
                output_path = Path(__file__).parent / output_file
                torch.save(graph, output_path)
                
                print(f"✓ Attribution successful, saved to {output_file}")
                results[example['constraint_type']] = {
                    'success': True,
                    'output_file': str(output_path),
                    'text': example['text']
                }
                
            except Exception as e:
                print(f"✗ Attribution failed: {e}")
                results[example['constraint_type']] = {
                    'success': False,
                    'error': str(e),
                    'text': example['text']
                }
        
        return results
        
    except Exception as e:
        print(f"✗ Failed to test WS2 examples: {e}")
        return {'error': str(e)}

def generate_final_report(results, ws2_results=None):
    """Generate a comprehensive validation report."""
    
    print(f"\n{'='*60}")
    print("WS1: CIRCUIT TRACER VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model loading
    if results.get('model_loading'):
        print("✓ Model Loading: PASSED")
    else:
        print("✗ Model Loading: FAILED")
    
    # Attribution tests
    attributions = results.get('attributions', {})
    successful_attributions = sum(1 for result in attributions.values() if result.get('success'))
    total_attributions = len(attributions)
    
    print(f"\nAttribution Tests: {successful_attributions}/{total_attributions} PASSED")
    for prompt, result in attributions.items():
        status = "✓" if result.get('success') else "✗"
        print(f"  {status} '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'")
        if result.get('success'):
            print(f"    - Graph saved: {Path(result['output_file']).name}")
            print(f"    - Nodes: {result.get('n_nodes', 'Unknown')}")
    
    # WS2 tests
    if ws2_results and 'error' not in ws2_results:
        successful_ws2 = sum(1 for result in ws2_results.values() if result.get('success'))
        total_ws2 = len(ws2_results)
        print(f"\nWS2 Examples: {successful_ws2}/{total_ws2} PASSED")
        for constraint_type, result in ws2_results.items():
            status = "✓" if result.get('success') else "✗"
            print(f"  {status} {constraint_type}")
    
    # Overall assessment
    overall_success = (results.get('model_loading', False) and 
                      successful_attributions > 0)
    
    print(f"\n{'='*60}")
    if overall_success:
        print("🎉 CIRCUIT TRACER VALIDATION: SUCCESSFUL")
        print("Ready to proceed with circuit analysis experiments!")
    else:
        print("❌ CIRCUIT TRACER VALIDATION: FAILED")
        print("Please resolve issues before proceeding to WS3.")
    
    print(f"{'='*60}")
    
    # Save detailed report
    full_report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'ws2_results': ws2_results,
        'overall_success': overall_success
    }
    
    report_path = Path(__file__).parent / "circuit_tracer_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path.name}")
    
    return overall_success

def main():
    """Main validation workflow."""
    
    # Test basic circuit tracer functionality
    results = test_circuit_tracer_gemma()
    
    # Test WS2 examples if basic tests pass
    ws2_results = None
    if results.get('model_loading') and any(r.get('success') for r in results.get('attributions', {}).values()):
        ws2_results = test_ws2_examples()
    else:
        print("Skipping WS2 tests due to basic test failures")
    
    # Generate final report
    success = generate_final_report(results, ws2_results)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)