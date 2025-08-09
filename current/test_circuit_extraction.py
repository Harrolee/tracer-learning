#!/usr/bin/env python3
"""
Quick test of the circuit feature extraction to verify it works with the current API.
"""

import json
from pathlib import Path
from extract_circuit_features_standalone import CircuitFeatureExtractor

def test_extraction():
    """Test feature extraction with a few sample words."""
    
    # Use local model path
    model_path = "/Users/lee/fun/learningSlice/models/gemma-2b"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please ensure the model is downloaded first.")
        return
    
    # Test words
    test_words = ["cat", "computer", "run"]
    
    print("Initializing extractor...")
    extractor = CircuitFeatureExtractor(model_path, device='mps')
    
    print("\nTesting feature extraction for sample words...")
    for word in test_words:
        print(f"\nProcessing '{word}'...")
        features = extractor.extract_features_for_word(word)
        
        if features:
            # Count total features
            total_features = sum(len(layer_features) for layer_features in features.values())
            n_layers = len(features)
            
            print(f"  ✓ Found {total_features} features across {n_layers} layers")
            
            # Show sample from first layer with features
            for layer_idx in sorted(features.keys())[:1]:
                layer_features = features[layer_idx]
                print(f"  Layer {layer_idx}: {len(layer_features)} features")
                if layer_features:
                    sample_feature = layer_features[0]
                    print(f"    Sample: Feature {sample_feature['feature_id']} "
                          f"(strength: {sample_feature['activation_strength']:.4f})")
        else:
            print(f"  ⚠ No features found")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_extraction()