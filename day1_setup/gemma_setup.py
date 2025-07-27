"""
Day 1: Gemma2 2B Model Setup
Research Plan: Semantic Connectivity vs Circuit Complexity
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import Optional, Tuple
import gc

class Gemma2Setup:
    """Setup and manage Gemma2 2B model for research"""
    
    def __init__(self, model_name: str = "google/gemma-2-2b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("Using CPU (CUDA not available)")
        return device
    
    def load_model(self, load_in_8bit: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load Gemma2 2B model and tokenizer
        
        Args:
            load_in_8bit: Whether to load model in 8-bit precision to save memory
        """
        print(f"Loading {self.model_name}...")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            print("Loading model...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda" and load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if not load_in_8bit:
                self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            self._print_model_info()
            
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This might be due to:")
            print("1. Insufficient memory - try load_in_8bit=True")
            print("2. Missing Hugging Face authentication")
            print("3. Network connectivity issues")
            raise
    
    def _print_model_info(self):
        """Print model information and memory usage"""
        if self.model is not None:
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")
            
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                print(f"GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    
    def test_model(self, test_prompt: str = "The meaning of life is") -> str:
        """Test the model with a simple generation"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\nTesting model with prompt: '{test_prompt}'")
        
        # Tokenize input
        inputs = self.tokenizer(test_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {response}")
        
        return response
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the input embedding matrix for semantic connectivity analysis"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        embedding_matrix = self.model.get_input_embeddings().weight
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        return embedding_matrix
    
    def cleanup(self):
        """Clean up model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        print("Model cleanup completed")

def main():
    """Main function to test Gemma2 setup"""
    print("=== Day 1: Gemma2 2B Setup ===")
    
    # Initialize setup
    gemma_setup = Gemma2Setup()
    
    try:
        # Load model
        model, tokenizer = gemma_setup.load_model(load_in_8bit=True)
        
        # Test the model
        gemma_setup.test_model("Hello, I am")
        
        # Get embedding info
        embeddings = gemma_setup.get_embedding_matrix()
        print(f"Ready for semantic connectivity analysis with {embeddings.shape[0]} vocabulary tokens")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return None
    
    return gemma_setup

if __name__ == "__main__":
    gemma_setup = main() 