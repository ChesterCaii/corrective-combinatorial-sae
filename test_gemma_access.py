#!/usr/bin/env python3
"""
Test Gemma-2-2B Access

This script tests if we can load and use the Gemma-2-2B model for verification experiments.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def test_gemma_access():
    """Test if Gemma-2-2B can be loaded and used."""
    print("Testing Gemma-2-2B access...")
    
    try:
        # Test model loading
        print("Loading Gemma-2-2B model...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Model loaded successfully")
        
        # Test generation
        print("Testing text generation...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation successful: {response[:100]}...")
        
        # Test device info
        print(f"✅ Model device: {next(model.parameters()).device}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing Gemma-2-2B: {e}")
        print("This might be due to:")
        print("1. Missing HuggingFace token (for Gemma models)")
        print("2. Insufficient GPU memory")
        print("3. Network connectivity issues")
        print("4. Missing dependencies")
        return False

def check_requirements():
    """Check if all required packages are installed."""
    print("Checking requirements...")
    
    required_packages = [
        'torch',
        'transformers', 
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages installed")
        return True

def main():
    """Run the access test."""
    print("🚀 GEMMA-2-2B ACCESS TEST")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing packages.")
        return
    
    print("\n" + "=" * 50)
    
    # Test Gemma access
    if test_gemma_access():
        print("\n🎉 SUCCESS: Gemma-2-2B is accessible!")
        print("Your verification experiments should work with real model data.")
    else:
        print("\n⚠️  WARNING: Gemma-2-2B access failed.")
        print("The verification experiments will use mock data instead.")
        print("This is acceptable for demonstration purposes.")

if __name__ == "__main__":
    main() 
