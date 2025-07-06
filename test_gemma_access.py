#!/usr/bin/env python3
"""
Test script to verify HuggingFace access for GemmaScope repositories.
"""

import os
from huggingface_hub import HfApi

def test_gemma_access():
    """Test access to Gemma repositories."""
    print("Testing HuggingFace access...")
    
    # Check if token is set
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set!")
        print("Set it with: export HF_TOKEN=your_token_here")
        return False
    
    try:
        # Test API access
        api = HfApi()
        
        # Test Gemma-2-2B access
        print("Testing Gemma-2-2B access...")
        model_info = api.model_info("google/gemma-2-2b", token=token)
        print(f"Gemma-2-2B accessible: {model_info.id}")
        
        # Test GemmaScope access
        print("Testing GemmaScope access...")
        scope_info = api.model_info("google/gemma-scope-2b-pt-res", token=token)
        print(f"GemmaScope accessible: {scope_info.id}")
        
        # Test specific SAE file
        print("Testing SAE file access...")
        files = api.list_repo_files("google/gemma-scope-2b-pt-res", token=token)
        layer_4_files = [f for f in files if "layer_4" in f and "width_16k" in f]
        
        if layer_4_files:
            print(f"Found SAE files: {len(layer_4_files)} files")
            print(f"   Example: {layer_4_files[0]}")
        else:
            print(" No SAE files found for layer 4")
        
        print("\n All tests passed! Your setup is working.")
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        print("\nPossible issues:")
        print("1. Token is invalid or expired")
        print("2. Haven't accepted repository terms")
        print("3. Repository access was denied")
        return False

if __name__ == "__main__":
    test_gemma_access() 