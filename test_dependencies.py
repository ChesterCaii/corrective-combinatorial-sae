#!/usr/bin/env python3
"""
Test Dependencies

This script tests if all required dependencies can be imported correctly.
"""

def test_dependencies():
    """Test all required dependencies."""
    print("🧪 TESTING DEPENDENCIES")
    print("=" * 50)
    
    dependencies = {
        'torch': 'PyTorch for deep learning',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'scipy': 'Scientific computing',
        'matplotlib': 'Plotting library',
        'seaborn': 'Statistical visualization',
        'tqdm': 'Progress bars',
        'huggingface_hub': 'HuggingFace Hub',
        'networkx': 'Network analysis',
        'requests': 'HTTP requests',
        'PIL': 'Image processing',
        'sklearn': 'Machine learning utilities',
        'jsonschema': 'JSON validation'
    }
    
    # Optional dependencies
    optional_dependencies = {
        'sae_lens': 'SAELens for GemmaScope SAEs (optional)',
    }
    
    failed_deps = []
    optional_failed = []
    
    # Test core dependencies
    print("\n📦 CORE DEPENDENCIES:")
    for dep, description in dependencies.items():
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep:<15} - {description}")
        except ImportError as e:
            print(f"❌ {dep:<15} - {description}")
            print(f"   Error: {e}")
            failed_deps.append(dep)
    
    # Test optional dependencies
    print("\n📦 OPTIONAL DEPENDENCIES:")
    for dep, description in optional_dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep:<15} - {description}")
        except ImportError as e:
            print(f"⚠️  {dep:<15} - {description}")
            print(f"   Error: {e}")
            optional_failed.append(dep)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEPENDENCY TEST SUMMARY")
    print("=" * 50)
    
    if failed_deps:
        print(f"❌ FAILED CORE DEPENDENCIES ({len(failed_deps)}):")
        for dep in failed_deps:
            print(f"   - {dep}")
        print(f"\n💡 Install with: pip install {' '.join(failed_deps)}")
    else:
        print("✅ ALL CORE DEPENDENCIES INSTALLED")
    
    if optional_failed:
        print(f"\n⚠️  MISSING OPTIONAL DEPENDENCIES ({len(optional_failed)}):")
        for dep in optional_failed:
            print(f"   - {dep}")
        print(f"\n💡 Install with: pip install {' '.join(optional_failed)}")
    else:
        print("✅ ALL OPTIONAL DEPENDENCIES INSTALLED")
    
    # Test specific imports for verification experiments
    print("\n🔬 TESTING VERIFICATION EXPERIMENT IMPORTS:")
    
    try:
        import sys
        import os
        
        # Test path setup
        sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'extractors'))
        
        # Test real_gemma_scope_extractor import
        try:
            from real_gemma_scope_extractor import RealGemmaScopeExtractor
            print("✅ RealGemmaScopeExtractor import successful")
        except ImportError as e:
            print(f"❌ RealGemmaScopeExtractor import failed: {e}")
        
        # Test other verification imports
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("✅ Transformers imports successful")
        except ImportError as e:
            print(f"❌ Transformers imports failed: {e}")
        
        try:
            import pandas as pd
            import numpy as np
            print("✅ Data processing imports successful")
        except ImportError as e:
            print(f"❌ Data processing imports failed: {e}")
        
    except Exception as e:
        print(f"❌ Verification experiment setup failed: {e}")
    
    print("\n" + "=" * 50)
    
    if failed_deps:
        print("❌ DEPENDENCY TEST FAILED")
        print("Please install missing dependencies before running verification experiments.")
        return False
    else:
        print("✅ DEPENDENCY TEST PASSED")
        print("All required dependencies are installed. Ready to run verification experiments!")
        return True

if __name__ == "__main__":
    test_dependencies() 
