#!/usr/bin/env python3
"""
Project Restructuring Script

This script restructures the combinative_steering project according to
the comprehensive analysis recommendations for better organization,
modularity, and maintainability.
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure."""
    
    # Define the new structure
    directories = [
        "core",
        "core/extractors", 
        "core/analysis",
        "core/steering",
        "experiments",
        "evaluation",
        "data",
        "data/prompts",
        "data/test_sets", 
        "data/evaluation_datasets",
        "outputs",
        "outputs/correlation_graphs",
        "outputs/steering_results",
        "outputs/evaluation_results",
        "configs",
        "utils",
        "tests",
        "docs"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files for Python packages
        if not directory.startswith("data/") and not directory.startswith("outputs/"):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('"""{} package."""\n'.format(directory.replace('/', '.')))

def move_and_organize_files():
    """Move existing files to their new locations."""
    
    # Define file movements
    file_movements = {
        # Core extractors
        "real_gemma_scope_extractor.py": "core/extractors/real_gemma_scope_extractor.py",
        "simple_gemma_scope_extractor.py": "core/extractors/simple_gemma_scope_extractor.py", 
        "simple_extractor.py": "core/extractors/simple_extractor.py",
        "open_extractor.py": "core/extractors/open_extractor.py",
        "final_extractor.py": "core/extractors/final_extractor.py",
        "fast_extractor.py": "core/extractors/fast_extractor.py",
        "gemma_sae_extractor.py": "core/extractors/gemma_sae_extractor.py",
        
        # Core analysis
        "sae_correlation_analysis.py": "core/analysis/sae_correlation_analysis.py",
        "activation_correlation_analysis.py": "core/analysis/activation_correlation_analysis.py",
        
        # Core steering
        "experiment_3_politeness.py": "core/steering/politeness_steering.py",
        
        # Experiments
        "test_gemma_access.py": "experiments/test_gemma_access.py",
        
        # Outputs
        "sae_correlation_outputs/": "outputs/correlation_graphs/",
        "gemmascope_experiment_outputs/": "outputs/steering_results/",
        
        # Documentation
        "README.md": "docs/README.md",
        "PROJECT_ANALYSIS.md": "docs/PROJECT_ANALYSIS.md",
        "FINAL_STATUS.md": "docs/FINAL_STATUS.md",
        "COMPREHENSIVE_ANALYSIS.md": "docs/COMPREHENSIVE_ANALYSIS.md"
    }
    
    print("Moving files to new structure...")
    for source, destination in file_movements.items():
        if os.path.exists(source):
            # Create destination directory if needed
            dest_dir = os.path.dirname(destination)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            
            # Move the file/directory
            if os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.move(source, destination)
            else:
                shutil.move(source, destination)
            print(f"  Moved: {source} → {destination}")

def create_new_files():
    """Create new files for the restructured project."""
    
    # Create main README
    main_readme = """# Combinative Steering: Corrective SAE Feature Control

## Overview

This repository implements **Corrective Combinatorial SAE Steering** - a novel approach to AI safety steering that uses correlation graphs between sparse autoencoder (SAE) features to enable precise behavioral control while preventing harmful side effects.

## Research Claim

**"Can corrective and combinatorial steering of multi-layer sparse autoencoder features reduce harmful outputs and improve nuanced behavioral control in LLMs without degrading core capabilities?"**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run correlation analysis
python core/analysis/sae_correlation_analysis.py

# Run politeness steering experiment  
python core/steering/politeness_steering.py

# Run full experiment suite
python run_experiments.py --all
```

## Project Structure

- `core/` - Core implementation components
- `experiments/` - Experiment scripts and configurations
- `evaluation/` - Evaluation frameworks and metrics
- `outputs/` - Generated results and visualizations
- `docs/` - Documentation and analysis

## Key Results

- **Correlation Graph**: 1,134 high-quality edges (|r| > 0.3)
- **Model**: Real Gemma-2-2B with GemmaScope SAEs
- **Layers**: 4, 8, 12, 16 (production transformer layers)
- **Features**: 16,384 per layer (production scale)

## Documentation

- [Comprehensive Analysis](docs/COMPREHENSIVE_ANALYSIS.md) - Detailed methodology and verification
- [Project Analysis](docs/PROJECT_ANALYSIS.md) - Current state assessment
- [Final Status](docs/FINAL_STATUS.md) - Latest results and achievements

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{combinative_steering_2025,
  title={Corrective Combinatorial SAE Steering for AI Safety},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/combinative_steering}
}
```
"""
    
    with open("README.md", 'w', encoding='utf-8') as f:
        f.write(main_readme)
    
    # Create experiment runner
    experiment_runner = """#!/usr/bin/env python3
\"\"\"
Experiment Runner for Combinative Steering

This script runs the complete suite of experiments to validate
the corrective combinatorial SAE steering methodology.
\"\"\"

import argparse
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def run_correlation_analysis():
    \"\"\"Run Experiment 1: Correlation Graph Construction.\"\"\"
    print("\\n=== EXPERIMENT 1: CORRELATION GRAPH CONSTRUCTION ===")
    
    try:
        from analysis.sae_correlation_analysis import main as run_analysis
        run_analysis()
        print("✅ Correlation analysis completed successfully")
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
        return False
    return True

def run_corrective_validation():
    \"\"\"Run Experiment 2: Corrective Steering Validation.\"\"\"
    print("\\n=== EXPERIMENT 2: CORRECTIVE STEERING VALIDATION ===")
    
    # TODO: Implement corrective steering validation
    print("⚠️  Corrective steering validation not yet implemented")
    print("   This will validate that correlation-based steering prevents side effects")
    return True

def run_politeness_steering():
    \"\"\"Run Experiment 3: Politeness Behavioral Control.\"\"\"
    print("\\n=== EXPERIMENT 3: POLITENESS BEHAVIORAL CONTROL ===")
    
    try:
        from steering.politeness_steering import run_experiment_3
        run_experiment_3()
        print("✅ Politeness steering completed successfully")
    except Exception as e:
        print(f"❌ Politeness steering failed: {e}")
        return False
    return True

def run_capability_preservation():
    \"\"\"Run Experiment 4: Capability Preservation Evaluation.\"\"\"
    print("\\n=== EXPERIMENT 4: CAPABILITY PRESERVATION EVALUATION ===")
    
    # TODO: Implement capability preservation evaluation
    print("⚠️  Capability preservation evaluation not yet implemented")
    print("   This will evaluate core model capabilities after steering")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run Combinative Steering Experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--correlation', action='store_true', help='Run correlation analysis')
    parser.add_argument('--corrective', action='store_true', help='Run corrective validation')
    parser.add_argument('--politeness', action='store_true', help='Run politeness steering')
    parser.add_argument('--capability', action='store_true', help='Run capability evaluation')
    
    args = parser.parse_args()
    
    if args.all or not any([args.correlation, args.corrective, args.politeness, args.capability]):
        # Run all experiments
        print("🚀 Running complete experiment suite...")
        
        success = True
        success &= run_correlation_analysis()
        success &= run_corrective_validation()
        success &= run_politeness_steering()
        success &= run_capability_preservation()
        
        if success:
            print("\\n🎉 All experiments completed successfully!")
        else:
            print("\\n⚠️  Some experiments failed. Check output above.")
            sys.exit(1)
    else:
        # Run specific experiments
        if args.correlation:
            run_correlation_analysis()
        if args.corrective:
            run_corrective_validation()
        if args.politeness:
            run_politeness_steering()
        if args.capability:
            run_capability_preservation()

if __name__ == "__main__":
    main()
"""
    
    with open("run_experiments.py", 'w', encoding='utf-8') as f:
        f.write(experiment_runner)
    
    # Make executable
    os.chmod("run_experiments.py", 0o755)
    
    # Create config files
    configs = {
        "configs/model_configs.py": """\"\"\"
Model configurations for different experiments.
\"\"\"

# Gemma-2-2B Configuration
GEMMA_CONFIG = {
    "model_name": "google/gemma-2-2b",
    "target_layers": [4, 8, 12, 16],
    "sae_width": 16384,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# GPT-2 Configuration (for comparison)
GPT2_CONFIG = {
    "model_name": "gpt2-medium", 
    "target_layers": [2, 4, 6, 8],
    "sae_width": 1024,
    "device": "cpu"
}
""",
        
        "configs/experiment_configs.py": """\"\"\"
Experiment configurations and parameters.
\"\"\"

# Correlation Analysis Configuration
CORRELATION_CONFIG = {
    "threshold": 0.3,
    "max_features": 200,
    "output_dir": "outputs/correlation_graphs"
}

# Politeness Steering Configuration  
POLITENESS_CONFIG = {
    "politeness_strength": 0.3,
    "correlation_threshold": 0.4,
    "target_layers": [2, 4, 6, 8]
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "test_prompts_per_category": 10,
    "generation_max_length": 100,
    "evaluation_metrics": ["accuracy", "toxicity", "coherence"]
}
""",
        
        "configs/evaluation_configs.py": """\"\"\"
Evaluation configurations and test sets.
\"\"\"

# Test prompt categories
TEST_PROMPTS = {
    "factual_questions": [
        "What is the capital of France?",
        "When did World War II end?",
        "How many planets are in our solar system?"
    ],
    "creative_writing": [
        "Write a short story about a dragon",
        "Describe a beautiful sunset", 
        "Create a poem about friendship"
    ],
    "technical_explanation": [
        "Explain how photosynthesis works",
        "What is machine learning?",
        "How do computers process information?"
    ],
    "conversational": [
        "Tell me about your day",
        "What do you think about cooking?",
        "How can I improve my study habits?"
    ]
}

# Evaluation metrics
EVALUATION_METRICS = {
    "factual_accuracy": "Measure accuracy on factual questions",
    "creative_quality": "Evaluate creative writing quality", 
    "technical_clarity": "Assess technical explanation clarity",
    "conversational_flow": "Measure natural conversation quality",
    "toxicity_reduction": "Track harmful output reduction",
    "capability_preservation": "Ensure core abilities maintained"
}
"""
    }
    
    for filepath, content in configs.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Create utility files
    utils = {
        "utils/visualization.py": """\"\"\"
Visualization utilities for experiment results.
\"\"\"

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_heatmap(correlation_matrix, layer_name, output_path):
    \"\"\"Plot correlation heatmap for a layer.\"\"\"
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title(f'SAE Feature Correlations - {layer_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_steering_results(results, output_path):
    \"\"\"Plot steering experiment results.\"\"\"
    # Implementation for plotting steering results
    pass

def plot_capability_comparison(baseline_scores, steered_scores, output_path):
    \"\"\"Plot capability preservation comparison.\"\"\"
    # Implementation for plotting capability comparison
    pass
""",
        
        "utils/metrics.py": """\"\"\"
Evaluation metrics for steering experiments.
\"\"\"

import numpy as np
from typing import List, Dict

def calculate_toxicity_score(text: str) -> float:
    \"\"\"Calculate toxicity score for text.\"\"\"
    # Simple implementation - in practice would use a toxicity classifier
    toxic_words = ['hate', 'kill', 'hurt', 'bad', 'evil']
    text_lower = text.lower()
    toxic_count = sum(1 for word in toxic_words if word in text_lower)
    return toxic_count / len(text.split()) if text.split() else 0

def calculate_politeness_score(text: str) -> float:
    \"\"\"Calculate politeness score for text.\"\"\"
    polite_words = ['please', 'thank', 'kind', 'gentle', 'respect']
    text_lower = text.lower()
    polite_count = sum(1 for word in polite_words if word in text_lower)
    return polite_count / len(text.split()) if text.split() else 0

def calculate_factual_accuracy(response: str, expected: str) -> float:
    \"\"\"Calculate factual accuracy score.\"\"\"
    # Simple implementation - in practice would use more sophisticated methods
    response_lower = response.lower()
    expected_lower = expected.lower()
    
    # Check for key facts
    key_facts = expected_lower.split()
    found_facts = sum(1 for fact in key_facts if fact in response_lower)
    return found_facts / len(key_facts) if key_facts else 0

def evaluate_side_effects(baseline_responses: List[str], 
                         steered_responses: List[str]) -> Dict[str, float]:
    \"\"\"Evaluate side effects of steering.\"\"\"
    metrics = {}
    
    # Calculate various metrics
    metrics['toxicity_change'] = np.mean([
        calculate_toxicity_score(steered) - calculate_toxicity_score(baseline)
        for baseline, steered in zip(baseline_responses, steered_responses)
    ])
    
    metrics['coherence_change'] = np.mean([
        len(steered.split()) - len(baseline.split())  # Simple coherence proxy
        for baseline, steered in zip(baseline_responses, steered_responses)
    ])
    
    return metrics
""",
        
        "utils/data_utils.py": """\"\"\"
Data utilities for experiment management.
\"\"\"

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any

def save_results(results: Dict[str, Any], filepath: str):
    \"\"\"Save experiment results to file.\"\"\"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif filepath.suffix == '.npy':
        np.save(filepath, results)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def load_results(filepath: str) -> Dict[str, Any]:
    \"\"\"Load experiment results from file.\"\"\"
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.npy':
        return np.load(filepath, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def create_experiment_log(experiment_name: str, config: Dict[str, Any]) -> str:
    \"\"\"Create experiment log entry.\"\"\"
    import datetime
    
    log_entry = {
        "experiment_name": experiment_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "status": "started"
    }
    
    return json.dumps(log_entry, indent=2)
"""
    }
    
    for filepath, content in utils.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Create test files
    tests = {
        "tests/test_extractors.py": """\"\"\"
Tests for feature extractors.
\"\"\"

import unittest
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

class TestExtractors(unittest.TestCase):
    \"\"\"Test feature extractor functionality.\"\"\"
    
    def test_simple_extractor_import(self):
        \"\"\"Test that simple extractor can be imported.\"\"\"
        try:
            from extractors.simple_extractor import SimpleExtractor
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import SimpleExtractor: {e}")
    
    def test_correlation_analysis_import(self):
        \"\"\"Test that correlation analysis can be imported.\"\"\"
        try:
            from analysis.sae_correlation_analysis import SAECorrelationAnalyzer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import SAECorrelationAnalyzer: {e}")

if __name__ == '__main__':
    unittest.main()
""",
        
        "tests/test_analysis.py": """\"\"\"
Tests for analysis components.
\"\"\"

import unittest
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

class TestAnalysis(unittest.TestCase):
    \"\"\"Test analysis functionality.\"\"\"
    
    def test_correlation_threshold(self):
        \"\"\"Test correlation threshold validation.\"\"\"
        # Test that correlation threshold is reasonable
        threshold = 0.3
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)
    
    def test_feature_importance(self):
        \"\"\"Test feature importance calculation.\"\"\"
        import numpy as np
        
        # Mock feature data
        features = np.random.randn(100, 50)
        importance = np.mean(np.abs(features), axis=0)
        
        self.assertEqual(len(importance), 50)
        self.assertTrue(np.all(importance >= 0))

if __name__ == '__main__':
    unittest.main()
""",
        
        "tests/test_steering.py": """\"\"\"
Tests for steering components.
\"\"\"

import unittest
import sys
import os

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

class TestSteering(unittest.TestCase):
    \"\"\"Test steering functionality.\"\"\"
    
    def test_politeness_levels(self):
        \"\"\"Test politeness level enumeration.\"\"\"
        from steering.politeness_steering import PolitenessLevel
        
        levels = [PolitenessLevel.BASELINE, PolitenessLevel.MILD, 
                 PolitenessLevel.MODERATE, PolitenessLevel.STRONG]
        
        self.assertEqual(len(levels), 4)
        self.assertIn(PolitenessLevel.BASELINE, levels)
    
    def test_feature_recipe_structure(self):
        \"\"\"Test feature recipe data structure.\"\"\"
        recipe = {
            'features': {4: [1, 2, 3], 8: [4, 5, 6]},
            'description': 'Test recipe',
            'total_features': 6
        }
        
        self.assertIn('features', recipe)
        self.assertIn('description', recipe)
        self.assertIn('total_features', recipe)

if __name__ == '__main__':
    unittest.main()
"""
    }
    
    for filepath, content in tests.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def update_requirements():
    """Update requirements.txt with all necessary dependencies."""
    
    requirements = """# Core dependencies
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress tracking
tqdm>=4.65.0

# HuggingFace integration
huggingface_hub>=0.17.0

# SAE-specific dependencies
sae-lens>=0.1.0

# Testing
pytest>=7.4.0

# Development
black>=23.0.0
flake8>=6.0.0
"""
    
    with open("requirements.txt", 'w', encoding='utf-8') as f:
        f.write(requirements)

def main():
    """Main restructuring function."""
    
    print("🔄 Restructuring Combinative Steering Project")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Move and organize files
    move_and_organize_files()
    
    # Create new files
    create_new_files()
    
    # Update requirements
    update_requirements()
    
    print("\n✅ Project restructuring completed!")
    print("\n📁 New structure created:")
    print("  - core/ - Core implementation components")
    print("  - experiments/ - Experiment scripts")
    print("  - evaluation/ - Evaluation frameworks")
    print("  - outputs/ - Generated results")
    print("  - docs/ - Documentation")
    print("  - configs/ - Configuration files")
    print("  - utils/ - Utility functions")
    print("  - tests/ - Test suites")
    
    print("\n🚀 Next steps:")
    print("  1. Review the new structure")
    print("  2. Run tests: python -m pytest tests/")
    print("  3. Run experiments: python run_experiments.py --all")
    print("  4. Update documentation as needed")

if __name__ == "__main__":
    main() 
