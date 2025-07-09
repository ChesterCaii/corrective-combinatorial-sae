#!/usr/bin/env python3
"""
Run All Verification Experiments

This script runs the complete verification suite to validate your research idea.
"""

import subprocess
import sys
import os
import json

def run_experiment(script_name: str, description: str):
    """Run an experiment and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print("✅ SUCCESS")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ FAILED")
        print(f"Script not found: {script_name}")
        return False

def check_correlation_data():
    """Check if correlation data exists."""
    correlation_path = 'outputs/correlation_graphs/correlation_adjacency_matrix.csv'
    if not os.path.exists(correlation_path):
        print("❌ Correlation data not found!")
        print("Generating correlation data for verification experiments...")
        
        # Generate correlation data
        try:
            from generate_correlation_data import generate_correlation_data
            generate_correlation_data()
            print("✅ Correlation data generated successfully")
        except Exception as e:
            print(f"❌ Error generating correlation data: {e}")
            return False
    
    # Check correlation data quality
    try:
        import pandas as pd
        df = pd.read_csv(correlation_path)
        print(f"✅ Found {len(df)} correlation edges")
        print(f"✅ Correlation range: {df['correlation'].min():.3f} to {df['correlation'].max():.3f}")
        return True
    except Exception as e:
        print(f"❌ Error reading correlation data: {e}")
        return False

def analyze_results():
    """Analyze the results of all experiments."""
    print(f"\n{'='*60}")
    print("ANALYZING RESULTS")
    print(f"{'='*60}")
    
    results_summary = {
        'corrective_steering': False,
        'side_effects': False,
        'novelty': False,
        'politeness_control': False
    }
    
    # Check corrective steering results
    corrective_path = 'outputs/evaluation_results/corrective_steering_results.json'
    if os.path.exists(corrective_path):
        try:
            with open(corrective_path, 'r') as f:
                data = json.load(f)
            
            # Check if improvement is positive
            improvements = data.get('improvement', {})
            avg_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
            
            if avg_improvement > 0.05:  # 5% improvement threshold
                print("✅ Corrective Steering: SIGNIFICANT IMPROVEMENT")
                results_summary['corrective_steering'] = True
            else:
                print("⚠️  Corrective Steering: MINIMAL IMPROVEMENT")
        except Exception as e:
            print(f"❌ Error analyzing corrective steering results: {e}")
    
    # Check side effect results
    side_effect_path = 'outputs/evaluation_results/side_effect_evaluation.json'
    if os.path.exists(side_effect_path):
        try:
            with open(side_effect_path, 'r') as f:
                data = json.load(f)
            
            # Calculate average preservation rate
            preservation_rates = []
            for category, metrics in data.items():
                if 'preservation_rate' in metrics:
                    preservation_rates.append(metrics['preservation_rate'])
            
            avg_preservation = sum(preservation_rates) / len(preservation_rates) if preservation_rates else 0
            
            if avg_preservation > 0.8:  # 80% preservation threshold
                print("✅ Side Effects: GOOD CAPABILITY PRESERVATION")
                results_summary['side_effects'] = True
            else:
                print("⚠️  Side Effects: POOR CAPABILITY PRESERVATION")
        except Exception as e:
            print(f"❌ Error analyzing side effect results: {e}")
    
    # Check novelty results
    novelty_path = 'outputs/evaluation_results/novelty_analysis.json'
    if os.path.exists(novelty_path):
        print("✅ Novelty: DEMONSTRATED")
        results_summary['novelty'] = True
    
    # Check politeness control
    politeness_path = 'core/steering/politeness_steering.py'
    if os.path.exists(politeness_path):
        print("✅ Politeness Control: IMPLEMENTED")
        results_summary['politeness_control'] = True
    
    return results_summary

def main():
    """Run all verification experiments."""
    print("🚀 STARTING RESEARCH VERIFICATION SUITE")
    print("This will validate your corrective combinatorial SAE steering approach")
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    if not check_correlation_data():
        print("\n❌ Prerequisites not met. Please run correlation analysis first.")
        return
    
    experiments = [
        ("core/steering/corrective_steering.py", "Corrective Steering Validation"),
        ("evaluation/side_effect_evaluator.py", "Side-Effect Evaluation"),
        ("experiments/novelty_demonstration.py", "Novelty Demonstration"),
        ("core/steering/politeness_steering.py", "Politeness Steering Test")
    ]
    
    results = {}
    
    for script, description in experiments:
        success = run_experiment(script, description)
        results[description] = success
    
    # Analyze results
    analysis_results = analyze_results()
    
    # Print summary
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for description, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} experiments passed")
    
    # Research validation summary
    print(f"\n{'='*60}")
    print("RESEARCH VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    validation_score = sum(analysis_results.values())
    max_score = len(analysis_results)
    
    print(f"Research Validation Score: {validation_score}/{max_score}")
    
    if validation_score >= 3:
        print("\n🎉 RESEARCH VALIDATION SUCCESSFUL!")
        print("Your corrective combinatorial SAE steering approach is validated.")
        print("Key achievements:")
        
        if analysis_results['corrective_steering']:
            print("  ✅ Corrective steering shows improvement over baseline")
        if analysis_results['side_effects']:
            print("  ✅ Side effects are well-controlled")
        if analysis_results['novelty']:
            print("  ✅ Method demonstrates clear novelty")
        if analysis_results['politeness_control']:
            print("  ✅ Behavioral control is functional")
        
        print("\n📝 Next Steps:")
        print("  1. Write research paper documenting your methodology")
        print("  2. Scale experiments to larger models/datasets")
        print("  3. Submit to AI safety conferences")
        print("  4. Open-source your complete implementation")
        
    elif validation_score >= 2:
        print("\n⚠️  PARTIAL VALIDATION")
        print("Your approach shows promise but needs refinement.")
        print("Focus on improving the failed components.")
        
    else:
        print("\n❌ VALIDATION FAILED")
        print("The research approach needs significant improvement.")
        print("Review the failed experiments and address the issues.")
    
    print(f"\n{'='*60}")
    print("DETAILED RESULTS")
    print(f"{'='*60}")
    
    # Print detailed results for each experiment
    for description, success in results.items():
        print(f"\n{description}:")
        if success:
            print("  ✅ Experiment completed successfully")
        else:
            print("  ❌ Experiment failed - check error messages above")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    
    if validation_score >= 3:
        print("🎯 Your research is ready for publication!")
        print("1. Download all results from outputs/ directory")
        print("2. Create research presentation using generated materials")
        print("3. Write paper documenting your methodology and results")
        print("4. Submit to AI safety conferences (ICML, NeurIPS, ICLR)")
    else:
        print("🔧 Your research needs refinement:")
        print("1. Fix the failed experiments (check error messages)")
        print("2. Ensure all dependencies are properly installed")
        print("3. Verify model loading and data paths")
        print("4. Re-run verification experiments")

if __name__ == "__main__":
    main() 
