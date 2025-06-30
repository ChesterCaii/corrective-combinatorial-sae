"""
Activation Correlation Analysis

Analyze correlations between activation features from our extracted
transformer layer activations (layers 2, 4, 6, 8).

This is an adapted version for our current pipeline that works with
the activation data we have, before we integrate with GemmaScope SAE.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import json

class ActivationCorrelationAnalyzer:
    """Analyze correlations between activation features."""
    
    def __init__(self, activations_path: str, output_dir: str = "activation_correlation_outputs"):
        """
        Initialize activation correlation analyzer.
        
        Args:
            activations_path: Path to saved activations (from fast_extractor.py)
            output_dir: Directory to save correlation results
        """
        self.activations_path = activations_path
        self.output_dir = output_dir
        self.activations = None
        self.layer_names = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_activations(self):
        """Load activation data from saved file."""
        print("📂 Loading activation data...")
        
        data = np.load(self.activations_path, allow_pickle=True).item()
        self.activations = data
        self.layer_names = list(data.keys())
        
        print(f"✅ Loaded activations from {len(self.layer_names)} layers:")
        for layer, acts in data.items():
            print(f"   Layer {layer}: {acts.shape}")
        
        return data
    
    def compute_within_layer_correlations(self, layer: int, max_features: int = 100, threshold: float = 0.3):
        """
        Compute correlations between features within a single layer.
        
        Args:
            layer: Layer number to analyze
            max_features: Maximum number of features to analyze (for computational efficiency)
            threshold: Correlation threshold for identifying significant pairs
            
        Returns:
            correlation_matrix: Correlation matrix between features
            significant_pairs: List of feature pairs with |correlation| > threshold
        """
        if layer not in self.activations:
            raise ValueError(f"Layer {layer} not found in activations")
            
        activations = self.activations[layer]  # Shape: (n_samples, n_features)
        n_samples, n_features = activations.shape
        
        # Limit features for computational efficiency
        n_analyze = min(max_features, n_features)
        print(f"🔍 Analyzing correlations for {n_analyze} features in layer {layer}...")
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(activations[:, :n_analyze].T)
        
        # Find significant correlation pairs
        significant_pairs = []
        for i in range(n_analyze):
            for j in range(i + 1, n_analyze):
                corr = correlation_matrix[i, j]
                if abs(corr) > threshold:
                    significant_pairs.append({
                        'feature_1': i,
                        'feature_2': j,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        # Sort by absolute correlation
        significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"✅ Found {len(significant_pairs)} significant correlations (|r| > {threshold})")
        
        return correlation_matrix, significant_pairs
    
    def compute_cross_layer_correlations(self, layer1: int, layer2: int, 
                                       max_features: int = 100, threshold: float = 0.3):
        """
        Compute correlations between features across different layers.
        
        Args:
            layer1, layer2: Layer numbers to compare
            max_features: Maximum number of features per layer to analyze
            threshold: Correlation threshold for identifying significant pairs
            
        Returns:
            cross_correlation_matrix: Cross-correlation matrix between layers
            significant_pairs: List of cross-layer feature pairs with |correlation| > threshold
        """
        if layer1 not in self.activations or layer2 not in self.activations:
            raise ValueError(f"One or both layers {layer1}, {layer2} not found in activations")
            
        acts1 = self.activations[layer1]  # Shape: (n_samples, n_features)
        acts2 = self.activations[layer2]  # Shape: (n_samples, n_features)
        
        n_features1 = min(max_features, acts1.shape[1])
        n_features2 = min(max_features, acts2.shape[1])
        
        print(f"🔍 Computing cross-layer correlations: Layer {layer1} ({n_features1} features) vs Layer {layer2} ({n_features2} features)...")
        
        # Compute cross-correlation matrix
        cross_correlation_matrix = np.zeros((n_features1, n_features2))
        significant_pairs = []
        
        for i in tqdm(range(n_features1), desc="Computing correlations"):
            for j in range(n_features2):
                corr, _ = pearsonr(acts1[:, i], acts2[:, j])
                cross_correlation_matrix[i, j] = corr
                
                if abs(corr) > threshold:
                    significant_pairs.append({
                        'layer1': layer1,
                        'feature1': i,
                        'layer2': layer2,
                        'feature2': j,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        # Sort by absolute correlation
        significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"✅ Found {len(significant_pairs)} significant cross-layer correlations (|r| > {threshold})")
        
        return cross_correlation_matrix, significant_pairs
    
    def analyze_all_layers(self, max_features: int = 100, threshold: float = 0.3):
        """
        Comprehensive analysis of all layers.
        
        Args:
            max_features: Maximum number of features to analyze per layer
            threshold: Correlation threshold
        """
        print("🚀 Starting comprehensive activation correlation analysis...")
        
        if self.activations is None:
            self.load_activations()
        
        results = {
            'within_layer': {},
            'cross_layer': {},
            'summary': {}
        }
        
        # 1. Within-layer correlations
        print("\n📊 Computing within-layer correlations...")
        for layer in self.layer_names:
            corr_matrix, sig_pairs = self.compute_within_layer_correlations(
                layer, max_features, threshold
            )
            results['within_layer'][f'layer_{layer}'] = {
                'correlation_matrix_shape': corr_matrix.shape,
                'significant_pairs': len(sig_pairs),
                'top_correlations': sig_pairs[:10]  # Top 10
            }
            
            # Save correlation matrix
            np.save(f"{self.output_dir}/layer_{layer}_correlation_matrix.npy", corr_matrix)
        
        # 2. Cross-layer correlations
        print("\n🔗 Computing cross-layer correlations...")
        layer_pairs = [
            (2, 4), (4, 6), (6, 8),  # Adjacent layers
            (2, 6), (2, 8), (4, 8)   # Skip connections
        ]
        
        for layer1, layer2 in layer_pairs:
            if layer1 in self.activations and layer2 in self.activations:
                cross_corr_matrix, sig_pairs = self.compute_cross_layer_correlations(
                    layer1, layer2, max_features, threshold
                )
                results['cross_layer'][f'layer_{layer1}_vs_{layer2}'] = {
                    'correlation_matrix_shape': cross_corr_matrix.shape,
                    'significant_pairs': len(sig_pairs),
                    'top_correlations': sig_pairs[:10]  # Top 10
                }
                
                # Save cross-correlation matrix
                np.save(f"{self.output_dir}/cross_layer_{layer1}_vs_{layer2}_correlation.npy", cross_corr_matrix)
        
        # 3. Summary statistics
        total_within_pairs = sum(r['significant_pairs'] for r in results['within_layer'].values())
        total_cross_pairs = sum(r['significant_pairs'] for r in results['cross_layer'].values())
        
        results['summary'] = {
            'total_within_layer_correlations': total_within_pairs,
            'total_cross_layer_correlations': total_cross_pairs,
            'threshold_used': threshold,
            'features_analyzed_per_layer': max_features,
            'layers_analyzed': self.layer_names
        }
        
        # Save results
        with open(f"{self.output_dir}/correlation_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📈 Analysis Complete!")
        print(f"   Within-layer correlations: {total_within_pairs}")
        print(f"   Cross-layer correlations: {total_cross_pairs}")
        print(f"   Results saved to: {self.output_dir}/")
        
        return results
    
    def create_visualization(self, layer: int, max_features: int = 50):
        """Create visualization of correlation matrix for a specific layer."""
        if self.activations is None:
            self.load_activations()
            
        if layer not in self.activations:
            raise ValueError(f"Layer {layer} not found in activations")
        
        # Compute correlation matrix
        activations = self.activations[layer]
        n_features = min(max_features, activations.shape[1])
        correlation_matrix = np.corrcoef(activations[:, :n_features].T)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   cmap='RdBu_r', 
                   center=0, 
                   vmin=-1, vmax=1,
                   xticklabels=False, 
                   yticklabels=False)
        plt.title(f'Feature Correlation Matrix - Layer {layer}\n({n_features} features)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.output_dir}/layer_{layer}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Correlation heatmap saved for layer {layer}")


def main():
    """Run the activation correlation analysis."""
    
    # Configuration
    ACTIVATIONS_PATH = "fast_heldout_codes.npy"
    MAX_FEATURES = 100  # Analyze first 100 features per layer for efficiency
    THRESHOLD = 0.3     # |correlation| > 0.3 considered significant
    
    # Initialize analyzer
    analyzer = ActivationCorrelationAnalyzer(ACTIVATIONS_PATH)
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_layers(
        max_features=MAX_FEATURES,
        threshold=THRESHOLD
    )
    
    # Create visualizations for each layer
    print("\n🎨 Creating visualizations...")
    for layer in analyzer.layer_names:
        analyzer.create_visualization(layer, max_features=50)
    
    print("\n🎯 Next Steps:")
    print("1. Review correlation analysis results in activation_correlation_outputs/")
    print("2. Examine high-correlation feature pairs for semantic relationships")
    print("3. Use results to guide SAE feature selection")
    print("4. Integrate with GemmaScope SAE for deeper analysis")
    
    return results


if __name__ == "__main__":
    results = main() 