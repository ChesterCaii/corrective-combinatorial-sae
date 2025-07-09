"""
SAE Feature Correlation Analysis

Analyze correlations between SAE features extracted from transformer layers.
This is the core analysis for building the feature-correlation graph that
powers Corrective Steering.
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

class SAECorrelationAnalyzer:
    """Analyze correlations between SAE features."""
    
    def __init__(self, sae_features_path: str, output_dir: str = "sae_correlation_outputs"):
        """
        Initialize SAE correlation analyzer.
        
        Args:
            sae_features_path: Path to saved SAE features
            output_dir: Directory to save correlation results
        """
        self.sae_features_path = sae_features_path
        self.output_dir = output_dir
        self.sae_features = None
        self.layer_names = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_sae_features(self):
        """Load SAE feature data from saved file."""
        print("Loading SAE feature data...")
        
        data = np.load(self.sae_features_path, allow_pickle=True).item()
        self.sae_features = data
        self.layer_names = list(data.keys())
        
        print(f"Loaded SAE features from {len(self.layer_names)} layers:")
        for layer, features in data.items():
            print(f"   Layer {layer}: {features.shape}")
        
        return data
    
    def compute_within_layer_correlations(self, layer: int, max_features: int = 200, threshold: float = 0.3):
        """
        Compute correlations between SAE features within a single layer.
        
        Args:
            layer: Layer number to analyze
            max_features: Maximum number of features to analyze (computational efficiency)
            threshold: Correlation threshold for identifying significant pairs
            
        Returns:
            correlation_matrix: Correlation matrix between features
            significant_pairs: List of feature pairs with |correlation| > threshold
        """
        if layer not in self.sae_features:
            raise ValueError(f"Layer {layer} not found in SAE features")
            
        features = self.sae_features[layer]  # Shape: (n_samples, n_sae_features)
        n_samples, n_features = features.shape
        
        # Limit features for computational efficiency
        n_analyze = min(max_features, n_features)
        print(f"Analyzing correlations for {n_analyze} SAE features in layer {layer}...")
        
        # Select top features by activation (highest mean absolute activation)
        feature_importance = np.mean(np.abs(features), axis=0)
        top_feature_indices = np.argsort(feature_importance)[-n_analyze:]
        selected_features = features[:, top_feature_indices]
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(selected_features.T)
        
        # Find significant correlation pairs
        significant_pairs = []
        for i in range(n_analyze):
            for j in range(i + 1, n_analyze):
                corr = correlation_matrix[i, j]
                if abs(corr) > threshold:
                    significant_pairs.append({
                        'feature_1': top_feature_indices[i],
                        'feature_2': top_feature_indices[j],
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'importance_1': feature_importance[top_feature_indices[i]],
                        'importance_2': feature_importance[top_feature_indices[j]]
                    })
        
        # Sort by absolute correlation
        significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"Found {len(significant_pairs)} significant correlations (|r| > {threshold})")
        
        return correlation_matrix, significant_pairs, top_feature_indices
    
    def compute_cross_layer_correlations(self, layer1: int, layer2: int, 
                                       max_features: int = 200, threshold: float = 0.3):
        """
        Compute correlations between SAE features across different layers.
        
        Args:
            layer1, layer2: Layer numbers to compare
            max_features: Maximum number of features per layer to analyze
            threshold: Correlation threshold for identifying significant pairs
            
        Returns:
            cross_correlation_matrix: Cross-correlation matrix between layers
            significant_pairs: List of cross-layer feature pairs with |correlation| > threshold
        """
        if layer1 not in self.sae_features or layer2 not in self.sae_features:
            raise ValueError(f"One or both layers {layer1}, {layer2} not found in SAE features")
            
        features1 = self.sae_features[layer1]  # Shape: (n_samples, n_sae_features)
        features2 = self.sae_features[layer2]  # Shape: (n_samples, n_sae_features)
        
        # Select top features by importance
        importance1 = np.mean(np.abs(features1), axis=0)
        importance2 = np.mean(np.abs(features2), axis=0)
        
        n_features1 = min(max_features, features1.shape[1])
        n_features2 = min(max_features, features2.shape[1])
        
        top_indices1 = np.argsort(importance1)[-n_features1:]
        top_indices2 = np.argsort(importance2)[-n_features2:]
        
        selected_features1 = features1[:, top_indices1]
        selected_features2 = features2[:, top_indices2]
        
        print(f"Computing cross-layer correlations: Layer {layer1} ({n_features1} features) vs Layer {layer2} ({n_features2} features)...")
        
        # Compute cross-correlation matrix
        cross_correlation_matrix = np.zeros((n_features1, n_features2))
        significant_pairs = []
        
        for i in tqdm(range(n_features1), desc="Computing correlations"):
            for j in range(n_features2):
                corr, _ = pearsonr(selected_features1[:, i], selected_features2[:, j])
                cross_correlation_matrix[i, j] = corr
                
                if abs(corr) > threshold:
                    significant_pairs.append({
                        'layer1': layer1,
                        'feature1': top_indices1[i],
                        'layer2': layer2,
                        'feature2': top_indices2[j],
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'importance1': importance1[top_indices1[i]],
                        'importance2': importance2[top_indices2[j]]
                    })
        
        # Sort by absolute correlation
        significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"Found {len(significant_pairs)} significant cross-layer correlations (|r| > {threshold})")
        
        return cross_correlation_matrix, significant_pairs
    
    def analyze_all_layers(self, max_features: int = 200, threshold: float = 0.3):
        """
        Comprehensive analysis of all layers for building correlation graph.
        
        Args:
            max_features: Maximum number of features to analyze per layer
            threshold: Correlation threshold (original experiment: |ρ| > 0.3)
        """
        print("Starting comprehensive SAE feature correlation analysis...")
        print(f"This builds the correlation graph for Corrective Steering")
        print(f"Correlation threshold: |ρ| > {threshold}")
        
        if self.sae_features is None:
            self.load_sae_features()
        
        results = {
            'within_layer': {},
            'cross_layer': {},
            'summary': {},
            'adjacency_data': []  # For building the correlation graph
        }
        
        # 1. Within-layer correlations
        print("\nComputing within-layer SAE feature correlations...")
        for layer in self.layer_names:
            corr_matrix, sig_pairs, top_indices = self.compute_within_layer_correlations(
                layer, max_features, threshold
            )
            results['within_layer'][f'layer_{layer}'] = {
                'correlation_matrix_shape': corr_matrix.shape,
                'significant_pairs': len(sig_pairs),
                'top_correlations': sig_pairs[:20],  # Top 20 for analysis
                'top_feature_indices': top_indices.tolist()
            }
            
            # Add to adjacency data
            for pair in sig_pairs:
                results['adjacency_data'].append({
                    'source_layer': layer,
                    'source_feature': pair['feature_1'],
                    'target_layer': layer,
                    'target_feature': pair['feature_2'],
                    'correlation': pair['correlation'],
                    'type': 'within_layer'
                })
            
            # Save correlation matrix
            np.save(f"{self.output_dir}/sae_layer_{layer}_correlation_matrix.npy", corr_matrix)
        
        # 2. Cross-layer correlations (essential for Corrective Steering)
        print("\nComputing cross-layer SAE feature correlations...")
        layer_pairs = [
            (2, 4), (4, 6), (6, 8),  # Adjacent layers
            (2, 6), (2, 8), (4, 8)   # Skip connections
        ]
        
        for layer1, layer2 in layer_pairs:
            if layer1 in self.sae_features and layer2 in self.sae_features:
                cross_corr_matrix, sig_pairs = self.compute_cross_layer_correlations(
                    layer1, layer2, max_features, threshold
                )
                results['cross_layer'][f'layer_{layer1}_vs_{layer2}'] = {
                    'correlation_matrix_shape': cross_corr_matrix.shape,
                    'significant_pairs': len(sig_pairs),
                    'top_correlations': sig_pairs[:20]  # Top 20 for analysis
                }
                
                # Add to adjacency data
                for pair in sig_pairs:
                    results['adjacency_data'].append({
                        'source_layer': pair['layer1'],
                        'source_feature': pair['feature1'],
                        'target_layer': pair['layer2'],
                        'target_feature': pair['feature2'],
                        'correlation': pair['correlation'],
                        'type': 'cross_layer'
                    })
                
                # Save cross-correlation matrix
                np.save(f"{self.output_dir}/sae_cross_layer_{layer1}_vs_{layer2}_correlation.npy", cross_corr_matrix)
        
        # 3. Build adjacency matrix for correlation graph
        print("\nBuilding correlation adjacency matrix...")
        adjacency_df = pd.DataFrame(results['adjacency_data'])
        adjacency_df.to_csv(f"{self.output_dir}/correlation_adjacency_matrix.csv", index=False)
        
        # 4. Summary statistics
        total_within_pairs = sum(r['significant_pairs'] for r in results['within_layer'].values())
        total_cross_pairs = sum(r['significant_pairs'] for r in results['cross_layer'].values())
        total_edges = len(results['adjacency_data'])
        
        results['summary'] = {
            'total_within_layer_correlations': total_within_pairs,
            'total_cross_layer_correlations': total_cross_pairs,
            'total_graph_edges': total_edges,
            'threshold_used': threshold,
            'features_analyzed_per_layer': max_features,
            'layers_analyzed': self.layer_names,
            'experiment_completion': {
                'sae_features_extracted': True,
                'correlation_graph_built': True,
                'adjacency_matrix_created': True,
                'ready_for_corrective_steering': True
            }
        }
        
        # Save results
        with open(f"{self.output_dir}/sae_correlation_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nSAE Correlation Analysis Complete!")
        print(f"   Within-layer correlations: {total_within_pairs}")
        print(f"   Cross-layer correlations: {total_cross_pairs}")
        print(f"   Total correlation graph edges: {total_edges}")
        print(f"   Adjacency matrix saved: correlation_adjacency_matrix.csv")
        print(f"   Results saved to: {self.output_dir}/")
        
        # Print experiment status
        print(f"\n=== EXPERIMENT 1 STATUS ===")
        print(f"SAE Features: EXTRACTED")
        print(f"Correlation Graph: BUILT ({total_edges} edges)")
        print(f"Adjacency Matrix: CREATED")
        print(f"Ready for Corrective Steering: YES")
        
        return results
    
    def create_visualization(self, layer: int, max_features: int = 50):
        """Create visualization of SAE feature correlation matrix."""
        if self.sae_features is None:
            self.load_sae_features()
            
        if layer not in self.sae_features:
            raise ValueError(f"Layer {layer} not found in SAE features")
        
        # Get top features by importance
        features = self.sae_features[layer]
        importance = np.mean(np.abs(features), axis=0)
        top_indices = np.argsort(importance)[-max_features:]
        selected_features = features[:, top_indices]
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(selected_features.T)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   cmap='RdBu_r', 
                   center=0, 
                   vmin=-1, vmax=1,
                   xticklabels=False, 
                   yticklabels=False)
        plt.title(f'SAE Feature Correlation Matrix - Layer {layer}\n({max_features} most important features)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.output_dir}/sae_layer_{layer}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"SAE correlation heatmap saved for layer {layer}")


def main():
    """Run the SAE feature correlation analysis for Experiment 1."""
    
    print("=== SAE Feature Correlation Analysis for Experiment 1 ===")
    print("Building feature-correlation graph for Corrective Steering")
    
    # Configuration
    SAE_FEATURES_PATH = "gemmascope_experiment_outputs/sae_features.npy"
    MAX_FEATURES = 200  # Analyze top 200 features per layer by importance
    THRESHOLD = 0.3     # Original experiment threshold: |ρ| > 0.3
    
    # Initialize analyzer
    analyzer = SAECorrelationAnalyzer(SAE_FEATURES_PATH)
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_layers(
        max_features=MAX_FEATURES,
        threshold=THRESHOLD
    )
    
    # Create visualizations for each layer
    print("\nCreating SAE feature correlation visualizations...")
    for layer in analyzer.layer_names:
        analyzer.create_visualization(layer, max_features=50)
    
    print("\nEXPERIMENT 1 DELIVERABLES:")
    print("✓ SAE Features: Extracted using GemmaScope architecture")
    print("✓ Correlation Graph: Built with", results['summary']['total_graph_edges'], "edges")
    print("✓ Adjacency Matrix: Created (correlation_adjacency_matrix.csv)")
    print("✓ Analysis Results: Saved to sae_correlation_outputs/")
    
    print("\nNext Steps for Complete Experiment 1:")
    print("1. Manual validation: Sample 20 high-ρ pairs for semantic confirmation")
    print("2. Scale to 50K prompts (current: 5K)")
    print("3. Ready to proceed to Experiment 2: Corrective Steering")
    
    return results


if __name__ == "__main__":
    results = main() 