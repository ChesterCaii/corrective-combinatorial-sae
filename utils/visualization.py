"""
Results Visualization for Research Presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

class ResultsVisualizer:
    """Create publication-ready visualizations of your results."""
    
    def __init__(self):
        self.output_dir = Path("outputs/visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_steering_comparison(self):
        """Plot your method vs SAS baseline."""
        try:
            with open('outputs/evaluation_results/corrective_steering_results.json', 'r') as f:
                data = json.load(f)
            
            # Extract data
            metrics = list(data['traditional_sas'].keys())
            sas_scores = [data['traditional_sas'][m] for m in metrics]
            your_scores = [data['corrective_steering'][m] for m in metrics]
            improvements = [data['improvement'][m] for m in metrics]
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar chart comparison
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, sas_scores, width, label='SAS Baseline', alpha=0.8)
            ax1.bar(x + width/2, your_scores, width, label='Your Method', alpha=0.8)
            ax1.set_xlabel('Capability Metrics')
            ax1.set_ylabel('Performance Score')
            ax1.set_title('Corrective Steering vs SAS Baseline')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Improvement plot
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax2.bar(metrics, improvements, color=colors, alpha=0.7)
            ax2.set_xlabel('Capability Metrics')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Improvement Over SAS Baseline')
            ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'steering_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Steering comparison plot saved")
            
        except Exception as e:
            print(f"❌ Error creating steering comparison plot: {e}")
    
    def plot_capability_preservation(self):
        """Plot capability preservation results."""
        try:
            with open('outputs/evaluation_results/side_effect_evaluation.json', 'r') as f:
                data = json.load(f)
            
            categories = list(data.keys())
            preservation_rates = [data[cat]['preservation_rate'] for cat in categories]
            degradations = [data[cat]['degradation'] for cat in categories]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Preservation rates
            colors = ['green' if rate > 0.9 else 'orange' if rate > 0.8 else 'red' for rate in preservation_rates]
            ax1.bar(categories, preservation_rates, color=colors, alpha=0.7)
            ax1.set_xlabel('Capability Categories')
            ax1.set_ylabel('Preservation Rate')
            ax1.set_title('Capability Preservation After Steering')
            ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
            ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (>90%)')
            ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good (>80%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Degradation plot
            ax2.bar(categories, degradations, color='red', alpha=0.7)
            ax2.set_xlabel('Capability Categories')
            ax2.set_ylabel('Degradation')
            ax2.set_title('Capability Degradation (Lower is Better)')
            ax2.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'capability_preservation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Capability preservation plot saved")
            
        except Exception as e:
            print(f"❌ Error creating capability preservation plot: {e}")
    
    def plot_correlation_graph_sample(self):
        """Plot a sample of your correlation graph."""
        try:
            df = pd.read_csv('outputs/correlation_graphs/correlation_adjacency_matrix.csv')
            
            # Sample high-correlation edges for visualization
            high_corr = df[abs(df['correlation']) > 0.8].head(50)
            
            # Create network visualization
            import networkx as nx
            
            G = nx.Graph()
            for _, row in high_corr.iterrows():
                G.add_edge(f"F{row['source_feature']}", f"F{row['target_feature']}", 
                          weight=abs(row['correlation']))
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw edges with correlation strength
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Reds, 
                                  width=2, alpha=0.7)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title('Sample of High-Correlation Feature Network\n(Your Steering Foundation)')
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), label='Correlation Strength')
            plt.axis('off')
            
            plt.savefig(self.output_dir / 'correlation_network.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Correlation network plot saved")
            
        except Exception as e:
            print(f"❌ Error creating correlation network plot: {e}")
    
    def create_research_summary(self):
        """Create a comprehensive research summary."""
        summary = {
            'correlation_graph': {},
            'steering_performance': {},
            'capability_preservation': {},
            'novelty_claims': {}
        }
        
        # Load correlation data
        try:
            df = pd.read_csv('outputs/correlation_graphs/correlation_adjacency_matrix.csv')
            summary['correlation_graph'] = {
                'total_edges': len(df),
                'high_correlation_edges': len(df[abs(df['correlation']) > 0.8]),
                'correlation_range': [df['correlation'].min(), df['correlation'].max()],
                'layers_analyzed': df['source_layer'].unique().tolist()
            }
        except Exception as e:
            print(f"Error loading correlation data: {e}")
        
        # Load steering results
        try:
            with open('outputs/evaluation_results/corrective_steering_results.json', 'r') as f:
                steering_data = json.load(f)
            
            improvements = steering_data.get('improvement', {})
            avg_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
            
            summary['steering_performance'] = {
                'average_improvement': avg_improvement,
                'improvements_by_metric': improvements
            }
        except Exception as e:
            print(f"Error loading steering results: {e}")
        
        # Load capability preservation
        try:
            with open('outputs/evaluation_results/side_effect_evaluation.json', 'r') as f:
                capability_data = json.load(f)
            
            preservation_rates = []
            for category, metrics in capability_data.items():
                if 'preservation_rate' in metrics:
                    preservation_rates.append(metrics['preservation_rate'])
            
            avg_preservation = sum(preservation_rates) / len(preservation_rates) if preservation_rates else 0
            
            summary['capability_preservation'] = {
                'average_preservation': avg_preservation,
                'preservation_by_category': capability_data
            }
        except Exception as e:
            print(f"Error loading capability data: {e}")
        
        # Load novelty claims
        try:
            with open('outputs/evaluation_results/novelty_analysis.json', 'r') as f:
                novelty_data = json.load(f)
            
            summary['novelty_claims'] = novelty_data.get('novelty_analysis', {})
        except Exception as e:
            print(f"Error loading novelty data: {e}")
        
        # Save summary
        with open(self.output_dir / 'research_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Research summary saved")
        return summary

def create_all_visualizations():
    """Create all visualizations for your research presentation."""
    print("🎨 Creating Research Visualizations...")
    
    visualizer = ResultsVisualizer()
    
    # Create all plots
    visualizer.plot_steering_comparison()
    visualizer.plot_capability_preservation()
    visualizer.plot_correlation_graph_sample()
    summary = visualizer.create_research_summary()
    
    print("\n🎨 All visualizations created!")
    print("📁 Check outputs/visualizations/ for your plots")
    
    # Print summary
    print("\n📈 Research Summary:")
    if 'correlation_graph' in summary:
        cg = summary['correlation_graph']
        print(f"  Correlation Graph: {cg.get('total_edges', 0)} edges")
    
    if 'steering_performance' in summary:
        sp = summary['steering_performance']
        print(f"  Average Improvement: {sp.get('average_improvement', 0):.1%}")
    
    if 'capability_preservation' in summary:
        cp = summary['capability_preservation']
        print(f"  Average Capability Preservation: {cp.get('average_preservation', 0):.1%}")

if __name__ == "__main__":
    create_all_visualizations() 
