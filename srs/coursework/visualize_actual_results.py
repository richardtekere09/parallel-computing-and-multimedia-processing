import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

class ActualResultsVisualizer:
    """Visualize actual results from your WANN and NEAT runs"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.images_dir = self.results_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_fitness_data(self):
        """Load fitness data from CSV files"""
        wann_fitness = None
        neat_fitness = None
        
        # Load WANN fitness
        wann_file = self.results_dir / 'wann_fitness.csv'
        if wann_file.exists():
            try:
                wann_fitness = np.loadtxt(wann_file, delimiter=',')
                print(f"✓ Loaded WANN fitness data: {len(wann_fitness)} generations")
            except Exception as e:
                print(f"✗ Error loading WANN data: {e}")
        else:
            print("✗ WANN fitness file not found")
        
        # Load NEAT fitness
        neat_file = self.results_dir / 'neat_fitness.csv'
        if neat_file.exists():
            try:
                neat_fitness = np.loadtxt(neat_file, delimiter=',')
                print(f"✓ Loaded NEAT fitness data: {len(neat_fitness)} generations")
            except Exception as e:
                print(f"✗ Error loading NEAT data: {e}")
        else:
            print("✗ NEAT fitness file not found")
        
        return wann_fitness, neat_fitness
    
    def plot_fitness_evolution(self):
        """Plot fitness evolution from actual runs"""
        wann_fitness, neat_fitness = self.load_fitness_data()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # WANN plot
        if wann_fitness is not None:
            axes[0].plot(wann_fitness, 'o-', linewidth=2, markersize=4, color='#e74c3c', label='WANN Fitness')
            axes[0].fill_between(range(len(wann_fitness)), wann_fitness, alpha=0.3, color='#e74c3c')
            axes[0].set_title('WANN Fitness Evolution\n(Actual Results)', fontweight='bold')
            axes[0].set_xlabel('Generation')
            axes[0].set_ylabel('Fitness Score')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Add statistics
            max_fitness = np.max(wann_fitness)
            final_fitness = wann_fitness[-1]
            axes[0].text(0.02, 0.98, f'Max: {max_fitness:.1f}\nFinal: {final_fitness:.1f}', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, 'No WANN data available\nRun WANN experiments first', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('WANN Fitness Evolution', fontweight='bold')
        
        # NEAT plot
        if neat_fitness is not None:
            axes[1].plot(neat_fitness, 's-', linewidth=2, markersize=4, color='#3498db', label='NEAT Fitness')
            axes[1].fill_between(range(len(neat_fitness)), neat_fitness, alpha=0.3, color='#3498db')
            axes[1].set_title('NEAT Fitness Evolution\n(Actual Results)', fontweight='bold')
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Fitness Score')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # Add statistics
            max_fitness = np.max(neat_fitness)
            final_fitness = neat_fitness[-1]
            axes[1].text(0.02, 0.98, f'Max: {max_fitness:.1f}\nFinal: {final_fitness:.1f}', 
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No NEAT data available\nRun NEAT experiments first', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('NEAT Fitness Evolution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'actual_fitness_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return wann_fitness, neat_fitness
    
    def create_performance_summary(self):
        """Create a summary of all available results"""
        print("\n=== Performance Summary ===")
        
        # Check what files exist
        files_to_check = [
            'wann_fitness.csv',
            'neat_fitness.csv', 
            'neat_best_genome.pkl',
            'generation_performance.csv',
            'performance_summary.json'
        ]
        
        existing_files = []
        for file in files_to_check:
            file_path = self.results_dir / file
            if file_path.exists():
                existing_files.append(file)
                size = file_path.stat().st_size
                print(f"✓ {file} ({size} bytes)")
            else:
                print(f"✗ {file} (not found)")
        
        return existing_files
    
    def plot_algorithm_comparison(self):
        """Compare WANN vs NEAT if both data exist"""
        wann_fitness, neat_fitness = self.load_fitness_data()
        
        if wann_fitness is not None and neat_fitness is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot both algorithms
            ax.plot(wann_fitness, 'o-', linewidth=2, label='WANN', color='#e74c3c', alpha=0.8)
            ax.plot(neat_fitness, 's-', linewidth=2, label='NEAT', color='#3498db', alpha=0.8)
            
            ax.set_title('WANN vs NEAT: Fitness Evolution Comparison\n(Actual Experimental Results)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add final comparison
            wann_final = wann_fitness[-1]
            neat_final = neat_fitness[-1]
            
            comparison_text = f'Final Fitness:\nWANN: {wann_final:.1f}\nNEAT: {neat_final:.1f}'
            if neat_final > wann_final:
                comparison_text += f'\nNEAT advantage: {neat_final - wann_final:.1f}'
            else:
                comparison_text += f'\nWANN advantage: {wann_final - neat_final:.1f}'
            
            ax.text(0.02, 0.98, comparison_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(self.images_dir / 'wann_vs_neat_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return True
        else:
            print("Cannot create comparison - need both WANN and NEAT data")
            return False

# Usage
if __name__ == "__main__":
    visualizer = ActualResultsVisualizer()
    
    print("Checking for actual experimental results...")
    existing_files = visualizer.create_performance_summary()
    
    print("\nGenerating visualizations from actual data...")
    wann_data, neat_data = visualizer.plot_fitness_evolution()
    
    # Try to create comparison
    visualizer.plot_algorithm_comparison()
    
    print(f"\nVisualizations saved to: {visualizer.images_dir}")
