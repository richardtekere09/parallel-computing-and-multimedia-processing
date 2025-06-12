import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from pathlib import Path
import neat

def load_neat_genome():
    """Load the NEAT genome"""
    genome_file = Path("results/neat_best_genome.pkl")
    if genome_file.exists():
        with open(genome_file, 'rb') as f:
            return pickle.load(f)
    return None

def analyze_genome_structure(genome):
    """Analyze the genome to understand its structure"""
    print("=== Genome Structure Analysis ===")
    print(f"Fitness: {genome.fitness}")
    print(f"Nodes: {len(genome.nodes)}")
    print(f"Connections: {len(genome.connections)}")
    
    # Print all nodes
    print("\nAll nodes:")
    for node_id, node in genome.nodes.items():
        print(f"  Node {node_id}: activation={node.activation}")
    
    # Print all connections
    print("\nAll connections:")
    for (src, dst), conn in genome.connections.items():
        status = "enabled" if conn.enabled else "disabled"
        print(f"  {src} -> {dst}: weight={conn.weight:.3f}, {status}")
    
    # Get all unique node IDs (from both nodes dict and connections)
    all_node_ids = set(genome.nodes.keys())
    for (src, dst), conn in genome.connections.items():
        all_node_ids.add(src)
        all_node_ids.add(dst)
    
    print(f"\nAll unique node IDs: {sorted(all_node_ids)}")
    return sorted(all_node_ids)

def simple_visualize(genome):
    """Simple visualization that just works"""
    if genome is None:
        print("No genome to visualize")
        return
    
    # Analyze structure first
    all_node_ids = analyze_genome_structure(genome)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add ALL nodes that appear anywhere
    for node_id in all_node_ids:
        G.add_node(node_id)
    
    # Add enabled connections
    for (src, dst), conn in genome.connections.items():
        if conn.enabled:
            G.add_edge(src, dst, weight=conn.weight)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Simple layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50)
    except:
        # Fallback to manual layout
        pos = {}
        for i, node_id in enumerate(all_node_ids):
            angle = 2 * np.pi * i / len(all_node_ids)
            pos[node_id] = (np.cos(angle), np.sin(angle))
    
    # Color nodes based on ID ranges (simple heuristic)
    node_colors = []
    node_labels = {}
    
    for node_id in G.nodes():
        # Simple coloring: negative = input, small positive = output, large = hidden
        if node_id < 0:
            node_colors.append('lightblue')  # Input
            node_labels[node_id] = f'I{node_id}'
        elif node_id < 10:  # Assuming outputs are 0-9
            node_colors.append('lightcoral')  # Output
            node_labels[node_id] = f'O{node_id}'
        else:
            node_colors.append('lightgreen')  # Hidden
            node_labels[node_id] = f'H{node_id}'
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8, ax=ax)
    
    # Draw edges
    if G.edges():
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_colors = ['red' if w < 0 else 'blue' for w in weights]
        max_weight = max(abs(w) for w in weights) if weights else 1
        edge_widths = [abs(w) / max_weight * 3 + 0.5 for w in weights]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                              alpha=0.6, arrows=True, arrowsize=20, ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_weight='bold', ax=ax)
    
    # Title
    ax.set_title(f'NEAT Network Topology\nFitness: {genome.fitness:.3f} | Nodes: {len(G.nodes())} | Connections: {len(G.edges())}', 
                fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Input Nodes (negative IDs)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markersize=10, label='Output Nodes (small positive IDs)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=10, label='Hidden Nodes (large positive IDs)'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Positive Weight'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Negative Weight')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save
    save_path = Path("results/images/neat_topology_simple.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Visualization saved to {save_path}")
    
    # Print final summary
    print(f"\n=== Network Summary ===")
    print(f"Total nodes: {len(G.nodes())}")
    print(f"Total connections: {len(G.edges())}")
    if G.edges():
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        print(f"Weight range: {min(weights):.3f} to {max(weights):.3f}")
        print(f"Positive weights: {sum(1 for w in weights if w > 0)}")
        print(f"Negative weights: {sum(1 for w in weights if w < 0)}")

def main():
    print("🔍 Loading NEAT genome...")
    genome = load_neat_genome()
    
    if genome:
        print("✓ Genome loaded successfully!")
        simple_visualize(genome)
        print("\n🎉 Visualization complete!")
    else:
        print("❌ No NEAT genome file found.")
        print("Run your NEAT experiment first: python main.py")

if __name__ == "__main__":
    main()
