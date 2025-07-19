"""
Knowledge Graph Visualization for Byte Latent Transformer (BLT)

This module creates and visualizes a knowledge graph representing the BLT architecture,
components, and relationships based on the repository structure and model design.
"""

import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


def visualize_knowledge_graph(G, save_plot=True, show_plot=True):
    """
    Visualize the knowledge graph with enhanced styling for BLT components.

    Args:
        G: NetworkX graph object
        save_plot: Whether to save the plot to a file
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(24, 18))  # Large figure for comprehensive view

    # Create layout with more spacing
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    # Calculate node sizes based on connections (centrality)
    centrality = nx.degree_centrality(G)
    node_sizes = [centrality[node] * 4000 + 1000 for node in G.nodes()]

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=node_sizes,
        font_size=10,
        font_weight="bold",
        alpha=0.9,
        arrowsize=20,
        arrowstyle="->",
    )

    # Draw edge labels for relationships
    edge_labels = nx.get_edge_attributes(G, "verb")
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_color="#E74C3C",  # Red for edge labels
            font_size=8,
            font_weight="bold",
        )

    plt.title(
        "Knowledge Graph",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    if save_plot:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/knowledge_graph.png", dpi=300, bbox_inches="tight")
        print("Knowledge graph visualization saved to outputs/knowledge_graph.png")

    if show_plot:
        plt.show()


def print_graph_stats(G):
    """Print statistics about the knowledge graph."""
    print("\n=== Knowledge Graph Statistics ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.3f}")

    print(f"\nNode types:")
    node_types = {}
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")

    print(f"\nMost connected nodes:")
    centrality = nx.degree_centrality(G)
    for node, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score:.3f}")


if __name__ == "__main__":
    # Load the pickled graph from outputs folder
    kg_path = "outputs/kg.pkl"

    if not os.path.exists(kg_path):
        print(f"Error: {kg_path} not found!")
        print("Please ensure you have a kg.pkl file in the outputs directory.")
        exit(1)

    with open(kg_path, "rb") as f:
        G = pickle.load(f)
    print(f"Knowledge graph loaded from {kg_path}")

    # Print graph statistics
    print_graph_stats(G)

    # Visualize the knowledge graph
    plt.figure(figsize=(20, 15))  # Large figure for the whole graph

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)  # k controls node spacing

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        font_size=9,
        alpha=0.85,
    )

    # Draw edge labels if edges have a 'verb' or 'label' attribute
    edge_attr = (
        "verb"
        if all("verb" in d for _, _, d in G.edges(data=True))
        else ("label" if all("label" in d for _, _, d in G.edges(data=True)) else None)
    )
    if edge_attr:
        edge_labels = nx.get_edge_attributes(G, edge_attr)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Full Knowledge Graph")
    plt.tight_layout()
    plt.show()
