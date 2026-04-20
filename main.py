import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from graph_utils import load_olist_data, build_bipartite_graph, project_to_product_graph, denoise_graph
from embedding_engine import Node2VecEngine
from clustering_engine import cluster_with_kmeans, cluster_with_louvain, get_modularity
from simulation import WarehouseSimulator
from visualizer import plot_warehouse_layout, plot_distance_comparison

def main():
    print("=== Graph-Based Warehouse Slotting Optimization ===")
    
    # 1. Data Loading
    data_dir = 'data'
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: Data directory '{data_dir}' is empty or does not exist.")
        print("Please place Olist CSV files in the 'data/' folder.")
        return

    print("Step 1: Loading Olist data...")
    items, df_products = load_olist_data(data_dir)
    if items is None: return

    # 2. Graph Construction
    print("Step 2: Constructing Bipartite Graph...")
    B, orders, prods = build_bipartite_graph(items)
    print(f"Graph nodes: {len(B.nodes())}, Edges: {len(B.edges())}")

    print("Step 3: Projecting to Product Graph (Jaccard Similarity)...")
    P = project_to_product_graph(B, prods, weight_type='jaccard')
    
    print("Step 4: Denoising Graph...")
    # Remove products with very few connections to focus on main patterns
    P_clean = denoise_graph(P, k=2) 
    print(f"Projected Graph nodes: {len(P_clean.nodes())}, Edges: {len(P_clean.edges())}")

    # 3. Node2Vec Embeddings
    print("Step 5: Generating Node2Vec Embeddings (Learning structure)...")
    n2v = Node2VecEngine(P_clean, dimensions=32, walk_length=20, num_walks=10)
    model = n2v.generate_embeddings(window=5, min_count=1, sg=1, workers=4)
    
    # Extract embeddings for clustering
    embeddings = {str(node): model.wv[str(node)] for node in P_clean.nodes()}

    # 4. Clustering & Zoning
    print("Step 6: Clustering products into storage zones...")
    zones, silhouette = cluster_with_kmeans(embeddings, n_clusters=12)
    print(f"Clustering complete. Silhouette Score: {silhouette:.4f}")
    
    # Louvain for modularity check
    partition = cluster_with_louvain(P_clean)
    modularity = get_modularity(P_clean, partition)
    print(f"Louvain Modularity: {modularity:.4f}")

    # 5. Simulation
    print("Step 7: Running Warehouse Picking Simulation...")
    sim_products = list(P_clean.nodes())
    simulator = WarehouseSimulator(sim_products, grid_size=(100, 100))
    
    # Prepare layouts
    baseline_layout = simulator.create_baseline_layout()
    optimized_layout = simulator.create_optimized_layout(zones)
    
    # Group products by order for simulation
    # Filter only orders that contain products present in our cleaned graph
    cleaned_prod_set = set(sim_products)
    order_groups = items.groupby('order_id')['product_id'].apply(list)
    sim_orders = [ord_prods for ord_prods in order_groups if any(p in cleaned_prod_set for p in ord_prods)]
    
    # Run simulation for 1,000 samples
    base_avg, base_total = simulator.run_simulation(sim_orders, baseline_layout, sample_size=1000)
    opt_avg, opt_total = simulator.run_simulation(sim_orders, optimized_layout, sample_size=1000)
    
    improvement = ((base_avg - opt_avg) / base_avg) * 100
    
    print("\n--- RESULTS ---")
    print(f"Baseline Avg Picking Distance:  {base_avg:.2f}")
    print(f"Optimized Avg Picking Distance: {opt_avg:.2f}")
    print(f"Travel Distance Reduction:      {improvement:.1f}%")
    print(f"Modularity Score:                {modularity:.4f}")
    print("----------------\n")

    # 6. Visualization
    print("Step 8: Generating visualizations...")
    os.makedirs('output', exist_ok=True)
    
    # Plot layout
    plot_warehouse_layout(optimized_layout, zones, title="Optimized Warehouse Slotting (By Zone)")
    plt.savefig('output/optimized_layout.png')
    plt.close()
    
    # Plot comparison
    plot_distance_comparison(base_avg, opt_avg)
    plt.savefig('output/performance_comparison.png')
    plt.close()
    
    print("Visualizations saved to 'output/'.")
    print("Process complete!")

if __name__ == "__main__":
    main()
