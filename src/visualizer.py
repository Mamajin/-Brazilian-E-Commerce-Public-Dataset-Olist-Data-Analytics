import matplotlib.pyplot as plt
import numpy as np

def plot_warehouse_layout(layout, product_zones, title="Warehouse Layout"):
    """
    Visualizes the warehouse grid with products colored by their zone.
    """
    x = [loc[0] for loc in layout.values()]
    y = [loc[1] for loc in layout.values()]
    colors = [product_zones.get(pid, 0) for pid in layout.keys()]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=colors, cmap='tab20', s=10)
    plt.colorbar(label='Zone ID')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    return plt

def plot_distance_comparison(base_avg, opt_avg):
    """
    Barchart comparing average distances.
    """
    labels = ['Baseline (Random)', 'Optimized (Graph-Based)']
    values = [base_avg, opt_avg]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['gray', 'skyblue'])
    plt.ylabel('Average Picking Distance (Manhattan)')
    plt.title('Performance Comparison')
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}", ha='center', va='bottom')
        
    return plt

def plot_embedding_clusters(embeddings, labels, title="Product Clusters (t-SNE/UMAP Projection)"):
    """
    Visualizes high-dim vectors in 2D.
    """
    # Assuming embeddings are already projected to 2D for simplicity
    # or we use PCA/t-SNE inside this function.
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.title(title)
    plt.colorbar(label='Zone')
    return plt
