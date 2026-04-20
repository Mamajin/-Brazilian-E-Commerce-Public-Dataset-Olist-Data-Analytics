import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import os

def load_olist_data(data_dir):
    """
    Loads necessary Olist datasets.
    """
    try:
        items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
        products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
        return items, products
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

def build_bipartite_graph(order_items_df):
    """
    Constructs a bipartite graph from order items.
    Nodes are either 'order_id' or 'product_id'.
    """
    B = nx.Graph()
    
    # Add nodes with 'bipartite' attribute
    orders = order_items_df['order_id'].unique()
    products = order_items_df['product_id'].unique()
    
    B.add_nodes_from(orders, bipartite=0)
    B.add_nodes_from(products, bipartite=1)
    
    # Add edges
    edges = list(zip(order_items_df['order_id'], order_items_df['product_id']))
    B.add_edges_from(edges)
    
    return B, orders, products

def project_to_product_graph(B, product_nodes, weight_type='jaccard'):
    """
    Projects the bipartite graph into a product-product graph.
    """
    # Overlap/Co-occurrence projection
    # weighted_projected_graph calculates weight based on shared neighbors
    P = bipartite.weighted_projected_graph(B, product_nodes)
    
    # Apply Jaccard similarity if requested
    if weight_type == 'jaccard':
        for u, v, d in P.edges(data=True):
            # Intersection is the weight from weighted_projected_graph (count of shared orders)
            intersection = d['weight']
            # Union = Degree(u) + Degree(v) - Intersection
            union = B.degree(u) + B.degree(v) - intersection
            d['weight'] = intersection / union if union > 0 else 0
            
    return P

def denoise_graph(P, k=None, min_weight=None):
    """
    Removes noise from the graph.
    - k: K-Core decomposition (removes nodes with degree < k)
    - min_weight: Removes edges with weight less than min_weight
    """
    # Filter by weight first
    if min_weight is not None:
        edges_to_remove = [(u, v) for u, v, d in P.edges(data=True) if d['weight'] < min_weight]
        P.remove_edges_from(edges_to_remove)
    
    # Filter isolated nodes
    P.remove_nodes_from(list(nx.isolates(P)))
    
    # K-Core
    if k is not None:
        P = nx.k_core(P, k=k)
        
    return P

if __name__ == "__main__":
    # Test stub (requires data/)
    items, products = load_olist_data('data')
    if items is not None:
        print(f"Loaded {len(items)} order items.")
        B, orders, prods = build_bipartite_graph(items)
        print(f"Bipartite graph built with {len(orders)} orders and {len(prods)} products.")
