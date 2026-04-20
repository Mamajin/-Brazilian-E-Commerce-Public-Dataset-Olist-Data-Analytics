from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import community as community_louvain # python-louvain
import numpy as np

def cluster_with_kmeans(embeddings_dict, n_clusters=10):
    """
    Groups product embeddings into N zones using K-Means.
    """
    # Extract vectors and keys
    product_ids = list(embeddings_dict.keys())
    vectors = np.array([embeddings_dict[pid] for pid in product_ids])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    # Calculate silhouette score
    if n_clusters > 1:
        score = silhouette_score(vectors, cluster_labels)
    else:
        score = 0
        
    # Map back to dict
    zones = {pid: int(label) for pid, label in zip(product_ids, cluster_labels)}
    
    return zones, score

def cluster_with_louvain(graph):
    """
    Groups products using the Louvain method directly on the graph structure.
    Used for modularity evaluation.
    """
    partition = community_louvain.best_partition(graph, weight='weight')
    return partition

def get_modularity(graph, partition):
    """
    Computes modularity score for the given partition.
    """
    return community_louvain.modularity(partition, graph, weight='weight')

if __name__ == "__main__":
    # Test stub
    import networkx as nx
    G = nx.fast_gnp_random_graph(20, 0.2)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    
    partition = cluster_with_louvain(G)
    mod = get_modularity(G, partition)
    print(f"Louvain Modularity: {mod:.4f}")
    print(f"Zones: {set(partition.values())}")
