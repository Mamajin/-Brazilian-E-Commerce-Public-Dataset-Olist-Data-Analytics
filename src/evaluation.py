import numpy as np
from sklearn.metrics import silhouette_score

def evaluate_clusters(embeddings_dict, cluster_labels_dict):
    """
    Evaluates the quality of clusters using Silhouette Coefficient.
    """
    product_ids = list(embeddings_dict.keys())
    # Ensure IDs match
    vectors = np.array([embeddings_dict[pid] for pid in product_ids])
    labels = np.array([cluster_labels_dict[pid] for pid in product_ids])
    
    if len(set(labels)) > 1:
        score = silhouette_score(vectors, labels)
    else:
        score = 0
    return score

def calculate_improvement(baseline_val, optimized_val):
    """
    Calculates percentage improvement.
    """
    if baseline_val == 0:
        return 0
    return ((baseline_val - optimized_val) / baseline_val) * 100
