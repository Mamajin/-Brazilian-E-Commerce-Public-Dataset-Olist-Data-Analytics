import random
import numpy as np
from gensim.models import Word2Vec
import networkx as nx

class Node2VecEngine:
    """
    Implementation of Node2Vec embedding generation.
    References: Grover & Leskovec (2016).
    """
    def __init__(self, graph, dimensions=64, walk_length=30, num_walks=10, p=1, q=1):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return parameter
        self.q = q  # In-out parameter
        self.is_directed = graph.is_directed()

    def node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from a node.
        """
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    next_node = self._get_next_node(prev, cur, cur_nbrs)
                    walk.append(next_node)
            else:
                break
        return [str(node) for node in walk]

    def _get_next_node(self, prev, cur, nbrs):
        """
        Biased random walk selection.
        """
        weights = []
        for nbr in nbrs:
            if nbr == prev:
                # Return to previous node
                weights.append(1/self.p)
            elif self.graph.has_edge(nbr, prev):
                # Node is connected to previous node
                weights.append(1)
            else:
                # Node is further away
                weights.append(1/self.q)
        
        # Normalize weights
        norm_weights = [float(i)/sum(weights) for i in weights]
        return random.choices(nbrs, weights=norm_weights, k=1)[0]

    def simulate_walks(self):
        """
        Generate random walks for all nodes.
        """
        nodes = list(self.graph.nodes())
        walks = []
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(node))
        return walks

    def generate_embeddings(self, **word2vec_kwargs):
        """
        Trains Word2Vec on the simulated walks.
        """
        walks = self.simulate_walks()
        model = Word2Vec(walks, vector_size=self.dimensions, **word2vec_kwargs)
        return model

if __name__ == "__main__":
    # Test with a simple graph
    G = nx.fast_gnp_random_graph(n=10, p=0.5)
    engine = Node2VecEngine(G)
    model = engine.generate_embeddings(window=5, min_count=0, sg=1, workers=4)
    print("Example embedding for node 0:", model.wv['0'][:5])
