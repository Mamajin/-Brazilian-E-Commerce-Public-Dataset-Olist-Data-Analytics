# Graph-Based Warehouse Slotting Optimization
 
A data analytics project that applies graph embedding techniques to optimize warehouse product placement based on real-world co-purchase patterns.
 
**Course:** Data Analytics (01219367) — Department of Computer Engineering, Kasetsart University  
**Instructor:** Asst. Prof. Dr. Kitsana Waiyamai  
**Team:** Nichakorn Chanajitpairee · Phasit Ruangmak · Panthut Ketphan
 
---
 
## Overview
 
Traditional warehouse slotting methods like ABC Analysis prioritize products by individual popularity, but ignore *co-purchase relationships* — the fact that items like phone cases and chargers are frequently bought together. Storing such products far apart increases picker travel distance and operational cost.
 
This project models product relationships as a graph, applies Node2Vec embeddings and Leiden community detection to discover product clusters, and simulates the effect of graph-based slotting on warehouse picking distance.
 
**Key result: 18.22% reduction in simulated picking distance** compared to a random layout baseline.
 
---
 
## Methodology
 
```
Raw Transactions (Olist)
        ↓
Bipartite Graph (Orders × Products)
        ↓
Product–Product Projection (co-occurrence edges, Jaccard-weighted)
        ↓
K-Core Denoising (k=2) → removes weak/noisy connections
        ↓
Node2Vec Embeddings (random walks + Skip-gram)
        ↓
Leiden Clustering → 7 warehouse zones
        ↓
Picking Distance Simulation (Manhattan distance, 13×13 grid)
```
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Modularity Score | 0.9048 (strong community structure) |
| Silhouette Score | 0.2511 (moderate embedding separation) |
| Random layout picking distance | 10,912 units |
| Graph-based layout picking distance | 8,924 units |
| **Improvement** | **18.22%** |
 
---
 
## Dataset
 
**Source:** [Brazilian E-Commerce Public Dataset (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) via Kaggle
 
| Stat | Value |
|---|---|
| Order-item records | 112,650 |
| Unique orders | 98,666 |
| Unique products | 32,951 |
| Graph density | 0.000084 |
 
> The dataset is not included in this repository. Download it from Kaggle and place the files in a `data/` directory before running the notebook.
 
---
 
## Setup & Usage
 
### Requirements
 
```bash
pip install networkx node2vec leidenalg igraph scikit-learn matplotlib pandas numpy
```
 
### Running the notebook
 
1. Clone the repository:
   ```bash
   git clone https://github.com/Mamajin/-Brazilian-E-Commerce-Public-Dataset-Olist-Data-Analytics.git
   cd -Brazilian-E-Commerce-Public-Dataset-Olist-Data-Analytics
   ```
 
2. Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place CSV files in a `data/` folder.
3. Open and run `Graph_Based_Warehouse_Optimization.ipynb` top to bottom.
---
 
## Node2Vec Configuration
 
| Parameter | Value |
|---|---|
| Dimensions | [fill in] |
| Walk length | [fill in] |
| Number of walks | [fill in] |
| p (return parameter) | [fill in] |
| q (in-out parameter) | [fill in] |
| Window size | [fill in] |
 
> Fill in the values above from your notebook. A lower *q* value biases walks toward outward (DFS-like) exploration, which helps capture community-level structure suited for warehouse zone discovery.
 
---
 
## Repository Structure
 
```
├── Graph_Based_Warehouse_Optimization.ipynb   # Main analysis notebook
├── README.md
├── LICENSE
└── .gitignore
```
 
---
 
## References
 
1. A. Grover and J. Leskovec, "node2vec: Scalable Feature Learning for Networks," *KDD*, 2016.
2. V. A. Traag, L. Waltman, and N. J. van Eck, "From Louvain to Leiden: guaranteeing well-connected communities," *Scientific Reports*, 2019.
3. F. Olist, "Brazilian E-Commerce Public Dataset," Kaggle, 2018.
4. NetworkX Documentation: https://networkx.org/documentation/stable/
5. Stanford CS224W: Machine Learning with Graphs — http://cs224w.stanford.edu
---
 
## License
 
This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
