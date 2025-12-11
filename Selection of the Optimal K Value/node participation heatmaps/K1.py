import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dhg import Graph, Hypergraph

def build_participation_matrix(hg, num_nodes):
    mat = np.zeros((num_nodes, len(hg.e[0])))
    for j, edge in enumerate(hg.e[0]):
        for i in edge:
            mat[i][j] = 1
    return mat

def main(k=2):
    df = pd.read_csv("CENAtrain.csv", header=None)
    node_set = set(df[0]).union(set(df[1]))
    node_map = {node: i for i, node in enumerate(sorted(node_set))}
    num_nodes = len(node_map)
    edge_list = [(node_map[a], node_map[b]) for a, b in df.values]

    g = Graph(num_nodes, edge_list)
    hg_khop = Hypergraph.from_graph_kHop(g, k=k)

    mat = build_participation_matrix(hg_khop, num_nodes)
    plt.figure(figsize=(12, 6))
    sns.heatmap(mat, cmap='Blues', cbar=True)
    plt.xlabel("Hyperedges")
    plt.ylabel("Nodes")
    plt.title(f"Node-Hyperedge Participation Heatmap (k={k})")
    plt.tight_layout()
    plt.savefig(f"participation_heatmap_k{k}.png")
    plt.show()

if __name__ == "__main__":
    main(k=2)
