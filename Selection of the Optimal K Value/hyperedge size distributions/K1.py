import pandas as pd
import matplotlib.pyplot as plt
from dhg import Graph, Hypergraph

def main(k=2):
    df = pd.read_csv("CENAtrain.csv", header=None)
    node_set = set(df[0]).union(set(df[1]))
    node_map = {node: i for i, node in enumerate(sorted(node_set))}
    num_nodes = len(node_map)
    edge_list = [(node_map[a], node_map[b]) for a, b in df.values]
    g = Graph(num_nodes, edge_list)

    hg = Hypergraph.from_graph_kHop(g, k=k)
    sizes = [len(e) for e in hg.e[0]]

    plt.hist(sizes, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Hyperedge Size (number of nodes)")
    plt.ylabel("Count")
    plt.title(f"Hyperedge Size Distribution (k={k})")
    plt.tight_layout()
    plt.savefig(f"hyperedge_size_dist_k{k}.png")
    plt.show()

if __name__ == "__main__":
    main(k=2)
