import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from scipy.stats import entropy, spearmanr
from sklearn.manifold import TSNE
import os
import pandas as pd


def load_hypergraph_edges(json_file):
    with open(json_file, "r") as f:
        edges = json.load(f)
    return [tuple(sorted(e)) for e in edges]


def hypergraph_to_graph(edges):
    """将超图边集转为普通图结构"""
    G = nx.Graph()
    for e in edges:
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                G.add_edge(e[i], e[j])
    return G


def compare_hypergraphs(file1, file2, save_dir="HyperMixup_Compare"):
    os.makedirs(save_dir, exist_ok=True)

    edges1 = load_hypergraph_edges(file1)
    edges2 = load_hypergraph_edges(file2)
    set1, set2 = set(edges1), set(edges2)


    node_counts1 = Counter([n for e in set1 for n in e])
    node_counts2 = Counter([n for e in set2 for n in e])
    common_nodes = sorted(list(set(node_counts1) | set(node_counts2)))

    deg1 = np.array([node_counts1.get(n, 0) for n in common_nodes])
    deg2 = np.array([node_counts2.get(n, 0) for n in common_nodes])

    corr, _ = spearmanr(deg1, deg2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.kdeplot(deg1, fill=True, color="#1f77b4")
    plt.title("Original 1-hop Node Degree Distribution")
    plt.xlabel("Degree"); plt.ylabel("Density")

    plt.subplot(1, 2, 2)
    sns.kdeplot(deg2, fill=True, color="#ff7f0e")
    plt.title("Mixup 1-hop Node Degree Distribution")
    plt.xlabel("Degree"); plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_distribution_separate.png"), dpi=300)
    plt.close()

    node_ids = np.arange(len(deg1))

    plt.figure(figsize=(10, 4))
    width = 0.4
    plt.bar(node_ids - width/2, deg1, width=width, label='Original', alpha=0.7)
    plt.bar(node_ids + width/2, deg2, width=width, label='Mixup', alpha=0.7)
    plt.xlabel("Node Index (unordered)")
    plt.ylabel("Degree")
    plt.title("Node Degree Comparison (Unordered)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_comparison_unordered.png"), dpi=300)
    plt.close()

    node_ids = np.arange(len(deg1))

    plt.figure(figsize=(10, 4))
    width = 0.4
    plt.bar(node_ids - width / 2, deg1, width=width, label='Original', alpha=0.7)
    plt.bar(node_ids + width / 2, deg2, width=width, label='Mixup', alpha=0.7)
    plt.xlabel("Node Index (unordered)")
    plt.ylabel("Degree")
    plt.title("Node Degree Comparison (Unordered)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_comparison_unordered.png"), dpi=300)
    plt.close()

    deg1_sorted_idx = np.argsort(deg1)
    deg1_sorted = deg1[deg1_sorted_idx]
    deg2_sorted = deg2[np.argsort(deg2)]

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(deg1_sorted)), deg1_sorted, label='Original', linewidth=2, color="#1f77b4")
    plt.plot(np.arange(len(deg2_sorted)), deg2_sorted, label='Mixup', linewidth=2, linestyle='--', color="#ff7f0e")
    plt.xlabel("Node Index (sorted by degree)")
    plt.ylabel("Degree")
    plt.title("Node Degree Comparison (Aligned by Degree)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_comparison_aligned_by_degree.png"), dpi=300)
    plt.close()

    G1, G2 = hypergraph_to_graph(set1), hypergraph_to_graph(set2)
    A1, A2 = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
    n = max(A1.shape[0], A2.shape[0])
    A1p, A2p = np.zeros((n, n)), np.zeros((n, n))
    A1p[:A1.shape[0], :A1.shape[1]] = A1
    A2p[:A2.shape[0], :A2.shape[1]] = A2

    emb_all = np.vstack([A1p, A2p])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    Y = tsne.fit_transform(emb_all)
    Y1, Y2 = Y[:n], Y[n:]

    plt.figure(figsize=(10, 5))
    plt.scatter(Y1[:, 0], Y1[:, 1], s=10, alpha=0.6, label="Original", color="#1f77b4")
    plt.scatter(Y2[:, 0], Y2[:, 1], s=10, alpha=0.6, label="Mixup", color="#ff7f0e")
    plt.legend()
    plt.title("Structural Embedding (t-SNE of Adjacency Matrix)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_structure_embedding.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    pos_orig = nx.spring_layout(G1, seed=42, k=0.5)
    nx.draw(
        G1, pos_orig,
        node_size=10,  # 节点大小
        node_color="#1f77b4",
        edge_color="#1f77b4",
        alpha=0.7,
        with_labels=False
    )
    plt.title("Original 1-hop Hypergraph Structure")

    plt.subplot(1, 2, 2)
    pos_mixup = nx.spring_layout(G2, seed=42, k=0.5)
    nx.draw(
        G2, pos_mixup,
        node_size=10,
        node_color="#ff7f0e",
        edge_color="#ff7f0e",
        alpha=0.7,
        with_labels=False
    )
    plt.title("Mixup 1-hop Hypergraph Structure")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "edges_visualization.png"), dpi=300)
    plt.close()

    print(f"\n✅ 所有对比图与指标文件已保存至: {save_dir}/")


if __name__ == "__main__":
    compare_hypergraphs("hg_1hop_edges.json", "hg_1hop_b_edges.json")
