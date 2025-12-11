import torch
import pandas as pd
import numpy as np
from itertools import combinations
from dhg import Graph, Hypergraph
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

def generate_hedrw_features(edge_list, hg_1hop, hg_3hop, num_nodes, feature_dim=32, num_walks=10, walk_length=5):
    np.random.seed(42)
    g = Graph(num_nodes, edge_list)
    adj = g.A.to_dense()
    hyperedges_1hop = hg_1hop.e[0]
    hyperedges_3hop = hg_3hop.e[0]
    deg_1hop = torch.zeros(num_nodes, dtype=torch.float32)
    deg_3hop = torch.zeros(num_nodes, dtype=torch.float32)
    for edge in hyperedges_1hop:
        for node in edge:
            deg_1hop[node] += 1
    for edge in hyperedges_3hop:
        for node in edge:
            deg_3hop[node] += 1
    hyper_weight = (deg_1hop + deg_3hop) / (deg_1hop.max() + deg_3hop.max() + 1e-6)
    cooc_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for _ in range(num_walks):
        current_nodes = np.random.choice(num_nodes, num_nodes)
        for step in range(walk_length):
            probs = torch.zeros(num_nodes, num_nodes)
            for v in range(num_nodes):
                neighbors = torch.where(adj[v] > 0)[0]
                if len(neighbors) == 0:
                    continue
                neighbor_weights = hyper_weight[neighbors] + 1e-6
                probs[v, neighbors] = neighbor_weights / neighbor_weights.sum()
            next_nodes = torch.multinomial(probs[current_nodes], 1).squeeze()
            cooc_matrix[current_nodes, next_nodes] += 1
            cooc_matrix[next_nodes, current_nodes] += 1
            current_nodes = next_nodes.numpy()
    cooc_matrix = cooc_matrix / (num_walks * walk_length + 1e-6)
    svd = TruncatedSVD(n_components=feature_dim)
    x = svd.fit_transform(cooc_matrix.numpy())
    x = torch.tensor(x, dtype=torch.float32)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    return x

def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=None, names=['node1', 'node2'])
    edge_list = [(row['node1'], row['node2']) for _, row in df.iterrows()]
    unique_nodes = sorted(set([n for e in edge_list for n in e]))
    num_nodes = len(unique_nodes)
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    reverse_mapping = {new_id: old_id for old_id, new_id in node_mapping.items()}
    edge_list_new = [(node_mapping[n1], node_mapping[n2]) for n1, n2 in edge_list]
    g = Graph(num_nodes, edge_list_new)
    hg_1hop = Hypergraph.from_graph_kHop(g, k=1)
    hg_3hop = Hypergraph.from_graph_kHop(g, k=3)
    pos_edges = torch.tensor(edge_list_new, dtype=torch.long).t()
    pos_edge_set = set(tuple(e) for e in edge_list_new)
    x = generate_hedrw_features(edge_list_new, hg_1hop, hg_3hop, num_nodes, feature_dim=32, num_walks=10, walk_length=5)
    distances = pairwise_distances(x.numpy(), metric='euclidean')
    all_possible_edges = list(combinations(range(num_nodes), 2))
    neg_candidates = [(i, j) for i, j in all_possible_edges if (i, j) not in pos_edge_set]
    neg_scores = [distances[i, j] for i, j in neg_candidates]
    num_neg_edges = len(edge_list_new)
    neg_indices = np.argsort(neg_scores)[:num_neg_edges]
    neg_edges = torch.tensor([neg_candidates[i] for i in neg_indices], dtype=torch.long).t()

    edge_indices = torch.cat([pos_edges, neg_edges], dim=1)
    edge_labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))], dim=0).long()
    train_edge_indices, val_edge_indices, train_edge_labels, val_edge_labels = train_test_split(
        edge_indices.t().cpu().numpy(), edge_labels.cpu().numpy(), test_size=0.1, random_state=42
    )
    train_edge_indices = torch.tensor(train_edge_indices, dtype=torch.long).t().to(edge_indices.device)
    val_edge_indices = torch.tensor(val_edge_indices, dtype=torch.long).t().to(edge_indices.device)
    train_edge_labels = torch.tensor(train_edge_labels, dtype=torch.long).to(edge_labels.device)
    val_edge_labels = torch.tensor(val_edge_labels, dtype=torch.long).to(edge_labels.device)
    return hg_1hop, hg_3hop, num_nodes, node_mapping, reverse_mapping, train_edge_indices, train_edge_labels, val_edge_indices, val_edge_labels, x