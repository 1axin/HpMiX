import torch
import numpy as np
from dhg import Hypergraph
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_groups(features, n_clusters=None):
    num_nodes = features.shape[0]
    if n_clusters is None:
        n_clusters = int(np.sqrt(num_nodes))
    similarity_matrix = cosine_similarity(features.cpu().numpy())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(similarity_matrix)
    groups = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        groups[label].append(i)
    return groups

def restricted_shuffle(num_nodes, groups):

    np.random.seed(42)
    new_idb = np.arange(num_nodes)
    for group in groups:
        group_indices = np.array(group)
        np.random.shuffle(group_indices)
        new_idb[group] = group_indices
    return torch.tensor(new_idb, dtype=torch.long)

def shuffle_hypergraph(hg, new_idb, num_nodes):
    hyperedge_list = hg.e[0]
    id_map = new_idb.clone().detach().tolist()
    hyperedge_list_b = [[id_map[node] for node in edge] for edge in hyperedge_list]
    return Hypergraph(num_nodes, hyperedge_list_b)

def mixup_hidden(model, h, hg_orig, hg_shuffled, new_idb):
    h1 = model.hgnn_1hop(h, hg_orig)
    h1 = torch.nn.functional.relu(h1)
    h_b = h[new_idb]
    h2 = model.hgnn_3hop(h_b, hg_shuffled)
    h2 = torch.nn.functional.relu(h2)
    np.random.seed(42)
    lam = np.random.beta(model.mixup_alpha, model.mixup_alpha) if model.training else 1.0
    h2_aligned = h2[torch.argsort(new_idb)]
    h_mix = h1 * lam + h2_aligned * (1 - lam)
    h_mix = torch.nn.functional.relu(h_mix)
    return h_mix