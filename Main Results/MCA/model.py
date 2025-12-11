import torch
import torch.nn.functional as F
from torch.nn import Linear, MultiheadAttention
from dhg.nn import HGNNConv
from mixup import compute_similarity_groups, restricted_shuffle, shuffle_hypergraph, mixup_hidden

class HyperMixupModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(HyperMixupModel, self).__init__()
        self.hgnn_layer1 = HGNNConv(in_channels, hidden_channels)
        self.hgnn_layer2 = HGNNConv(hidden_channels, hidden_channels)
        self.hgnn_layer3 = HGNNConv(hidden_channels, hidden_channels)
        self.hgnn_1hop = HGNNConv(hidden_channels, hidden_channels)
        self.hgnn_3hop = HGNNConv(hidden_channels, hidden_channels)
        self.hgnn_layer2_mix = HGNNConv(hidden_channels, hidden_channels)
        self.hgnn_layer3_mix = HGNNConv(hidden_channels, hidden_channels)
        self.mixup_attention = MultiheadAttention(hidden_channels, num_heads=2)
        self.channel_fusion_attention = MultiheadAttention(hidden_channels, num_heads=4)
        self.edge_classifier = Linear(hidden_channels * 2, out_channels)
        self.num_nodes = num_nodes
        self.mixup_alpha = 4.0
        self.beta = torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, x, hg_1hop, hg_3hop, edge_indices, new_idb=None, mixup_flag=True):
        h1 = self.hgnn_layer1(x, hg_1hop)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)
        h2 = self.hgnn_layer2(h1, hg_1hop) + h1
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.3, training=self.training)
        h3_channel1 = self.hgnn_layer3(h2, hg_1hop) + h2
        h3_channel1 = F.relu(h3_channel1)
        h3_channel1 = F.dropout(h3_channel1, p=0.3, training=self.training)

        if mixup_flag and new_idb is not None:
            hg_1hop_b = shuffle_hypergraph(hg_1hop, new_idb, self.num_nodes)
            hg_3hop_b = shuffle_hypergraph(hg_3hop, new_idb, self.num_nodes)
            h1_mix_1hop = mixup_hidden(self, h1, hg_1hop, hg_1hop_b, new_idb)
            h1_mix_3hop = mixup_hidden(self, h1, hg_3hop, hg_3hop_b, new_idb)
            h1_mix_stack = torch.stack([h1_mix_1hop, h1_mix_3hop], dim=0)
            h1_mix, _ = self.mixup_attention(h1_mix_stack, h1_mix_stack, h1_mix_stack)
            h1_mix = h1_mix.mean(dim=0)
            h2_mix = self.hgnn_layer2_mix(h1_mix, hg_1hop) + h1_mix
            h2_mix = F.relu(h2_mix)
            h2_mix = F.dropout(h2_mix, p=0.3, training=self.training)
            h3_channel2 = self.hgnn_layer3_mix(h2_mix, hg_1hop) + h2_mix
            h3_channel2 = F.relu(h3_channel2)
            h3_channel2 = F.dropout(h3_channel2, p=0.3, training=self.training)
            h3_channel2 = h3_channel2 * self.beta
        else:
            h3_channel2 = h3_channel1

        h3_stack = torch.stack([h3_channel1, h3_channel2], dim=0)
        h3, _ = self.channel_fusion_attention(h3_stack, h3_stack, h3_stack)
        h3 = h3.mean(dim=0)
        edge_emb = torch.cat([h3[edge_indices[0]], h3[edge_indices[1]]], dim=1)
        edge_pred = self.edge_classifier(edge_emb)
        return edge_pred, h3