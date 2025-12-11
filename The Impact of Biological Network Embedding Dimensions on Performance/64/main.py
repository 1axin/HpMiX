import warnings
import torch
import pandas as pd
import numpy as np
from model import HyperMixupModel
from utils import load_data_from_csv
from train import train_model
from FeatureProcess import match_features
from PredictionTask import predict_task
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 设置全局随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果使用 GPU
    np.random.seed(42)

    csv_file = "CENAtrain.csv"
    hg_1hop, hg_3hop, num_nodes, node_mapping, reverse_mapping, train_edge_indices, train_edge_labels, val_edge_indices, val_edge_labels, x = load_data_from_csv(
        csv_file)
    in_channels = 32
    hidden_channels = 64
    out_channels = 2

    # 保存初始特征
    initial_feature_data = []
    for i in range(num_nodes):
        original_id = reverse_mapping[i]
        features = x[i].detach().cpu().numpy().tolist()
        initial_feature_data.append([original_id] + features)
    initial_feature_df = pd.DataFrame(initial_feature_data,
                                      columns=['node_id'] + [f'feature{i + 1}' for i in range(in_channels)])
    initial_feature_df.to_csv("initial_features.csv", index=False)

    model = HyperMixupModel(in_channels, hidden_channels, out_channels, num_nodes)
    node_features = train_model(model, x, hg_1hop, hg_3hop, train_edge_indices, train_edge_labels,
                                val_edge_indices, val_edge_labels, epochs=200, lr=0.01, patience=5)

    feature_data_mixup = []
    for i in range(num_nodes):
        original_id = reverse_mapping[i]
        features = node_features[i].detach().cpu().numpy().tolist()
        feature_data_mixup.append([original_id] + features)

    sample_features = match_features("CCAprediction.csv", feature_data_mixup, feature_dim=hidden_channels)
    predict_task(sample_features, output_dir="results")
