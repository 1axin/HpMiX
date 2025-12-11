import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from mixup import compute_similarity_groups, restricted_shuffle

def train_model(model, x, hg_1hop, hg_3hop, train_edge_indices, train_edge_labels, val_edge_indices, val_edge_labels,
                epochs=200, lr=0.01, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_auc = 0
    wait = 0
    best_features = None
    similarity_groups = compute_similarity_groups(x)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        torch.manual_seed(42)
        new_idb = restricted_shuffle(model.num_nodes, similarity_groups)
        edge_pred, _ = model(x, hg_1hop, hg_3hop, train_edge_indices, new_idb, mixup_flag=True)
        train_loss = criterion(edge_pred, train_edge_labels)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            edge_pred, features = model(x, hg_1hop, hg_3hop, val_edge_indices, mixup_flag=False)
            val_loss = criterion(edge_pred, val_edge_labels)
            val_pred_prob = F.softmax(edge_pred, dim=1)[:, 1].cpu().numpy()
            val_labels_np = val_edge_labels.cpu().numpy()
            val_acc = accuracy_score(val_labels_np, edge_pred.argmax(dim=1).cpu().numpy())
            val_auc = roc_auc_score(val_labels_np, val_pred_prob)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            wait = 0
            best_features = features
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1} with best AUC: {best_auc:.4f}")
                break

    return best_features