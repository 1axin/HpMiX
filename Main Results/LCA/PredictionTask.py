import os
import warnings
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import numpy as np
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb
warnings.filterwarnings('ignore')

def predict_task(sample_features, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    def write_csv(data, file_name):
        with open(file_name, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    sample_features = np.array(sample_features, dtype=float)

    sample_labels = np.concatenate([np.ones(1066), np.zeros(1066)])

    np.random.seed(42)
    permutation = np.random.permutation(sample_labels.shape[0])
    sample_features = sample_features[permutation]
    sample_labels = sample_labels[permutation]

    print('Starting model training...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10000)
    mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 10000)
    mean_average_precision = []
    results = []
    y_test_all = []
    y_pred_all = []

    plt.figure(figsize=(8, 6))
    for i, (train, test) in enumerate(cv.split(sample_features, sample_labels)):
        model = lgb.LGBMClassifier(num_leaves=33, learning_rate=0.06, n_estimators=33, random_state=42)
        model.fit(sample_features[train], sample_labels[train])
        predicted = model.predict_proba(sample_features[test])
        predicted1 = model.predict(sample_features[test])

        fpr, tpr, _ = roc_curve(sample_labels[test], predicted[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1.5, alpha=1, label=f'ROC fold {i} (AUC = {roc_auc:.4f})')

        precision, recall, _ = precision_recall_curve(sample_labels[test], predicted[:, 1])
        average_precision = average_precision_score(sample_labels[test], predicted[:, 1])
        mean_average_precision.append(average_precision)
        mean_precision += interp(mean_recall, recall, precision)

        print(f"\n=== Fold {i+1} ===")
        print(f'Test accuracy: {accuracy_score(sample_labels[test], predicted1):.4f}')
        report = classification_report(sample_labels[test], predicted1, digits=4, output_dict=True)
        print(classification_report(sample_labels[test], predicted1, digits=4))
        print(confusion_matrix(sample_labels[test], predicted1))
        results.append(report)

        y_test_all.extend(sample_labels[test])
        y_pred_all.extend(predicted1)

        np.save(f'{output_dir}/Y_test_fold{i}.npy', sample_labels[test])
        np.save(f'{output_dir}/Y_pred_fold{i}.npy', predicted[:, 1])

    write_csv([[str(r)] for r in results], f'{output_dir}/result.csv')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})', lw=1.5)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.5, label='± 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/ROC-5fold-test.tif')
    plt.close()

    print(f"\nMean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean Average Precision: {np.mean(mean_average_precision):.4f}")
