import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os

# Use a non-interactive backend for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#

def plot_confusion_matrix(y_true_path='output/supervised_true.npy', y_pred_path='output/supervised_pred.npy', save_path='output/figure_1_confusion_matrix.png'):
    """Plots the confusion matrix for the supervised model."""
    print("Attempting to generate Figure 1: Supervised Model Confusion Matrix...")
    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        print("--> Error: Supervised results not found. Please run option 1 from the main menu first.\n")
        return
    y_true, y_pred = np.load(y_true_path), np.load(y_pred_path)
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Normal', 'Disturbed']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16}, ax=ax)
    ax.set_title('Fig 1: Supervised Model Classification Performance', fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"--> Figure 1 saved to {save_path}\n")
    plt.close(fig)

def plot_unsupervised_clusters(data_path='output/unlabeled_dataset_full.csv', labels_path='output/cluster_labels.npy', save_path='output/figure_2_unsupervised_clusters.png'):
    """Plots the clusters discovered by the unsupervised model using PCA."""
    print("Attempting to generate Figure 2: Unsupervised Model Clusters...")
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("--> Error: Unsupervised results not found. Please run option 2 from the main menu first.\n")
        return
    data, labels = pd.read_csv(data_path), np.load(labels_path)
    if len(data) != len(labels):
        print(f"--> Error: Data size ({len(data)}) and label size ({len(labels)}) do not match.")
        print("--> Please re-run option 2 from the main menu to regenerate the unsupervised results.\n")
        return
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(StandardScaler().fit_transform(data))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    ax.set_title('Fig 2: Unsupervised Discovery of Data Clusters', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(len(np.unique(labels)))], title="Discovered Clusters")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"--> Figure 2 saved to {save_path}\n")
    plt.close(fig)

def plot_reinforcement_learning_results(q_table_path='output/q_table.npy', lines_path='output/line_names.txt', save_path='output/figure_3_reinforcement_policy.png'):
    """Plots the most critical lines identified by the reinforcement learning agent."""
    print("Attempting to generate Figure 3: Reinforcement Learning Policy...")
    if not os.path.exists(q_table_path) or not os.path.exists(lines_path):
        print("--> Error: Reinforcement learning results not found. Please run option 3 from the main menu first.\n")
        return
    q_table = np.load(q_table_path, allow_pickle=True).item()
    with open(lines_path, 'r') as f:
        line_names = [line.strip() for line in f.readlines()]
    overload_counts = defaultdict(int)
    for state in q_table.keys():
        if sum(state) > 0:
            for line_index, status in enumerate(state):
                if status == 1: overload_counts[line_index] += 1
    if not overload_counts:
        print("--> No overload states found in Q-table. Cannot generate plot.")
        return
    overload_series = pd.Series(overload_counts).sort_values(ascending=False)
    overload_series.index = [line_names[i] for i in overload_series.index]
    top_n = 15
    fig, ax = plt.subplots(figsize=(12, 8))
    overload_series.head(top_n).sort_values(ascending=True).plot(kind='barh', color='darkcyan', ax=ax)
    ax.set_title(f'Fig 3: Reinforcement Agent - Top {top_n} Critical Lines Identified', fontsize=16)
    ax.set_xlabel('Number of Unique Overload States Encountered', fontsize=12)
    ax.set_ylabel('Power Line', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"--> Figure 3 saved to {save_path}\n")
    plt.close(fig)
