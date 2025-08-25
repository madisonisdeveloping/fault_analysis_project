import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import time

def load_unlabeled_dataset(path='output/unlabeled_dataset_full.csv'):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df.select_dtypes(include=[np.number])

def apply_pca(df, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(scaled_data)

def cluster_and_plot(df, n_clusters=3):
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Identify anomalous cluster more robustly
    cluster_centers = kmeans.cluster_centers_
    distances = np.sum(cluster_centers**2, axis=1)
    anomalous_cluster_label = np.argmax(distances)
    print(f"Identified anomalous cluster: {anomalous_cluster_label} (center furthest from origin)")

    print("Saving cluster labels and metrics for analysis...")
    np.save('output/cluster_labels.npy', labels)
    np.save('output/anomalous_cluster_label.npy', anomalous_cluster_label)

    score = silhouette_score(X_scaled, labels)
    print(f"Final Silhouette Score: {score:.4f}")

    metrics = {"silhouette_score": score}
    np.save('output/unsupervised_metrics.npy', metrics)

    # --- VISUAL ENHANCEMENT: Define descriptive labels and high-contrast colors ---
    normal_cluster_label = next(l for l in np.unique(labels) if l != anomalous_cluster_label)

    label_map = {
        anomalous_cluster_label: "Anomalous Event",
        normal_cluster_label: "Normal Operation"
    }
    color_map = {
        anomalous_cluster_label: '#d62728',  # Strong red
        normal_cluster_label: '#1f77b4'   # Muted blue
    }
    plot_colors = [color_map[l] for l in labels]

    # Apply PCA for visualization
    X_pca = apply_pca(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot with new colors
    ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=plot_colors,
        s=50,
        alpha=0.7
    )

    ax.set_title("K-Means Clustering with PCA (2D)", fontsize=16)
    ax.set_xlabel("PCA Component 1 (Major Power Flow Pattern)", fontsize=12)
    ax.set_ylabel("PCA Component 2 (Secondary Power Flow Pattern)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- VISUAL ENHANCEMENT: Create a custom, descriptive legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label_map[normal_cluster_label],
               markerfacecolor=color_map[normal_cluster_label], markersize=10),
        Line2D([0], [0], marker='o', color='w', label=label_map[anomalous_cluster_label],
               markerfacecolor=color_map[anomalous_cluster_label], markersize=10)
    ]
    ax.legend(handles=legend_elements, title="Cluster Groups")

    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/unsupervised_clusters.png")
    plt.close(fig)
    print("Cluster visualization saved to output/unsupervised_clusters.png")


def auto_cluster(df, max_k=6):
    from silhouette_analysis import run_silhouette_analysis
    start_time = time.time()
    best_k = run_silhouette_analysis(df, max_k=max_k)
    cluster_and_plot(df, n_clusters=best_k)
    end_time = time.time()
    total_time = end_time - start_time
    if os.path.exists('output/unsupervised_metrics.npy'):
        metrics = np.load('output/unsupervised_metrics.npy', allow_pickle=True).item()
        metrics["training_time_sec"] = total_time
        np.save('output/unsupervised_metrics.npy', metrics)
        print(f"Unsupervised analysis complete. Total time: {total_time:.2f}s")
