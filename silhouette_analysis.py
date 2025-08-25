import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_silhouette_analysis(df, max_k=10, save_plot=True):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    scores = []
    k_values = list(range(2, max_k + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        scores.append(score)
        print(f"k = {k}, Silhouette Score = {score:.4f}")

    best_k = k_values[np.argmax(scores)]
    print(f"\nBest number of clusters based on silhouette score: {best_k}")

    if save_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, scores, marker='o')
        ax.set_title("Silhouette Score vs. Number of Clusters (k)")
        ax.set_xlabel("k (Clusters)")
        ax.set_ylabel("Silhouette Score")
        ax.grid(True)
        fig.tight_layout()
        plt.savefig("output/silhouette_scores.png")
        plt.close(fig)

    return best_k
