"""
Visualization module for generating publication-quality plots and charts.
"""

import math
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Set default styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150


def plot_elbow_curve(
    inertias: List[float],
    optimal_k: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots the Elbow Method inertia curve to identify the inflection point.

    Args:
        inertias (List[float]): Inertia values from k=1 to max_k.
        optimal_k (Optional[int]): Selected optimal k to highlight.
        save_path (Optional[str]): Filepath to save the plot figure.
    """
    k_range = range(1, len(inertias) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, marker="o", color="#2563eb", linewidth=2.2, markersize=7)
    
    if optimal_k:
        ax.axvline(
            x=optimal_k,
            color="#dc2626",
            linestyle="--",
            label=f"Optimal k = {optimal_k}",
        )
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

    ax.set_title("Elbow Method for Optimal k Selection", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax.set_xticks(list(k_range))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pca_clusters(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA 2D Cluster Projection",
    archetype_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a 2D scatter visualization of the PCA-reduced TF-IDF feature space.

    Args:
        coords_2d (np.ndarray): (N, 2) PCA coordinates.
        labels (np.ndarray): Cluster assignment indices.
        title (str): Plot title.
        archetype_names (Optional[Dict[int, str]]): Human-readable cluster names.
        save_path (Optional[str]): Filepath to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("tab10", len(unique_labels))

    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        label_text = (
            archetype_names.get(cluster_id, f"Cluster {cluster_id}")
            if archetype_names
            else f"Cluster {cluster_id}"
        )
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[idx]],
            label=label_text,
            alpha=0.65,
            edgecolors="w",
            s=45,
        )

    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_top_keywords_per_cluster(
    model: KMeans,
    feature_names: List[str],
    n_top_words: int = 10,
    archetype_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots horizontal bar charts of the highest TF-IDF weighted terms for each cluster.

    Args:
        model (KMeans): Fitted KMeans clustering model.
        feature_names (List[str]): Vectorizer vocabulary terms.
        n_top_words (int): Number of keywords per cluster.
        archetype_names (Optional[Dict[int, str]]): Cluster names.
        save_path (Optional[str]): Filepath to save the plot.
    """
    n_clusters = len(model.cluster_centers_)
    cols = 3
    rows = math.ceil(n_clusters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for i in range(n_clusters):
        top_indices = order_centroids[i, :n_top_words]
        top_words = [feature_names[ind] for ind in top_indices]
        top_weights = [model.cluster_centers_[i, ind] for ind in top_indices]

        title = (
            archetype_names.get(i, f"Cluster {i}")
            if archetype_names
            else f"Cluster {i}"
        )
        ax = axes[i]
        sns.barplot(x=top_weights, y=top_words, ax=ax, palette="Blues_r")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("TF-IDF Centroid Weight", fontsize=9)

    # Hide unused axes
    for j in range(n_clusters, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Dominant Skill N-Grams by Cluster Archetype", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
