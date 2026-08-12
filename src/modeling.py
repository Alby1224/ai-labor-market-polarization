"""
Unsupervised Machine Learning and Dimensionality Reduction Pipeline.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    KMEANS_N_INIT,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from src.preprocessing import get_combined_stop_words


def extract_tfidf_features(
    corpus: pd.Series,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Transforms text corpus into a weighted TF-IDF term matrix.

    Args:
        corpus (pd.Series): Cleaned job description strings.
        max_features (int): Top n-gram vocabulary capacity.
        ngram_range (tuple): Range of n-grams (unigrams, bigrams).
        min_df (int): Minimum document frequency threshold.

    Returns:
        Tuple[csr_matrix, TfidfVectorizer]: Sparse feature matrix and fitted vectorizer.
    """
    stop_words = get_combined_stop_words()
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def compute_elbow_curve(
    X: csr_matrix, max_k: int = 10, random_state: int = RANDOM_STATE
) -> List[float]:
    """
    Computes within-cluster sum of squares (inertia) across a range of k values.

    Args:
        X (csr_matrix): TF-IDF feature matrix.
        max_k (int): Upper bound for number of clusters to test.
        random_state (int): Seed for deterministic initialization.

    Returns:
        List[float]: Inertia values for k in [1, max_k].
    """
    inertias: List[float] = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=KMEANS_N_INIT)
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias


def fit_kmeans_model(
    X: csr_matrix, n_clusters: int, random_state: int = RANDOM_STATE
) -> Tuple[KMeans, np.ndarray]:
    """
    Fits K-Means algorithm to partition latent skill representations into archetypes.

    Args:
        X (csr_matrix): TF-IDF matrix.
        n_clusters (int): Optimal number of archetypes (k).
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[KMeans, np.ndarray]: Fitted KMeans model and array of cluster assignments.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=KMEANS_N_INIT)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def project_pca_2d(
    X: csr_matrix, n_components: int = 2, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """
    Projects high-dimensional TF-IDF vectors into a 2D Euclidean space using PCA.

    Args:
        X (csr_matrix): TF-IDF matrix.
        n_components (int): Target dimensions (default 2).
        random_state (int): Deterministic seed.

    Returns:
        np.ndarray: (N, 2) Coordinates for 2D scatter visualization.
    """
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    pca = PCA(n_components=n_components, random_state=random_state)
    coords_2d = pca.fit_transform(X_dense)
    return coords_2d
