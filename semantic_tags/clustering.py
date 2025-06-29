from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def choose_k(embeddings: np.ndarray, k_min: int = 2, k_max: int = None) -> int:
    n_samples = embeddings.shape[0]
    if k_max is None:
        k_max = int(np.sqrt(n_samples)) + 1
    best_k = k_min
    best_score = -1
    for k in range(k_min, min(k_max, n_samples) + 1):
        km = KMeans(n_clusters=k, n_init="auto")
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_embeddings(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels, km
