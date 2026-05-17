import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


def neg_davies_bouldin_scorer(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_
    num_labels = len(np.unique(labels))
    if num_labels < 2 or num_labels >= len(X):
        return -1.0
    return -1 * davies_bouldin_score(X, labels)


def calinski_harabasz_scorer(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_
    num_labels = len(np.unique(labels))
    if num_labels < 2 or num_labels >= len(X):
        return -1.0
    return calinski_harabasz_score(X, labels)


def dbcv_scorer(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_
    X = np.asarray(X)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return -1.0
    dist_matrix = cdist(X, X)
    k = min(4, len(X) - 1)
    core_dists = np.sort(dist_matrix, axis=1)[:, k]
    mrd = np.maximum(dist_matrix, np.maximum(core_dists[:, None], core_dists[None, :]))
    cluster_validities = []
    cluster_sizes = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_sizes.append(np.sum(cluster_mask))
        internal_mrd = mrd[cluster_mask][:, cluster_mask]
        if internal_mrd.size == 0:
            continue
        mst = minimum_spanning_tree(internal_mrd).toarray()
        dsc = np.max(mst)
        dspc = np.inf
        for other_label in unique_labels:
            if label == other_label:
                continue
            other_mask = labels == other_label
            sep_mrd = mrd[cluster_mask][:, other_mask]
            if sep_mrd.size > 0:
                dspc = min(dspc, np.min(sep_mrd))
        if dspc == np.inf:
            dspc = 0.0
        max_separation_sparseness = max(dspc, dsc)
        if max_separation_sparseness > 0:
            v_index = (dspc - dsc) / max_separation_sparseness
        else:
            v_index = 0.0
        cluster_validities.append(v_index)
    total_points = sum(cluster_sizes)
    if total_points > 0:
        final_score = (
            sum(v * s for v, s in zip(cluster_validities, cluster_sizes)) / total_points
        )
    else:
        final_score = -1.0
    return final_score


def silhouette_scorer(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_
    num_labels = len(np.unique(labels))
    if num_labels < 2 or num_labels >= len(X):
        return -1.0
    return silhouette_score(X, labels)
