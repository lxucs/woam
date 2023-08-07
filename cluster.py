import hdbscan
import torch
from collections import defaultdict
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def get_precompute_dist(span_hidden, normalize=True):
    """ Get pairwise cosine distance to be consumed by clustering algorithm.
    cosine distance: 1 - cosine similarity
    """
    device = span_hidden.device
    with torch.no_grad():
        span_hidden = span_hidden.to(device)
        # Normalize if hasn't
        if normalize:
            span_hidden = F.normalize(span_hidden, p=2, dim=-1, eps=1e-8)
        # Get pairwise cosine distance
        pairwise_dist = (1 - torch.matmul(span_hidden, span_hidden.t()))  # [num_spans, num_spans]
        pairwise_dist[pairwise_dist <= 0] *= 0  # Numerical issue
    return pairwise_dist


def hdbscan_clustering(hidden, metric, eps=0, min_cluster_size=10, min_samples=5):
    # labels: -1 means no cluster
    clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=min_cluster_size, min_samples=min_samples,
                                cluster_selection_epsilon=eps)
    clusterer.fit(hidden)
    return clusterer.labels_, clusterer.probabilities_  # Numpy


def dbscan_clustering(hidden, metric, eps=0.5, min_samples=5, n_jobs=8):
    # labels: -1 means no cluster
    clusterer = DBSCAN(metric=metric, eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    labels = clusterer.fit_predict(hidden)
    return labels  # Numpy
