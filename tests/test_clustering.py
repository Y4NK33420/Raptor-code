import numpy as np
import pytest

from raptor.clustering import CLUSTERERS, RaptorGMMClustering, check_clusters

pytestmark = pytest.mark.slow


def blobs(n_per=15, centers=4, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(centers, dim)) * 5
    x = np.concatenate([m + rng.normal(size=(n_per, dim)) for m in means]).astype(np.float32)
    return x


def test_covers_every_point_and_is_deterministic():
    x = blobs()
    tokens = [50] * len(x)
    a = RaptorGMMClustering(random_state=7).cluster(x, tokens)
    b = RaptorGMMClustering(random_state=7).cluster(x, tokens)
    assert {i for c in a for i in c} == set(range(len(x)))
    assert a == b  # seeded UMAP + GMM -> identical clusters
    assert 2 <= len(a) < len(x)


def test_duplicate_embeddings_map_to_their_own_indices():
    x = blobs(n_per=8)
    x = np.concatenate([x, x[:5]])  # exact duplicates of the first five rows
    clusters = RaptorGMMClustering(random_state=1).cluster(x, [10] * len(x))
    assert {i for c in clusters for i in c} == set(range(len(x)))
    assert all(len(set(c)) == len(c) for c in clusters)


def test_token_limit_is_enforced():
    x = blobs(n_per=10)
    tokens = [400] * len(x)
    clusters = RaptorGMMClustering(max_cluster_tokens=1000, random_state=3).cluster(x, tokens)
    for c in clusters:
        assert len(c) == 1 or sum(tokens[i] for i in c) <= 1000


def test_tiny_inputs_terminate():
    algo = RaptorGMMClustering(max_cluster_tokens=10)
    assert algo.cluster(np.zeros((0, 4), np.float32), []) == []
    assert algo.cluster(np.ones((1, 4), np.float32), [100]) == [[0]]
    out = algo.cluster(np.eye(3, dtype=np.float32), [100, 100, 100])
    assert sorted(i for c in out for i in c) == [0, 1, 2]


def test_check_clusters_rejects_gaps():
    with pytest.raises(ValueError):
        check_clusters([[0, 1]], 3)
    assert check_clusters([[1, 0, 1], [], [2]], 3) == [[0, 1], [2]]


def test_registry():
    assert isinstance(CLUSTERERS.create("raptor_gmm"), RaptorGMMClustering)
