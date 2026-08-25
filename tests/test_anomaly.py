"""Unit test per src/anomaly.py (Fase 4). Dati sintetici piccoli, nessun I/O reale."""
import numpy as np
import pytest

from src.anomaly import (
    classify_anomalies,
    compute_centroid_distances,
    compute_cluster_thresholds,
    compute_percentile_threshold,
    resolve_row_thresholds,
)

# --- compute_centroid_distances ---


def test_compute_centroid_distances_example_3_4_5():
    X = np.array([[3.0, 4.0]])
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([0])
    distances = compute_centroid_distances(X, labels, centroids)
    assert distances[0] == pytest.approx(5.0, abs=1e-12)


def test_compute_centroid_distances_zero_on_centroid():
    X = np.array([[1.5, -2.0, 3.0]])
    centroids = np.array([[1.5, -2.0, 3.0]])
    labels = np.array([0])
    distances = compute_centroid_distances(X, labels, centroids)
    assert distances[0] == pytest.approx(0.0, abs=1e-12)


def test_compute_centroid_distances_increases_moving_away_from_centroid():
    centroid = np.array([[0.0, 0.0]])
    near = compute_centroid_distances(np.array([[1.0, 0.0]]), np.array([0]), centroid)
    far = compute_centroid_distances(np.array([[5.0, 0.0]]), np.array([0]), centroid)
    farther = compute_centroid_distances(np.array([[10.0, 0.0]]), np.array([0]), centroid)
    assert near[0] < far[0] < farther[0]


def test_compute_centroid_distances_uses_respective_centroid_per_cluster():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels = np.array([0, 1])
    distances = compute_centroid_distances(X, labels, centroids)
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert distances[1] == pytest.approx(0.0, abs=1e-12)


def test_compute_centroid_distances_output_length_matches_input():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([0, 0, 0])
    distances = compute_centroid_distances(X, labels, centroids)
    assert len(distances) == len(X) == len(labels)


def test_compute_centroid_distances_rejects_nan():
    X = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, np.array([0]), np.array([[0.0, 0.0]]))


def test_compute_centroid_distances_rejects_infinite():
    X = np.array([[1.0, np.inf]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, np.array([0]), np.array([[0.0, 0.0]]))


def test_compute_centroid_distances_rejects_non_2d_X():
    X = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, np.array([0, 0, 0]), np.array([[0.0], [1.0]]))


def test_compute_centroid_distances_rejects_incompatible_dimensions():
    X = np.array([[1.0, 2.0, 3.0]])
    centroids = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, np.array([0]), centroids)


def test_compute_centroid_distances_rejects_mismatched_labels_length():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_non_integer_labels():
    X = np.array([[0.0, 0.0]])
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([0.0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_negative_cluster():
    X = np.array([[0.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([-1])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_out_of_range_cluster():
    X = np.array([[0.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([2])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_non_2d_centroids():
    X = np.array([[0.0, 0.0]])
    centroids = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, np.array([0]), centroids)


def test_compute_centroid_distances_rejects_empty_X():
    X = np.empty((0, 2))
    centroids = np.array([[0.0, 0.0]])
    labels = np.array([], dtype=int)
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


# --- compute_percentile_threshold ---


def test_compute_percentile_threshold_matches_numpy_linear():
    distances = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0])
    threshold = compute_percentile_threshold(distances, 75)
    assert threshold == pytest.approx(np.percentile(distances, 75, method="linear"))


def test_compute_percentile_threshold_rejects_out_of_range_percentile():
    distances = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        compute_percentile_threshold(distances, 101)
    with pytest.raises(ValueError):
        compute_percentile_threshold(distances, -1)


def test_compute_percentile_threshold_rejects_non_finite_distances():
    with pytest.raises(ValueError):
        compute_percentile_threshold(np.array([1.0, np.nan, 3.0]), 95)
    with pytest.raises(ValueError):
        compute_percentile_threshold(np.array([1.0, np.inf, 3.0]), 95)


def test_compute_percentile_threshold_rejects_negative_distances():
    with pytest.raises(ValueError):
        compute_percentile_threshold(np.array([1.0, -2.0, 3.0]), 95)


def test_compute_percentile_threshold_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_percentile_threshold(np.array([]), 95)


def test_compute_percentile_threshold_rejects_non_1d_input():
    with pytest.raises(ValueError):
        compute_percentile_threshold(np.array([[1.0, 2.0], [3.0, 4.0]]), 95)


# --- classify_anomalies: valori esattamente sulla soglia ---


def test_classify_anomalies_ties_on_threshold_stay_normal():
    distances = np.array([5.0, 5.0, 6.0, 4.9])
    threshold = 5.0
    flags = classify_anomalies(distances, threshold)
    assert flags.tolist() == [False, False, True, False]
    assert flags.mean() == pytest.approx(0.25)


# --- compute_cluster_thresholds: fallback sotto la numerosità minima ---


def test_compute_cluster_thresholds_small_cluster_falls_back_to_global():
    rng = np.random.RandomState(0)
    large_cluster_distances = np.abs(rng.normal(loc=3.0, scale=1.0, size=150))
    small_cluster_distances = np.abs(rng.normal(loc=10.0, scale=1.0, size=10))
    distances = np.concatenate([large_cluster_distances, small_cluster_distances])
    labels = np.concatenate([np.zeros(150, dtype=int), np.ones(10, dtype=int)])

    result = compute_cluster_thresholds(distances, labels, n_clusters=2, percentile=99, min_cluster_size=100)

    assert result["fallback_used"][0] is False
    assert result["fallback_used"][1] is True
    assert result["per_cluster_threshold"][1] == pytest.approx(result["global_threshold"])
    expected_cluster_0 = compute_percentile_threshold(large_cluster_distances, 99)
    assert result["per_cluster_threshold"][0] == pytest.approx(expected_cluster_0)


def test_compute_cluster_thresholds_large_cluster_uses_own_quantile():
    rng = np.random.RandomState(1)
    distances = np.abs(rng.normal(loc=3.0, scale=1.0, size=500))
    labels = np.zeros(500, dtype=int)

    result = compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95, min_cluster_size=100)

    assert result["fallback_used"][0] is False
    assert result["per_cluster_threshold"][0] != pytest.approx(result["global_threshold"] + 1)
    assert result["per_cluster_threshold"][0] == pytest.approx(compute_percentile_threshold(distances, 95))


def test_compute_cluster_thresholds_rejects_mismatched_lengths():
    distances = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95)


# --- sentinella: calibrazione dipende soltanto dal train ---


def test_cluster_thresholds_depend_only_on_train_not_on_separate_test_data():
    rng = np.random.RandomState(2)
    train_distances = np.abs(rng.normal(loc=3.0, scale=1.0, size=500))
    train_labels = rng.randint(0, 2, size=500)

    thresholds_before = compute_cluster_thresholds(train_distances, train_labels, n_clusters=2, percentile=99)

    # dati di "test" estremi definiti a parte: non entrano mai nella calibrazione, la firma
    # della funzione accetta soltanto le distanze/label del train.
    _test_distances_never_used = np.full(500, 1000.0)
    _test_labels_never_used = np.zeros(500, dtype=int)

    thresholds_after = compute_cluster_thresholds(train_distances, train_labels, n_clusters=2, percentile=99)

    assert thresholds_after == thresholds_before


# --- validazione degli input di compute_centroid_distances, classify_anomalies,
# compute_cluster_thresholds e resolve_row_thresholds: NaN, infiniti, overflow, forme e
# conteggi non validi devono sollevare ValueError, non propagarsi in silenzio ---


def test_compute_centroid_distances_rejects_nan_centroids():
    X = np.array([[1.0, 2.0]])
    labels = np.array([0])
    centroids = np.array([[np.nan, 0.0]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_infinite_centroids():
    X = np.array([[1.0, 2.0]])
    labels = np.array([0])
    centroids = np.array([[np.inf, 0.0]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_overflow_result():
    X = np.array([[1e200, 0.0]])
    labels = np.array([0])
    centroids = np.array([[-1e200, 0.0]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_non_1d_labels():
    X = np.array([[1.0, 2.0]])
    labels = np.array([[0]])
    centroids = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_classify_anomalies_rejects_non_1d_distances():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0]))


def test_classify_anomalies_rejects_nan_distances():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([np.nan]), np.array([1.0]))


def test_classify_anomalies_rejects_infinite_distances():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([np.inf]), np.array([1.0]))


def test_classify_anomalies_rejects_negative_distances():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([-1.0]), np.array([1.0]))


def test_classify_anomalies_rejects_incompatible_threshold_shape():
    distances = np.array([1.0, 2.0, 3.0, 4.0])
    thresholds = np.array([[1.0], [2.0], [3.0], [4.0]])
    with pytest.raises(ValueError):
        classify_anomalies(distances, thresholds)


def test_compute_cluster_thresholds_rejects_zero_n_clusters():
    distances = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=0, percentile=95)


def test_compute_cluster_thresholds_rejects_negative_n_clusters():
    distances = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=-1, percentile=95)


def test_compute_cluster_thresholds_rejects_non_positive_min_cluster_size():
    distances = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95, min_cluster_size=0)
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95, min_cluster_size=-5)


def test_resolve_row_thresholds_rejects_missing_cluster():
    labels = np.array([0, 1])
    per_cluster_threshold = {0: 5.0}
    with pytest.raises(ValueError):
        resolve_row_thresholds(labels, per_cluster_threshold)


def test_classify_anomalies_row_order_independent():
    distances = np.array([1.0, 10.0, 3.0, 20.0])
    labels = np.array([0, 1, 0, 1])
    per_cluster_threshold = {0: 2.0, 1: 15.0}

    row_thresholds = resolve_row_thresholds(labels, per_cluster_threshold)
    flags = classify_anomalies(distances, row_thresholds)

    perm = np.array([3, 1, 0, 2])
    row_thresholds_perm = resolve_row_thresholds(labels[perm], per_cluster_threshold)
    flags_perm = classify_anomalies(distances[perm], row_thresholds_perm)

    assert flags_perm.tolist() == flags[perm].tolist()


# --- validazione delle soglie e delle etichette: mappa di soglie per cluster, array di
# distanza/etichetta e feature devono rifiutare valori NaN, infiniti, negativi, di forma o
# conteggio non validi invece di produrre un risultato silenzioso ---


def test_classify_anomalies_rejects_nan_threshold():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError):
        classify_anomalies(np.array([1.0]), np.nan)


def test_classify_anomalies_rejects_infinite_threshold():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([1.0]), np.array([np.inf]))
    with pytest.raises(ValueError):
        classify_anomalies(np.array([1.0]), np.inf)


def test_classify_anomalies_rejects_negative_threshold():
    with pytest.raises(ValueError):
        classify_anomalies(np.array([1.0]), np.array([-1.0]))
    with pytest.raises(ValueError):
        classify_anomalies(np.array([1.0]), -1.0)


def test_compute_cluster_thresholds_rejects_2d_labels():
    distances = np.array([1.0, 2.0])
    labels = np.array([[0], [0]])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95)


def test_compute_cluster_thresholds_rejects_non_integer_labels():
    distances = np.array([1.0, 2.0])
    labels = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95)


def test_compute_cluster_thresholds_rejects_out_of_range_labels():
    distances = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 2])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95)


def test_compute_cluster_thresholds_rejects_negative_labels():
    distances = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, -1])
    with pytest.raises(ValueError):
        compute_cluster_thresholds(distances, labels, n_clusters=1, percentile=95)


def test_compute_cluster_thresholds_rejects_scalar_distances():
    with pytest.raises(ValueError):
        compute_cluster_thresholds(np.array(5.0), np.array([0]), n_clusters=1, percentile=95)


def test_compute_cluster_thresholds_rejects_scalar_labels():
    with pytest.raises(ValueError):
        compute_cluster_thresholds(np.array([5.0]), np.array(0), n_clusters=1, percentile=95)


def test_compute_cluster_thresholds_rejects_empty_distances():
    with pytest.raises(ValueError):
        compute_cluster_thresholds(np.array([]), np.array([], dtype=int), n_clusters=1, percentile=95)


def test_resolve_row_thresholds_rejects_non_1d_labels():
    with pytest.raises(ValueError):
        resolve_row_thresholds(np.array([[0]]), {0: 1.0})


def test_resolve_row_thresholds_rejects_non_integer_labels():
    with pytest.raises(ValueError):
        resolve_row_thresholds(np.array([0.0]), {0: 1.0})


def test_resolve_row_thresholds_rejects_nan_threshold_in_map():
    with pytest.raises(ValueError):
        resolve_row_thresholds(np.array([0]), {0: np.nan})


def test_resolve_row_thresholds_rejects_infinite_threshold_in_map():
    with pytest.raises(ValueError):
        resolve_row_thresholds(np.array([0]), {0: np.inf})


def test_resolve_row_thresholds_rejects_negative_threshold_in_map():
    with pytest.raises(ValueError):
        resolve_row_thresholds(np.array([0]), {0: -1.0})


def test_compute_centroid_distances_rejects_zero_feature_arrays():
    X = np.empty((1, 0))
    centroids = np.empty((1, 0))
    labels = np.array([0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_zero_feature_centroids_only():
    X = np.array([[1.0]])
    centroids = np.empty((1, 0))
    labels = np.array([0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)


def test_compute_centroid_distances_rejects_centroids_without_rows():
    X = np.array([[1.0]])
    centroids = np.empty((0, 1))
    labels = np.array([0])
    with pytest.raises(ValueError):
        compute_centroid_distances(X, labels, centroids)
