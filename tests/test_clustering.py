"""Unit test per src/clustering.py (Fase 3). Dati sintetici piccoli, nessun I/O reale."""
import numpy as np
import pytest

from src.clustering import compute_k_selection_curves, fit_final_kmeans


def _well_separated_blobs(n_per_blob=50, n_blobs=3, spread=0.5, offset=20.0, seed=0):
    rng = np.random.RandomState(seed)
    blocks = []
    for i in range(n_blobs):
        center = np.full(2, i * offset)
        blocks.append(rng.normal(loc=center, scale=spread, size=(n_per_blob, 2)))
    return np.vstack(blocks)


# --- fit_final_kmeans ---

def test_fit_final_kmeans_deterministic_given_seed():
    X = _well_separated_blobs()
    km1 = fit_final_kmeans(X, n_clusters=3, random_state=42)
    km2 = fit_final_kmeans(X, n_clusters=3, random_state=42)
    assert km1.inertia_ == pytest.approx(km2.inertia_)
    assert km1.labels_.tolist() == km2.labels_.tolist()


def test_fit_final_kmeans_separates_well_separated_blobs():
    X = _well_separated_blobs(n_per_blob=50, n_blobs=3, spread=0.5, offset=20.0)
    km = fit_final_kmeans(X, n_clusters=3, random_state=42)
    labels = km.labels_
    # ogni blocco di 50 punti generato dallo stesso centro deve finire in un unico cluster
    for i in range(3):
        block_labels = labels[i * 50:(i + 1) * 50]
        assert len(set(block_labels.tolist())) == 1
    # e i tre blocchi devono finire in cluster diversi tra loro
    assert len({labels[0], labels[50], labels[100]}) == 3


def test_fit_final_kmeans_no_subsampling_uses_all_rows():
    X = _well_separated_blobs()
    km = fit_final_kmeans(X, n_clusters=3, random_state=42)
    assert len(km.labels_) == len(X)


# --- compute_k_selection_curves ---

def test_compute_k_selection_curves_silhouette_undefined_at_k1():
    X = _well_separated_blobs()
    result = compute_k_selection_curves(X, k_range=[1, 2, 3], sample_size=30, random_state=42)
    assert result["silhouette"][0] is None
    assert result["silhouette"][1] is not None
    assert result["silhouette"][2] is not None


def test_compute_k_selection_curves_inertia_decreases_with_k():
    X = _well_separated_blobs()
    result = compute_k_selection_curves(X, k_range=[1, 2, 3, 4], sample_size=30, random_state=42)
    inertias = result["inertia"]
    assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))


def test_compute_k_selection_curves_silhouette_high_for_well_separated_blobs():
    X = _well_separated_blobs(n_per_blob=50, n_blobs=3, spread=0.5, offset=20.0)
    result = compute_k_selection_curves(X, k_range=[2, 3, 4], sample_size=100, random_state=42)
    k_to_silhouette = dict(zip(result["k"], result["silhouette"]))
    # con blob ben separati e K corretto (3), la silhouette deve essere alta (vicina a 1)
    assert k_to_silhouette[3] > 0.9


def test_compute_k_selection_curves_deterministic_given_seed():
    X = _well_separated_blobs()
    result1 = compute_k_selection_curves(X, k_range=[2, 3], sample_size=30, random_state=42)
    result2 = compute_k_selection_curves(X, k_range=[2, 3], sample_size=30, random_state=42)
    assert result1["inertia"] == pytest.approx(result2["inertia"])
    assert result1["silhouette"] == pytest.approx(result2["silhouette"])


def test_compute_k_selection_curves_sample_size_larger_than_data_is_clamped():
    X = _well_separated_blobs(n_per_blob=10, n_blobs=2, spread=0.5, offset=20.0)  # 20 righe totali
    # sample_size (1000) supera le righe disponibili (20): non deve sollevare un errore
    result = compute_k_selection_curves(X, k_range=[2], sample_size=1000, random_state=42)
    assert result["silhouette"][0] is not None
