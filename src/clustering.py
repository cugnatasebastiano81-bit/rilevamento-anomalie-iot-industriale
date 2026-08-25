"""Fase 3 — clustering: funzioni pure, testate in isolamento in tests/test_clustering.py.

Il notebook (notebooks/) importa queste funzioni invece di ridefinirle inline; il ragionamento
sul perché di ogni scelta resta nel notebook, qui c'è solo il codice.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_k_selection_curves(X, k_range, sample_size, random_state=42, n_init=10):
    """Inerzia e silhouette per ogni K in k_range, per la diagnostica elbow/silhouette.

    L'inerzia è O(n) e viene calcolata sul K-Means fit su tutto X. La silhouette è O(n^2):
    viene calcolata con `silhouette_score(..., sample_size=sample_size)`, che valuta la metrica
    su un campione casuale fisso (stesso `random_state`) invece che su tutto X, mantenendo
    comunque il fit del K-Means su tutto X per ogni K. Il campionamento riguarda solo questa
    diagnostica, mai il fit finale del modello (vedi `fit_final_kmeans`).

    Silhouette non è definito per K=1 (richiede almeno 2 cluster): il valore restituito in
    quel caso è `None`.
    """
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k < 2:
            silhouettes.append(None)
        else:
            effective_sample_size = min(sample_size, len(X))
            silhouettes.append(
                silhouette_score(X, labels, sample_size=effective_sample_size, random_state=random_state)
            )
    return {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}


def fit_final_kmeans(X, n_clusters, random_state=42, n_init=10):
    """K-Means finale, addestrato su tutto X (nessun campionamento)."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    km.fit(X)
    return km
