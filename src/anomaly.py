"""Fase 4 — soglia di anomalia: funzioni pure, testate in isolamento in tests/test_anomaly.py.

Il notebook (notebooks/) importa queste funzioni invece di ridefinirle inline; il ragionamento
sul perché di ogni scelta resta nel notebook, qui c'è solo il codice.
"""
import numpy as np


def compute_centroid_distances(X, labels, centroids):
    """Distanza euclidea di ogni riga di X dal centroide del proprio cluster (`centroids[labels[i]]`).

    Convalida esplicitamente il contratto invece di propagare in silenzio valori non finiti o
    cluster non validi: `X` deve essere una matrice 2D finita, `labels` interi nell'intervallo
    valido dato `centroids`, con lunghezza coerente tra `X` e `labels`.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X deve avere 2 dimensioni, trovate {X.ndim}")
    if centroids.ndim != 2:
        raise ValueError(f"centroids deve avere 2 dimensioni, trovate {centroids.ndim}")
    if labels.ndim != 1:
        raise ValueError(f"labels deve avere 1 dimensione, trovate {labels.ndim}")
    if X.shape[0] == 0:
        raise ValueError("X non può essere vuoto")
    if X.shape[1] == 0:
        raise ValueError("X non può avere zero feature")
    if centroids.shape[0] == 0:
        raise ValueError("centroids non può essere vuoto (zero righe)")
    if centroids.shape[1] == 0:
        raise ValueError("centroids non può avere zero feature")
    if labels.shape[0] != X.shape[0]:
        raise ValueError(f"labels ha {labels.shape[0]} elementi, X ha {X.shape[0]} righe: devono coincidere")
    if X.shape[1] != centroids.shape[1]:
        raise ValueError(f"X ha {X.shape[1]} colonne, centroids ne ha {centroids.shape[1]}: dimensioni incompatibili")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene valori non finiti (NaN o infinito)")
    if not np.all(np.isfinite(centroids)):
        raise ValueError("centroids contiene valori non finiti (NaN o infinito)")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels deve contenere interi, trovato dtype {labels.dtype}")
    if labels.shape[0] and (np.any(labels < 0) or np.any(labels >= centroids.shape[0])):
        raise ValueError(f"labels contiene valori fuori dall'intervallo valido [0, {centroids.shape[0] - 1}]")

    assigned_centroids = centroids[labels]
    result = np.sqrt(np.sum((X - assigned_centroids) ** 2, axis=1))
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "la distanza calcolata non è finita: overflow numerico con input estremi nonostante "
            "X e centroids finiti"
        )
    return result


def compute_percentile_threshold(distances, percentile):
    """Soglia come percentile empirico delle distanze, metodo `linear` (coerente con `numpy.percentile`)."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1:
        raise ValueError(f"distances deve avere 1 dimensione, trovate {distances.ndim}")
    if distances.shape[0] == 0:
        raise ValueError("distances non può essere vuoto")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances contiene valori non finiti (NaN o infinito)")
    if np.any(distances < 0):
        raise ValueError("distances contiene valori negativi: una distanza euclidea non può esserlo")
    if not (0 <= percentile <= 100):
        raise ValueError(f"percentile deve essere in [0, 100], ricevuto {percentile}")
    return float(np.percentile(distances, percentile, method="linear"))


def compute_cluster_thresholds(distances, labels, n_clusters, percentile, min_cluster_size=100):
    """Soglia al percentile dato, globale e per ciascun cluster in `range(n_clusters)`.

    Un cluster con meno di `min_cluster_size` distanze finite usa esplicitamente la soglia
    globale dello stesso percentile invece di un quantile calcolato su un campione troppo
    piccolo per essere affidabile; il fallback è segnalato per cluster in `fallback_used`.
    """
    distances = np.asarray(distances, dtype=float)
    labels = np.asarray(labels)
    if distances.ndim != 1:
        raise ValueError(f"distances deve avere 1 dimensione, trovate {distances.ndim}")
    if distances.shape[0] == 0:
        raise ValueError("distances non può essere vuoto")
    if labels.ndim != 1:
        raise ValueError(f"labels deve avere 1 dimensione, trovate {labels.ndim}")
    if distances.shape[0] != labels.shape[0]:
        raise ValueError("distances e labels devono avere la stessa lunghezza")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError(f"n_clusters deve essere un intero positivo, ricevuto {n_clusters}")
    if not isinstance(min_cluster_size, (int, np.integer)) or min_cluster_size < 1:
        raise ValueError(f"min_cluster_size deve essere un intero positivo, ricevuto {min_cluster_size}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels deve contenere interi, trovato dtype {labels.dtype}")
    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError(f"labels contiene valori fuori dall'intervallo valido [0, {n_clusters - 1}]")

    global_threshold = compute_percentile_threshold(distances, percentile)

    per_cluster_threshold = {}
    fallback_used = {}
    for cluster in range(n_clusters):
        cluster_distances = distances[labels == cluster]
        if cluster_distances.shape[0] < min_cluster_size:
            per_cluster_threshold[cluster] = global_threshold
            fallback_used[cluster] = True
        else:
            per_cluster_threshold[cluster] = compute_percentile_threshold(cluster_distances, percentile)
            fallback_used[cluster] = False

    return {
        "global_threshold": global_threshold,
        "per_cluster_threshold": per_cluster_threshold,
        "fallback_used": fallback_used,
    }


def resolve_row_thresholds(labels, per_cluster_threshold):
    """Mappa ogni riga alla soglia del proprio cluster (dict `cluster -> soglia`)."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels deve avere 1 dimensione, trovate {labels.ndim}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels deve contenere interi, trovato dtype {labels.dtype}")
    try:
        row_thresholds = np.array([per_cluster_threshold[label] for label in labels], dtype=float)
    except KeyError as exc:
        raise ValueError(f"nessuna soglia definita per il cluster {exc.args[0]}") from exc
    if not np.all(np.isfinite(row_thresholds)):
        raise ValueError("le soglie risolte contengono valori non finiti (NaN o infinito)")
    if np.any(row_thresholds < 0):
        raise ValueError("le soglie risolte contengono valori negativi")
    return row_thresholds


def classify_anomalies(distances, thresholds):
    """Flag booleano: anomalo soltanto quando `distanza > soglia` (confronto stretto, non `>=`,
    così un valore esattamente uguale alla soglia resta classificato come normale).

    `thresholds` può essere uno scalare (soglia globale) o un array 1D della stessa lunghezza di
    `distances` (tipicamente prodotto da `resolve_row_thresholds`).
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1:
        raise ValueError(f"distances deve avere 1 dimensione, trovate {distances.ndim}")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances contiene valori non finiti (NaN o infinito)")
    if np.any(distances < 0):
        raise ValueError("distances contiene valori negativi: uno score di distanza non può esserlo")

    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim not in (0, 1) or (thresholds.ndim == 1 and thresholds.shape[0] != distances.shape[0]):
        raise ValueError(
            f"thresholds deve essere uno scalare o un array 1D della stessa lunghezza di distances "
            f"({distances.shape[0]}), trovata forma {thresholds.shape}"
        )
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds contiene valori non finiti (NaN o infinito)")
    if np.any(thresholds < 0):
        raise ValueError("thresholds contiene valori negativi: una soglia di distanza non può esserlo")

    return distances > thresholds
