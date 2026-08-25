"""Fase 5 — validazione tecnica: funzioni pure, testate in isolamento in tests/test_validation.py.

Il notebook (notebooks/) importa queste funzioni invece di ridefinirle inline; il ragionamento
sul perché di ogni scelta resta nel notebook, qui c'è solo il codice. Nessuna funzione qui
modifica scaler, PCA, K-Means, centroidi, distanze o soglie: prendono in input predizioni e label
già calcolate altrove e producono soltanto metriche di valutazione.
"""
import numpy as np
import pandas as pd

MIN_RELIABLE_COUNT = 20
# z-score per l'intervallo di confidenza bilaterale al 95% di una normale standard.
Z_95 = 1.959963984540054


def merge_predictions_and_labels(predictions, labels, key_cols=("asset_id", "timestamp")):
    """Allinea predizioni e label tramite la chiave univoca `key_cols`, mai un join posizionale.

    Richiede chiavi uniche in entrambi i DataFrame e un merge uno-a-uno che non perde né duplica
    righe: `predictions` e `labels` devono avere la stessa cardinalità e lo stesso insieme di
    chiavi, altrimenti solleva `ValueError` esplicito invece di produrre un allineamento silenzioso.

    Ogni colonna chiave il cui nome contiene "timestamp" deve essere una colonna datetime
    timezone-aware, senza `NaT`, con lo stesso timezone in `predictions` e in `labels`: un
    confronto tra timestamp naive o tra timezone diverse può allineare righe che non rappresentano
    davvero lo stesso istante, quindi qui viene rifiutato esplicitamente invece di essere accettato
    in silenzio.
    """
    key_cols = list(key_cols)
    for name, df in (("predictions", predictions), ("labels", labels)):
        missing = [c for c in key_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} non contiene le colonne chiave {missing}")
        if df.duplicated(subset=key_cols).any():
            raise ValueError(f"{name} contiene chiavi duplicate su {key_cols}")

    for col in key_cols:
        if "timestamp" not in col.lower():
            continue
        for name, df in (("predictions", predictions), ("labels", labels)):
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise ValueError(f"{name}['{col}'] deve essere una colonna datetime")
            if df[col].isna().any():
                raise ValueError(f"{name}['{col}'] contiene valori NaT")
            if df[col].dt.tz is None:
                raise ValueError(f"{name}['{col}'] deve essere timezone-aware")
        if str(predictions[col].dt.tz) != str(labels[col].dt.tz):
            raise ValueError(
                f"predictions['{col}'] e labels['{col}'] hanno timezone diverse: "
                f"{predictions[col].dt.tz} vs {labels[col].dt.tz}"
            )

    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions e labels hanno cardinalità diversa: {len(predictions)} vs {len(labels)}"
        )
    if set(map(tuple, predictions[key_cols].to_numpy())) != set(map(tuple, labels[key_cols].to_numpy())):
        raise ValueError("predictions e labels non condividono lo stesso insieme di chiavi")

    # A questo punto le chiavi sono uniche in entrambi i DataFrame, coincidono come insieme e hanno
    # la stessa cardinalità: un merge 1-a-1 su queste chiavi produce sempre esattamente
    # len(predictions) righe. `validate="one_to_one"` resta comunque come controllo di pandas.
    return predictions.merge(labels, on=key_cols, how="inner", validate="one_to_one")


def to_binary_array(series, name):
    """Converte una colonna in array booleano, rifiutando NaN, valori non finiti o diversi da 0/1."""
    arr = np.asarray(series)
    if arr.ndim != 1:
        raise ValueError(f"{name} deve avere 1 dimensione, trovate {arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} non può essere vuoto")
    if arr.dtype == bool:
        return arr.copy()
    if arr.dtype == object:
        raise ValueError(f"{name} contiene valori non numerici/booleani")

    arr_float = arr.astype(float)
    if not np.all(np.isfinite(arr_float)):
        raise ValueError(f"{name} contiene valori mancanti o non finiti (NaN o infinito)")
    unique_vals = set(np.unique(arr_float))
    if not unique_vals <= {0.0, 1.0}:
        raise ValueError(f"{name} contiene valori diversi da 0/1: {sorted(unique_vals - {0.0, 1.0})}")
    return arr_float.astype(bool)


def build_fault_targets(fault_code_true, anomaly_label, admitted_fault_codes=(0, 1, 2, 3, 4)):
    """Costruisce `fault_known` e `anomaly_partial` dalle due colonne sorgente, validandole prima.

    `fault_code_true` deve contenere soltanto codici interi finiti tra quelli effettivamente
    ammessi dal dataset (`admitted_fault_codes`, per default `0` = nessun guasto e `1..4` i codici
    di guasto reali osservati nei dati); un `NaN`, un infinito o un codice fuori da questo insieme
    solleva `ValueError` invece di essere convertito silenziosamente in "nessun guasto" o in
    "guasto". `anomaly_label` è validata come binaria 0/1 da `to_binary_array`. Le due colonne
    devono avere la stessa lunghezza. Ritorna `(fault_known, anomaly_partial)`, due array int 0/1.
    """
    fault_arr = np.asarray(fault_code_true)
    if fault_arr.ndim != 1:
        raise ValueError(f"fault_code_true deve avere 1 dimensione, trovate {fault_arr.ndim}")
    if fault_arr.shape[0] == 0:
        raise ValueError("fault_code_true non può essere vuoto")
    if fault_arr.dtype == object:
        raise ValueError("fault_code_true contiene valori non numerici")

    fault_float = fault_arr.astype(float)
    if not np.all(np.isfinite(fault_float)):
        raise ValueError("fault_code_true contiene valori mancanti o non finiti (NaN o infinito)")
    if not np.all(fault_float == np.round(fault_float)):
        raise ValueError("fault_code_true deve contenere soltanto codici interi")
    fault_int = fault_float.astype(np.int64)

    admitted = set(admitted_fault_codes)
    unexpected = sorted(set(np.unique(fault_int).tolist()) - admitted)
    if unexpected:
        raise ValueError(f"fault_code_true contiene codici non ammessi: {unexpected}")

    anomaly_bool = to_binary_array(anomaly_label, "anomaly_label")
    if fault_int.shape[0] != anomaly_bool.shape[0]:
        raise ValueError(
            f"fault_code_true e anomaly_label devono avere la stessa lunghezza: "
            f"{fault_int.shape[0]} vs {anomaly_bool.shape[0]}"
        )

    fault_known = (fault_int != 0).astype(int)
    anomaly_partial = anomaly_bool.astype(int)
    return fault_known, anomaly_partial


def confusion_counts(y_true, y_pred):
    """TN/FP/FN/TP con convenzione righe=classe reale [normale, anomala], colonne=predetta.

    Confronto entro tolleranza esatta (interi): nessun arrotondamento coinvolto, i due vettori
    devono avere la stessa lunghezza dopo la validazione binaria di entrambi.
    """
    y_true = to_binary_array(y_true, "y_true")
    y_pred = to_binary_array(y_pred, "y_pred")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true e y_pred devono avere la stessa lunghezza: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )
    tn = int(np.sum((~y_true) & (~y_pred)))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    tp = int(np.sum(y_true & y_pred))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def classification_metrics(y_true, y_pred):
    """Precision, recall, F1 sulla classe positiva e conteggi di supporto.

    Policy divisioni per zero: precision/recall/F1 valgono 0.0 quando il denominatore è nullo
    (coerente con `zero_division=0`), con un indicatore booleano esplicito accanto a ogni metrica
    così un risultato zero non informativo non viene confuso con un risultato zero calcolato.
    """
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    support = tn + fp + fn + tp
    positives_real = fn + tp
    positives_pred = fp + tp

    precision_undefined = positives_pred == 0
    recall_undefined = positives_real == 0
    precision = 0.0 if precision_undefined else tp / positives_pred
    recall = 0.0 if recall_undefined else tp / positives_real
    f1_undefined = (precision + recall) == 0
    f1 = 0.0 if f1_undefined else 2 * precision * recall / (precision + recall)

    return {
        **counts,
        "support": support,
        "positives_real": positives_real,
        "positives_pred": positives_pred,
        "prevalence": positives_real / support,
        "alert_rate": positives_pred / support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_undefined": precision_undefined,
        "recall_undefined": recall_undefined,
        "f1_undefined": f1_undefined,
    }


def wilson_interval(successes, n, z=Z_95):
    """Intervallo di confidenza di Wilson bilaterale per una proporzione `successes / n`.

    Ritorna `None` quando `n` è zero: la proporzione non è definita, nessun intervallo improprio
    viene restituito al suo posto.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"successes={successes} e n={n} non descrivono una proporzione valida")
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return (float(lo), float(hi))


def classification_metrics_with_intervals(y_true, y_pred):
    """`classification_metrics` più intervalli di Wilson al 95% per alert rate, precision e recall.

    F1 non riceve un intervallo Wilson (non è una proporzione binomiale semplice): resta assente
    dal risultato, non sostituito da un intervallo improprio.
    """
    metrics = classification_metrics(y_true, y_pred)
    metrics["alert_rate_ci95"] = wilson_interval(metrics["positives_pred"], metrics["support"])
    metrics["precision_ci95"] = wilson_interval(metrics["tp"], metrics["positives_pred"])
    metrics["recall_ci95"] = wilson_interval(metrics["tp"], metrics["positives_real"])
    return metrics


def metrics_by_group(df, group_col, y_true_col, y_pred_col, min_count=MIN_RELIABLE_COUNT):
    """Metriche di classificazione per ciascun valore di `group_col`.

    Un gruppo è marcato `fragile` (stima non affidabile, da non usare per ordinare o proclamare un
    gruppo migliore/peggiore) quando i positivi reali o i positivi predetti nel gruppo sono sotto
    `min_count`. Gli intervalli di Wilson restano comunque calcolati quando il denominatore lo
    consente.
    """
    if group_col not in df.columns:
        raise ValueError(f"{group_col} non è una colonna di df")
    rows = []
    for value, part in df.groupby(group_col, sort=True):
        metrics = classification_metrics_with_intervals(part[y_true_col], part[y_pred_col])
        metrics[group_col] = value
        metrics["n"] = len(part)
        metrics["fragile"] = metrics["positives_real"] < min_count or metrics["positives_pred"] < min_count
        rows.append(metrics)
    return pd.DataFrame(rows).set_index(group_col)


def recall_by_fault_type(df, fault_type_col, y_pred_col, min_count=MIN_RELIABLE_COUNT):
    """Tasso di rilevamento (recall) per tipo di guasto, su righe già filtrate a guasto noto.

    Ogni riga in ingresso deve appartenere alla classe positiva di `fault_known` (il chiamante
    filtra prima); non esiste una classe negativa dentro ciascun gruppo, quindi qui si riporta
    soltanto quanti di quei guasti sono stati rilevati (`recall`), non precision/F1.
    """
    if fault_type_col not in df.columns:
        raise ValueError(f"{fault_type_col} non è una colonna di df")
    if len(df) == 0:
        raise ValueError("df non può essere vuoto")
    rows = []
    for fault_type, part in df.groupby(fault_type_col, sort=True):
        y_pred = to_binary_array(part[y_pred_col], y_pred_col)
        n = len(part)
        detected = int(y_pred.sum())
        rows.append({
            fault_type_col: fault_type,
            "n": n,
            "n_detected": detected,
            "recall": detected / n,
            "recall_ci95": wilson_interval(detected, n),
            "fragile": n < min_count,
        })
    return pd.DataFrame(rows).set_index(fault_type_col)


def select_confusion_cases(df, y_true_col, y_pred_col, ratio_col, key_cols, category, n=10):
    """Seleziona fino a `n` casi di `category` ('fp' o 'fn'), ordinati in modo deterministico.

    Ordine: `ratio_col` (tipicamente distanza/soglia) decrescente, poi `key_cols` crescente — mai
    una scelta discrezionale di quali esempi mostrare.
    """
    if category not in ("fp", "fn"):
        raise ValueError(f"category deve essere 'fp' o 'fn', ricevuto {category!r}")
    y_true = to_binary_array(df[y_true_col], y_true_col)
    y_pred = to_binary_array(df[y_pred_col], y_pred_col)
    mask = (~y_true) & y_pred if category == "fp" else y_true & (~y_pred)

    subset = df.loc[mask].copy()
    subset = subset.sort_values(
        by=[ratio_col, *key_cols], ascending=[False] + [True] * len(key_cols)
    )
    return subset.head(n)


def segment_fault_events(df, asset_col, timestamp_col, fault_col, expected_freq=pd.Timedelta(minutes=1)):
    """Segmenta finestre temporali contigue con `fault_col != 0`, separatamente per `asset_col`.

    Due righe di fault consecutive dello stesso asset appartengono allo stesso evento soltanto se
    il delta tra i loro timestamp non supera `expected_freq` (di default 1 minuto, la frequenza
    dichiarata del dataset): un gap più ampio — tipicamente righe scartate a monte per un buco di
    missing troppo lungo da interpolare — chiude l'evento corrente e ne apre uno nuovo, invece di
    fondere silenziosamente un intervallo temporale mai osservato in un solo evento.

    Un evento è marcato troncato all'inizio o alla fine quando coincide con la prima o l'ultima
    riga osservata per quell'asset nel DataFrame in ingresso: il guasto potrebbe essere iniziato
    prima o continuare dopo l'intervallo osservato, quindi non va trattato come un evento completo.
    """
    required = {asset_col, timestamp_col, fault_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti: {sorted(missing)}")
    if not isinstance(expected_freq, pd.Timedelta) or expected_freq <= pd.Timedelta(0):
        raise ValueError(f"expected_freq deve essere un pd.Timedelta positivo, ricevuto {expected_freq!r}")

    events = []
    for asset, part in df.sort_values([asset_col, timestamp_col]).groupby(asset_col, sort=True):
        part = part.reset_index(drop=True)
        is_fault = (part[fault_col] != 0).to_numpy()
        timestamps = part[timestamp_col]
        n = len(part)
        if not is_fault.any():
            continue

        def _emit(start_pos, end_pos, asset=asset, timestamps=timestamps, n=n):
            events.append({
                asset_col: asset,
                "start": timestamps.iloc[start_pos],
                "end": timestamps.iloc[end_pos],
                "n_rows": end_pos - start_pos + 1,
                "truncated_start": start_pos == 0,
                "truncated_end": end_pos == n - 1,
            })

        start_pos = None
        prev_pos = None
        for i in range(n):
            if is_fault[i]:
                if start_pos is None:
                    start_pos = i
                elif timestamps.iloc[i] - timestamps.iloc[prev_pos] > expected_freq:
                    _emit(start_pos, prev_pos)
                    start_pos = i
                prev_pos = i
            elif start_pos is not None:
                _emit(start_pos, prev_pos)
                start_pos = None
                prev_pos = None
        if start_pos is not None:
            _emit(start_pos, prev_pos)

    return pd.DataFrame(
        events,
        columns=[asset_col, "start", "end", "n_rows", "truncated_start", "truncated_end"],
    )


def event_alert_coverage(events, df, asset_col, timestamp_col, alert_col):
    """Per ciascun evento di `segment_fault_events`: presenza di almeno un alert, ritardo in minuti
    del primo alert rispetto all'inizio dell'evento (`NaN` se nessun alert nella finestra) e
    copertura (frazione di righe dell'evento con alert attivo).
    """
    rows = []
    for _, ev in events.iterrows():
        asset = ev[asset_col]
        window = df[
            (df[asset_col] == asset)
            & (df[timestamp_col] >= ev["start"])
            & (df[timestamp_col] <= ev["end"])
        ]
        alert_flags = window[alert_col].to_numpy()
        has_alert = bool(alert_flags.any())
        if has_alert:
            first_alert_ts = window.loc[window[alert_col], timestamp_col].min()
            delay_minutes = (first_alert_ts - ev["start"]).total_seconds() / 60.0
        else:
            delay_minutes = np.nan
        rows.append({
            asset_col: asset,
            "start": ev["start"],
            "end": ev["end"],
            "n_rows": ev["n_rows"],
            "truncated_start": ev["truncated_start"],
            "truncated_end": ev["truncated_end"],
            "has_alert": has_alert,
            "delay_minutes": delay_minutes,
            "coverage": float(alert_flags.mean()),
        })
    return pd.DataFrame(rows)


def alerts_near_fault_windows(df, asset_col, timestamp_col, alert_col, events, window_minutes=15):
    """Quota di alert (`alert_col` vero) entro `window_minutes` da una finestra di guasto dello
    stesso asset (bordi della finestra inclusi, estesi di `window_minutes` su entrambi i lati).
    """
    alerts = df.loc[df[alert_col]].copy()
    if alerts.empty:
        return {"n_alerts": 0, "n_near": 0, "fraction_near": np.nan}

    delta = pd.Timedelta(minutes=window_minutes)
    near = pd.Series(False, index=alerts.index)
    for asset, asset_events in events.groupby(asset_col):
        asset_mask = alerts[asset_col] == asset
        if not asset_mask.any():
            continue
        asset_alert_ts = alerts.loc[asset_mask, timestamp_col]
        asset_near = pd.Series(False, index=asset_alert_ts.index)
        for _, ev in asset_events.iterrows():
            lo, hi = ev["start"] - delta, ev["end"] + delta
            asset_near |= asset_alert_ts.between(lo, hi)
        near.loc[asset_near.index] = asset_near

    n_near = int(near.sum())
    return {"n_alerts": len(alerts), "n_near": n_near, "fraction_near": n_near / len(alerts)}
