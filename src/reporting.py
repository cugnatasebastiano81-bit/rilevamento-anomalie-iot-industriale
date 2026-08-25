"""Fase 6 — reporting e visualizzazione: funzioni pure, testate in isolamento.

Il notebook importa queste funzioni invece di ridefinirle inline: qui vivono le aggregazioni e le
selezioni deterministiche che alimentano i grafici finali, così testo, tabelle e figure derivano
dagli stessi calcoli invece di essere ricopiati a mano.

Nessuna funzione di questo modulo modifica scaler, PCA, K-Means, distanze, soglie o flag di
anomalia: riceve in input risultati già calcolati nelle fasi precedenti e produce soltanto viste
destinate alla presentazione. Il campionamento, dove presente, riguarda esclusivamente la
leggibilità di un grafico: è deterministico a parità di input e seed, non tocca mai i conteggi,
le metriche o le decisioni del modello.
"""
import numpy as np
import pandas as pd

from src.validation import to_binary_array

# Segnali di macchina usati nella matrice di dispersione della Fase 6: uno per famiglia fisica
# (giri, temperatura, vibrazione, corrente assorbita), per non riempire la griglia di variabili
# che raccontano la stessa cosa.
PAIR_PLOT_SIGNALS = ["rpm", "temp_c", "vib_rms", "current_a"]


def _require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} non contiene le colonne {missing}")


def _require_non_empty(df, name):
    if len(df) == 0:
        raise ValueError(f"{name} non può essere vuoto")


def _require_tz_aware_timestamps(df, col, name):
    """Stesso contratto temporale già richiesto dall'allineamento per chiave della fase precedente.

    Un timestamp naive o con timezone diversa può far cadere nello stesso intervallo istanti che
    non coincidono davvero: qui viene rifiutato esplicitamente invece di essere accettato in
    silenzio e produrre una cella di heatmap plausibile ma sbagliata.
    """
    series = df[col]
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError(f"{name}: la colonna '{col}' deve essere di tipo datetime")
    if series.isna().any():
        raise ValueError(f"{name}: la colonna '{col}' contiene valori mancanti (NaT)")
    if getattr(series.dtype, "tz", None) is None:
        raise ValueError(f"{name}: la colonna '{col}' deve essere timezone-aware")


def _require_unique_keys(df, key_cols, name):
    _require_columns(df, key_cols, name)
    if df.duplicated(subset=list(key_cols)).any():
        raise ValueError(f"{name} contiene chiavi duplicate su {list(key_cols)}")


def alert_heatmap_table(
    df,
    timestamp_col="timestamp",
    group_col="regime",
    alert_col="is_anomaly_primary",
    freq="1h",
    group_values=None,
    key_cols=("asset_id", "timestamp"),
):
    """Concentrazione degli alert per cella (intervallo temporale x gruppo), con i denominatori.

    Restituisce una tabella in forma lunga con una riga per ogni combinazione di intervallo
    temporale e gruppo, comprese le combinazioni senza alcuna osservazione: `support` conta le
    righe disponibili nella cella, `alerts` gli alert prodotti e `alert_rate` il rapporto tra i
    due. Una cella senza osservazioni ha `support` 0 e `alert_rate` non definito (`NaN`), mai
    `0%`: le due cose significherebbero l'opposto l'una dell'altra e vanno distinte nel grafico.

    Gli intervalli coprono senza buchi il periodo osservato, in modo che un'ora senza dati resti
    visibile come tale invece di sparire dall'asse.

    Le righe devono essere identificate da una chiave univoca (`key_cols`, per questo progetto asset e
    istante): due righe con la stessa chiave sarebbero la stessa osservazione conteggiata due volte, e
    farebbero salire in silenzio sia il supporto sia il numero di alert della cella. Vengono quindi
    rifiutate, mentre due entità distinte allo stesso istante restano due osservazioni valide.
    """
    _require_non_empty(df, "df")
    _require_columns(df, [timestamp_col, group_col, alert_col], "df")
    _require_unique_keys(df, key_cols, "df")
    _require_tz_aware_timestamps(df, timestamp_col, "df")

    offset = pd.tseries.frequencies.to_offset(freq)
    if pd.Timedelta(offset) <= pd.Timedelta(0):
        raise ValueError(f"freq deve corrispondere a una durata positiva, ricevuto {freq!r}")

    if df[group_col].isna().any():
        raise ValueError(f"la colonna di raggruppamento '{group_col}' contiene valori mancanti")

    alerts = to_binary_array(df[alert_col], alert_col)

    bins = df[timestamp_col].dt.floor(freq)
    if group_values is None:
        group_values = sorted(df[group_col].unique())
    else:
        group_values = list(group_values)
        if len(set(group_values)) != len(group_values):
            raise ValueError("group_values contiene valori duplicati")
        unexpected = sorted(set(df[group_col].unique()) - set(group_values))
        if unexpected:
            raise ValueError(f"il gruppo contiene valori non dichiarati in group_values: {unexpected}")

    counted = df[[group_col]].copy()
    counted["bin_start"] = bins
    counted["alert"] = alerts
    grouped = counted.groupby(["bin_start", group_col], observed=True)["alert"].agg(["size", "sum"])
    grouped.columns = ["support", "alerts"]

    all_bins = pd.date_range(start=bins.min(), end=bins.max(), freq=offset)
    full_index = pd.MultiIndex.from_product([all_bins, group_values], names=["bin_start", group_col])
    table = grouped.reindex(full_index, fill_value=0).reset_index()

    table["alert_rate"] = np.where(table["support"] > 0, table["alerts"] / table["support"], np.nan)
    return table.sort_values(["bin_start", group_col]).reset_index(drop=True)


def stratified_sample(df, group_col, fraction=None, size=None, seed=42, key_cols=("asset_id", "timestamp")):
    """Campione stratificato deterministico, pensato solo per la leggibilità di un grafico.

    Le righe vengono scelte all'interno di ciascun gruppo dopo averle ordinate per chiave, quindi
    lo stesso insieme di dati produce sempre le stesse chiavi anche se arriva in un ordine diverso.
    Va indicata esattamente una tra `fraction` (quota per gruppo) e `size` (numero di righe per
    gruppo); se un gruppo ha meno righe di quante ne sono richieste, viene preso per intero.

    Il risultato serve a disegnare, mai a calcolare: conteggi, metriche e soglie restano quelli
    dell'insieme completo.
    """
    key_cols = list(key_cols)
    _require_non_empty(df, "df")
    _require_columns(df, [group_col], "df")
    _require_unique_keys(df, key_cols, "df")

    if (fraction is None) == (size is None):
        raise ValueError("indicare esattamente una tra fraction e size")
    if fraction is not None:
        if not np.isfinite(fraction) or not (0 < fraction <= 1):
            raise ValueError(f"fraction deve stare in (0, 1], ricevuto {fraction!r}")
    if size is not None:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError(f"size deve essere un intero >= 1, ricevuto {size!r}")

    ordered = df.sort_values(key_cols)
    selected = []
    rng = np.random.default_rng(seed)
    for value in sorted(df[group_col].unique()):
        part = ordered[ordered[group_col] == value]
        n_available = len(part)
        if size is not None:
            n_wanted = min(int(size), n_available)
        else:
            n_wanted = max(1, int(n_available * fraction))
        positions = np.sort(rng.choice(n_available, size=n_wanted, replace=False))
        selected.append(part.iloc[positions])

    return pd.concat(selected).sort_values(key_cols)


def split_background_and_alerts(
    df,
    alert_col="is_anomaly_primary",
    group_col="cluster",
    fraction=0.05,
    seed=42,
    key_cols=("asset_id", "timestamp"),
):
    """Separa lo sfondo campionabile dagli alert, che restano tutti.

    Un grafico che dichiara di mostrare gli alert deve mostrarli tutti: qui il campionamento tocca
    soltanto le righe non segnalate, che servono a dare contesto visivo. Restituisce due tabelle
    disgiunte, sfondo e alert, entrambe ordinate per chiave.
    """
    _require_non_empty(df, "df")
    _require_columns(df, [alert_col, group_col], "df")
    key_cols = list(key_cols)
    _require_unique_keys(df, key_cols, "df")

    alerts_mask = to_binary_array(df[alert_col], alert_col)
    alerts = df[alerts_mask].sort_values(key_cols)
    others = df[~alerts_mask]
    if len(others) == 0:
        background = others.sort_values(key_cols)
    else:
        background = stratified_sample(
            others, group_col=group_col, fraction=fraction, seed=seed, key_cols=key_cols
        )
    return background, alerts


def distances_by_flag(distances, flags):
    """Divide le distanze dal centroide tra righe classificate normali e righe segnalate.

    Restituisce le due distribuzioni e le rispettive numerosità, che il grafico deve dichiarare:
    la separazione tra le due è in parte costruita per definizione, perché la classe deriva dalla
    stessa distanza confrontata con la soglia, quindi il valore informativo sta nella forma delle
    distribuzioni e nella loro sovrapposizione, non nel fatto che siano diverse.
    """
    dist = np.asarray(distances, dtype=float)
    if dist.ndim != 1:
        raise ValueError(f"distances deve avere 1 dimensione, trovate {dist.ndim}")
    if dist.shape[0] == 0:
        raise ValueError("distances non può essere vuoto")
    if not np.all(np.isfinite(dist)):
        raise ValueError("distances contiene valori mancanti o non finiti (NaN o infinito)")
    if np.any(dist < 0):
        raise ValueError("distances contiene valori negativi")

    mask = to_binary_array(flags, "flags")
    if mask.shape[0] != dist.shape[0]:
        raise ValueError(f"distances ({dist.shape[0]}) e flags ({mask.shape[0]}) hanno lunghezze diverse")

    return {
        "normal": dist[~mask],
        "anomalous": dist[mask],
        "n_normal": int((~mask).sum()),
        "n_anomalous": int(mask.sum()),
    }


def pair_plot_frame(
    df,
    feature_cols=PAIR_PLOT_SIGNALS,
    group_col="cluster",
    size=400,
    seed=42,
    key_cols=("asset_id", "timestamp"),
):
    """Sottoinsieme deterministico per la matrice di dispersione dei segnali di macchina.

    Restituisce chiavi, segnali richiesti e gruppo di appartenenza per un numero fisso di righe per
    gruppo: disegnare tutte le righe renderebbe illeggibile ogni riquadro. Il gruppo di ogni riga
    resta quello già assegnato, il campionamento non lo ricalcola.
    """
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols non può essere vuoto")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols contiene colonne duplicate")
    _require_non_empty(df, "df")
    _require_columns(df, feature_cols + [group_col], "df")

    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"la colonna '{col}' contiene valori mancanti o non finiti")

    sample = stratified_sample(df, group_col=group_col, size=size, seed=seed, key_cols=key_cols)
    return sample[list(key_cols) + feature_cols + [group_col]]
