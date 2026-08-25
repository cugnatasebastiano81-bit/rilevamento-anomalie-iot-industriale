"""Fase 1 — ingestione e controllo qualità: funzioni pure, testate in isolamento in tests/test_pipeline.py.

Il notebook (notebooks/) importa queste funzioni invece di ridefinirle inline; il ragionamento
sul perché di ogni scelta resta nel notebook, qui c'è solo il codice.
"""
import pandas as pd

PHYSICAL_BOUNDS = {
    "rpm": (0, None),
    "current_a": (0, None),
    "pressure_bar": (0, None),
    "flow_lpm": (0, None),
    "vib_rms": (0, None),
    "load_pct": (0, 100),
    "humidity_pct": (0, 100),
}

SENSOR_COLS = [
    "ambient_temp_c", "humidity_pct", "load_pct",
    "rpm", "current_a", "pressure_bar", "flow_lpm",
    "temp_c", "vib_rms", "vib_crest", "vib_kurtosis",
]


def load_and_sort(path, parse_dates=("timestamp",)):
    df = pd.read_csv(path, parse_dates=list(parse_dates))
    return df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


def clip_physical_bounds(df, bounds=PHYSICAL_BOUNDS):
    """Clippa i valori fisicamente impossibili ai limiti dichiarati; ritorna (df_clippato, report)."""
    df = df.copy()
    report = {}
    for col, (lo, hi) in bounds.items():
        n_below = int((df[col] < lo).sum()) if lo is not None else 0
        n_above = int((df[col] > hi).sum()) if hi is not None else 0
        df[col] = df[col].clip(lower=lo, upper=hi)
        report[col] = {"n_below_min": n_below, "n_above_max": n_above}
    return df, report


def split_temporal(df, train_end, test_start):
    """Split train/test temporale, nessuno shuffle. Deve avvenire PRIMA di interpolate_per_asset.

    `df_train` è `timestamp <= train_end` (confine incluso), `df_test` è `timestamp >= test_start`
    (confine incluso). Un eventuale gap tra `train_end` e `test_start` è ammesso (va scelto
    intenzionalmente dal chiamante passando i due argomenti), ma `train_end` deve restare
    strettamente precedente a `test_start`: confini uguali o invertiti produrrebbero una riga
    duplicata o assegnata a entrambe le parti, quindi sollevano `ValueError` esplicito invece di
    passare inosservati. Solleva `ValueError` anche se `df["timestamp"]` contiene valori non validi
    (`NaT`) o se una delle due parti risulta vuota.
    """
    if pd.isna(train_end) or pd.isna(test_start):
        raise ValueError("train_end e test_start devono essere timestamp validi, non NaT")
    if train_end >= test_start:
        raise ValueError(
            f"train_end ({train_end}) deve essere strettamente precedente a test_start "
            f"({test_start}): confini uguali o sovrapposti non sono ammessi"
        )
    if df["timestamp"].isna().any():
        raise ValueError("df contiene timestamp non validi (NaT): impossibile splittare in modo affidabile")

    df_train = df[df["timestamp"] <= train_end].copy()
    df_test = df[df["timestamp"] >= test_start].copy()

    if df_train.empty:
        raise ValueError(f"df_train risultante è vuoto: nessuna riga con timestamp <= {train_end}")
    if df_test.empty:
        raise ValueError(f"df_test risultante è vuoto: nessuna riga con timestamp >= {test_start}")

    return df_train, df_test


def interpolate_per_asset(part, cols=SENSOR_COLS, limit=5):
    """Interpolazione temporale (method="time") per singolo asset, con limite massimo di gap colmabile.

    Va chiamata separatamente su df_train e df_test (mai sull'unione) per evitare che un buco vicino
    al confine dello split venga stimato usando dati dall'altro lato.
    """
    part_idx = part.set_index("timestamp")

    def interpolate_asset(group):
        group[cols] = group[cols].interpolate(method="time", limit=limit, limit_direction="forward")
        return group

    out = part_idx.groupby("asset_id", group_keys=False)[part_idx.columns].apply(interpolate_asset)
    return out.reset_index()


def drop_residual_na(df, cols=SENSOR_COLS):
    """Esclude le righe con NaN residuo dopo l'interpolazione (gap troppo lunghi per essere colmati)."""
    return df.dropna(subset=cols).copy()
