"""Fase 2 — feature engineering: funzioni pure, testate in isolamento in tests/test_features.py.

Il notebook (notebooks/) importa queste funzioni invece di ridefinirle inline; il ragionamento
sul perché di ogni scelta resta nel notebook, qui c'è solo il codice.
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MACHINE_COLS = [
    "rpm", "current_a", "pressure_bar", "flow_lpm",
    "temp_c", "vib_rms", "vib_crest", "vib_kurtosis",
]
CONTEXT_NUM_COLS = ["ambient_temp_c", "humidity_pct", "load_pct"]


def add_regime_onehot(part, categories):
    """One-hot su `regime`, con colonne fissate su `categories` (tipicamente le categorie osservate
    nel train) così che train e test producano sempre lo stesso insieme di colonne.

    Un valore mancante o non compreso in `categories` non deve diventare silenziosamente un vettore
    one-hot tutto a zero (categoria implicita non documentata): si solleva `ValueError` esplicito,
    così un dato fuori contratto viene segnalato invece di passare inosservato nella pipeline.
    """
    regime = part["regime"]
    if regime.isna().any():
        raise ValueError(
            f"regime contiene {int(regime.isna().sum())} valori mancanti: nessuna policy per i "
            "missing è definita, servono valori validi in tutte le righe"
        )
    unknown = sorted(set(regime.unique()) - set(categories))
    if unknown:
        raise ValueError(
            f"regime contiene valori non tra le categorie ammesse {sorted(categories)}: {unknown}"
        )
    dummies = pd.get_dummies(regime, prefix="regime")
    dummies = dummies.reindex(columns=[f"regime_{r}" for r in categories], fill_value=False)
    return pd.concat([part.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def regime_onehot_cols(categories):
    return [f"regime_{r}" for r in categories]


def add_rolling_and_diff(part, machine_cols=MACHINE_COLS, window="15min", min_periods=15):
    """Media/deviazione standard mobile (finestra basata sul tempo) e diff, per-asset.

    Finestra a offset temporale (non a conteggio campioni) perché la serie per asset non è più
    garantita contigua al minuto dopo l'esclusione delle righe con gap troppo lunghi in Fase 1.
    min_periods esplicito perché il default di pandas per una finestra a tempo (min_periods=1)
    produrrebbe una statistica anche con un solo campione disponibile.
    """
    part = part.sort_values(["asset_id", "timestamp"]).copy()
    part_idx = part.set_index("timestamp")

    def per_asset(group):
        for col in machine_cols:
            group[f"{col}_roll_mean"] = group[col].rolling(window, min_periods=min_periods).mean()
            group[f"{col}_roll_std"] = group[col].rolling(window, min_periods=min_periods).std()
            group[f"{col}_diff"] = group[col].diff()
        return group

    out = part_idx.groupby("asset_id", group_keys=False)[part_idx.columns].apply(per_asset)
    return out.reset_index()


def derived_feature_cols(machine_cols=MACHINE_COLS):
    return (
        [f"{c}_roll_mean" for c in machine_cols]
        + [f"{c}_roll_std" for c in machine_cols]
        + [f"{c}_diff" for c in machine_cols]
    )


def fit_scaler(df_train, feature_cols):
    """StandardScaler fit esclusivamente su df_train (mai sul test, per evitare leakage)."""
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    return scaler


def fit_pca(X_train, n_components, random_state=42):
    """PCA fit esclusivamente su X_train (già standardizzato)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    return pca
