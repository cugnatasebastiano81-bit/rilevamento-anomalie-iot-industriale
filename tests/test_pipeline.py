"""Unit test per src/pipeline.py (Fase 1). Dati sintetici piccoli, nessun I/O reale."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    clip_physical_bounds,
    drop_residual_na,
    interpolate_per_asset,
    split_temporal,
)


def _asset_series(asset_id, start, n, freq="1min", value_fn=lambda i: float(i)):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "asset_id": asset_id,
        "timestamp": ts,
        "value": [value_fn(i) for i in range(n)],
    })


# --- clip_physical_bounds ---

def test_clip_physical_bounds_clips_negative_to_zero():
    df = pd.DataFrame({"rpm": [-5.0, 10.0, 0.0]})
    out, report = clip_physical_bounds(df, bounds={"rpm": (0, None)})
    assert out["rpm"].tolist() == [0.0, 10.0, 0.0]
    assert report["rpm"] == {"n_below_min": 1, "n_above_max": 0}


def test_clip_physical_bounds_clips_upper_bound():
    df = pd.DataFrame({"load_pct": [50.0, 120.0, -3.0]})
    out, report = clip_physical_bounds(df, bounds={"load_pct": (0, 100)})
    assert out["load_pct"].tolist() == [50.0, 100.0, 0.0]
    assert report["load_pct"] == {"n_below_min": 1, "n_above_max": 1}


def test_clip_physical_bounds_does_not_mutate_input():
    df = pd.DataFrame({"rpm": [-5.0]})
    clip_physical_bounds(df, bounds={"rpm": (0, None)})
    assert df["rpm"].tolist() == [-5.0]


# --- split_temporal ---

def test_split_temporal_no_overlap_and_covers_all_rows():
    ts = pd.date_range("2025-02-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    train_end = ts[6]
    test_start = ts[7]
    df_train, df_test = split_temporal(df, train_end, test_start)
    assert len(df_train) + len(df_test) == len(df)
    assert df_train["timestamp"].max() <= train_end
    assert df_test["timestamp"].min() >= test_start
    assert set(df_train.index).isdisjoint(set(df_test.index))


def test_split_temporal_allows_intentional_gap():
    # gap tra train_end e test_start: ammesso, deve restare una scelta esplicita del chiamante,
    # non un errore in sé.
    ts = pd.date_range("2025-02-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    train_end = ts[4]
    test_start = ts[7]  # salta ts[5], ts[6]: gap intenzionale
    df_train, df_test = split_temporal(df, train_end, test_start)
    assert len(df_train) + len(df_test) < len(df)
    assert df_train["timestamp"].max() <= train_end
    assert df_test["timestamp"].min() >= test_start


def test_split_temporal_equal_boundaries_raises():
    ts = pd.date_range("2025-02-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    with pytest.raises(ValueError, match="strettamente precedente"):
        split_temporal(df, ts[2], ts[2])


def test_split_temporal_overlap_raises():
    ts = pd.date_range("2025-02-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    with pytest.raises(ValueError, match="strettamente precedente"):
        split_temporal(df, ts[3], ts[1])  # train_end dopo test_start: sovrapposizione


def test_split_temporal_empty_train_raises():
    ts = pd.date_range("2025-02-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    train_end = ts[0] - pd.Timedelta(days=1)  # prima di ogni riga
    with pytest.raises(ValueError, match="df_train risultante è vuoto"):
        split_temporal(df, train_end, ts[0])


def test_split_temporal_empty_test_raises():
    ts = pd.date_range("2025-02-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    test_start = ts[-1] + pd.Timedelta(days=1)  # dopo ogni riga
    with pytest.raises(ValueError, match="df_test risultante è vuoto"):
        split_temporal(df, ts[-1], test_start)


def test_split_temporal_invalid_timestamp_raises():
    ts = pd.date_range("2025-02-01", periods=5, freq="1D", tz="UTC").to_list()
    ts[2] = pd.NaT
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    with pytest.raises(ValueError, match="timestamp non validi"):
        split_temporal(df, ts[1], ts[3])


def test_split_temporal_nat_boundary_raises():
    ts = pd.date_range("2025-02-01", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "asset_id": 1})
    with pytest.raises(ValueError, match="NaT"):
        split_temporal(df, pd.NaT, ts[3])


# --- interpolate_per_asset ---

def test_interpolate_per_asset_does_not_cross_assets():
    # due asset interlacciati nello stesso dataframe: un buco nell'asset 1 deve essere stimato
    # solo dai vicini dell'asset 1, non dai valori (molto diversi) dell'asset 2.
    a1 = _asset_series(1, "2025-02-01", 5, value_fn=lambda i: 10.0 + i)  # 10,11,12,13,14
    a2 = _asset_series(2, "2025-02-01", 5, value_fn=lambda i: 1000.0 + i)  # 1000..1004
    df = pd.concat([a1, a2], ignore_index=True)
    df.loc[(df["asset_id"] == 1) & (df["value"] == 12.0), "value"] = np.nan

    out = interpolate_per_asset(df, cols=["value"], limit=5)
    filled = out.loc[(out["asset_id"] == 1) & (out["timestamp"] == a1["timestamp"].iloc[2]), "value"].item()
    assert filled == pytest.approx(12.0)  # media tra 11 e 13, non contaminata da ~1000 dell'asset 2


def test_interpolate_per_asset_respects_limit():
    a1 = _asset_series(1, "2025-02-01", 10, value_fn=lambda i: float(i))
    # buco di 6 minuti consecutivi (oltre il limit=5): deve restare NaN
    df = a1.copy()
    df.loc[2:7, "value"] = np.nan

    out = interpolate_per_asset(df, cols=["value"], limit=5)
    assert out["value"].isna().sum() > 0


def test_interpolate_per_asset_fills_short_gap():
    a1 = _asset_series(1, "2025-02-01", 5, value_fn=lambda i: float(i))
    df = a1.copy()
    df.loc[2, "value"] = np.nan  # buco di 1 minuto, ben sotto il limit=5

    out = interpolate_per_asset(df, cols=["value"], limit=5)
    assert out["value"].isna().sum() == 0


def test_split_before_interpolate_prevents_boundary_leakage():
    """Regressione: bug reale trovato durante il debug di fine Fase 1.

    Interpolare l'intera serie e poi tagliare in train/test permette a un NaN vicino al confine
    di essere stimato usando un valore dall'altro lato dello split (limit_direction="forward" con
    limit=5 attinge fino a 5 minuti anche a cavallo del confine). Il fix è invertire l'ordine:
    split prima, interpolazione indipendente dopo. Questo test blocca la regressione a quell'ordine
    sbagliato.
    """
    a1 = _asset_series(1, "2025-02-01 00:00", 6, value_fn=lambda i: float(i))  # 0,1,2,3,4,5
    train_end = a1["timestamp"].iloc[3]  # confine tra indice 3 (train) e indice 4 (test)
    test_start = a1["timestamp"].iloc[4]

    df = a1.copy()
    df.loc[4, "value"] = np.nan  # NaN appena dopo il confine, nel test

    # comportamento CORRETTO: split prima, interpolazione indipendente dopo
    df_train, df_test = split_temporal(df, train_end, test_start)
    df_test_interp = interpolate_per_asset(df_test, cols=["value"], limit=5)
    # il buco è la primissima riga del test: nessun valore noto precedente esiste nel test stesso
    # per stimarlo con limit_direction="forward", quindi deve restare NaN
    value_at_gap = df_test_interp.loc[df_test_interp["timestamp"] == a1["timestamp"].iloc[4], "value"].item()
    assert pd.isna(value_at_gap), (
        "con lo split prima dell'interpolazione il buco a inizio test non ha dati propri per essere "
        "stimato e deve restare NaN, non essere riempito con informazione presa dal train"
    )

    # comportamento SBAGLIATO (quello del bug originale): interpola tutta la serie, poi splitta
    df_interp_then_split = interpolate_per_asset(df, cols=["value"], limit=5)
    train_wrong, test_wrong = split_temporal(df_interp_then_split, train_end, test_start)
    value_leaked = test_wrong.loc[test_wrong["timestamp"] == a1["timestamp"].iloc[4], "value"].item()
    assert value_leaked == pytest.approx(4.0), (
        "atteso: il vecchio ordine (sbagliato) stima il buco per interpolazione temporale tra il "
        "valore di train immediatamente precedente (3.0) e quello di test successivo (5.0), dando "
        "4.0 — usando quindi informazione dal train per riempire un punto del test. Se questo assert "
        "fallisse, il test stesso andrebbe rivisto"
    )


# --- drop_residual_na ---

def test_drop_residual_na_removes_only_rows_with_na():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, np.nan]})
    out = drop_residual_na(df, cols=["a", "b"])
    assert len(out) == 1
    assert out["a"].tolist() == [1.0]
