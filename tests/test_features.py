"""Unit test per src/features.py (Fase 2). Dati sintetici piccoli, nessun I/O reale."""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_regime_onehot,
    add_rolling_and_diff,
    derived_feature_cols,
    fit_pca,
    fit_scaler,
)


def _asset_series(asset_id, n, start="2025-02-01", freq="1min", value_fn=lambda i: float(i)):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "asset_id": asset_id,
        "timestamp": ts,
        "signal": [value_fn(i) for i in range(n)],
    })


# --- add_regime_onehot ---

def test_add_regime_onehot_creates_expected_columns():
    df = pd.DataFrame({"regime": [0, 1, 2, 1]})
    out = add_regime_onehot(df, categories=[0, 1, 2])
    assert list(out.columns[-3:]) == ["regime_0", "regime_1", "regime_2"]
    assert out["regime_1"].tolist() == [False, True, False, True]


def test_add_regime_onehot_missing_category_in_data_is_all_false():
    # categoria 2 dichiarata (es. osservata nel train) ma assente in questa porzione di dati (es. test)
    df = pd.DataFrame({"regime": [0, 1, 0]})
    out = add_regime_onehot(df, categories=[0, 1, 2])
    assert "regime_2" in out.columns
    assert not out["regime_2"].any()


def test_add_regime_onehot_unknown_value_raises():
    # valore fuori contratto (mai osservato nel train, non tra le categorie dichiarate): deve
    # fallire esplicitamente, non diventare silenziosamente un vettore one-hot tutto a zero.
    df = pd.DataFrame({"regime": [0, 1, 99]})
    with pytest.raises(ValueError, match="non tra le categorie ammesse"):
        add_regime_onehot(df, categories=[0, 1, 2])


def test_add_regime_onehot_missing_value_raises():
    df = pd.DataFrame({"regime": [0, np.nan, 1]})
    with pytest.raises(ValueError, match="valori mancanti"):
        add_regime_onehot(df, categories=[0, 1, 2])


def test_add_regime_onehot_train_test_dummy_columns_are_consistent():
    # stesse colonne dummy, nello stesso ordine, per due porzioni di dati con regimi osservati diversi
    train = pd.DataFrame({"regime": [0, 1, 2]})
    test = pd.DataFrame({"regime": [0, 0, 1]})  # regime 2 assente in questa porzione
    categories = sorted(train["regime"].unique())
    out_train = add_regime_onehot(train, categories)
    out_test = add_regime_onehot(test, categories)
    dummy_cols = [f"regime_{c}" for c in categories]
    assert [c for c in out_train.columns if c.startswith("regime_")] == dummy_cols
    assert [c for c in out_test.columns if c.startswith("regime_")] == dummy_cols


# --- add_rolling_and_diff ---

def test_add_rolling_and_diff_no_cross_asset_contamination():
    a1 = _asset_series(1, 20, value_fn=lambda i: 10.0)  # costante, media mobile sempre 10
    a2 = _asset_series(2, 20, value_fn=lambda i: 1000.0)  # costante, molto diversa
    df = pd.concat([a1, a2], ignore_index=True)

    out = add_rolling_and_diff(df, machine_cols=["signal"], window="15min", min_periods=15)
    a1_means = out.loc[out["asset_id"] == 1, "signal_roll_mean"].dropna()
    assert (a1_means == 10.0).all(), "la media mobile dell'asset 1 non deve risentire dei valori dell'asset 2"


def test_add_rolling_and_diff_min_periods_enforced():
    a1 = _asset_series(1, 20, value_fn=lambda i: float(i))
    out = add_rolling_and_diff(a1, machine_cols=["signal"], window="15min", min_periods=15)
    out = out.sort_values("timestamp").reset_index(drop=True)
    # con finestra "15min" a passo 1 minuto, i primi 14 campioni non hanno 15 osservazioni piene
    assert out["signal_roll_mean"].iloc[:14].isna().all()
    assert out["signal_roll_mean"].iloc[14:].notna().all()


def test_add_rolling_and_diff_diff_is_previous_row_delta():
    a1 = _asset_series(1, 5, value_fn=lambda i: float(i) ** 2)  # 0,1,4,9,16
    out = add_rolling_and_diff(a1, machine_cols=["signal"], window="15min", min_periods=1)
    out = out.sort_values("timestamp").reset_index(drop=True)
    assert out["signal_diff"].iloc[0] is None or pd.isna(out["signal_diff"].iloc[0])
    assert out["signal_diff"].iloc[1:].tolist() == pytest.approx([1.0, 3.0, 5.0, 7.0])


# --- derived_feature_cols ---

def test_derived_feature_cols_names():
    cols = derived_feature_cols(machine_cols=["rpm"])
    assert cols == ["rpm_roll_mean", "rpm_roll_std", "rpm_diff"]


# --- fit_scaler (anti-leakage) ---

def test_fit_scaler_uses_only_train_data():
    df_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({"x": [100.0, 200.0, 300.0]})  # scala molto diversa

    scaler = fit_scaler(df_train, feature_cols=["x"])
    scaler_on_both = fit_scaler(pd.concat([df_train, df_test]), feature_cols=["x"])

    assert scaler.mean_[0] == pytest.approx(2.0)  # media di [1,2,3], non influenzata dal test
    assert scaler.mean_[0] != pytest.approx(scaler_on_both.mean_[0])


# --- fit_pca ---

def test_fit_pca_returns_requested_number_of_components():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 5))
    pca = fit_pca(X, n_components=3)
    assert pca.n_components_ == 3
    assert pca.explained_variance_ratio_.shape == (3,)


def test_fit_pca_explained_variance_is_decreasing():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 5))
    pca = fit_pca(X, n_components=5)
    ratios = pca.explained_variance_ratio_
    assert all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))
