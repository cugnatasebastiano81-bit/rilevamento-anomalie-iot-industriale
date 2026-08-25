"""Unit test per src/reporting.py (Fase 6). Dati sintetici piccoli, nessun I/O reale."""
import numpy as np
import pandas as pd
import pytest

from src.reporting import (
    PAIR_PLOT_SIGNALS,
    alert_heatmap_table,
    distances_by_flag,
    pair_plot_frame,
    split_background_and_alerts,
    stratified_sample,
)


def make_frame(n_per_asset=6, assets=("A", "B"), start="2025-02-08 00:00", freq="1min", tz="UTC"):
    """Piccolo frame sintetico con chiave (asset_id, timestamp), regime, cluster e flag di alert."""
    rows = []
    for asset in assets:
        stamps = pd.date_range(start=start, periods=n_per_asset, freq=freq, tz=tz)
        for i, ts in enumerate(stamps):
            rows.append(
                {
                    "asset_id": asset,
                    "timestamp": ts,
                    "regime": i % 3,
                    "cluster": i % 2,
                    "is_anomaly_primary": (i % 4 == 0),
                    "dist_centroid": float(i),
                    "rpm": 1000.0 + i,
                    "temp_c": 40.0 + i,
                    "vib_rms": 0.5 + i / 10,
                    "current_a": 12.0 + i / 2,
                }
            )
    return pd.DataFrame(rows)


# --- alert_heatmap_table -------------------------------------------------------------------


def test_alert_heatmap_table_counts_and_rate_on_hand_computable_case():
    ts = pd.date_range("2025-02-08 00:00", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "asset_id": ["A", "A", "A", "A"],
            "timestamp": ts,
            "regime": [0, 0, 1, 1],
            "is_anomaly_primary": [True, False, True, True],
        }
    )
    table = alert_heatmap_table(df, freq="1h")

    # due intervalli orari x due regimi = quattro celle
    assert len(table) == 4
    first_hour_r0 = table[(table["bin_start"] == ts[0]) & (table["regime"] == 0)].iloc[0]
    assert first_hour_r0["support"] == 2
    assert first_hour_r0["alerts"] == 1
    assert first_hour_r0["alert_rate"] == pytest.approx(0.5)

    second_hour_r1 = table[(table["bin_start"] == ts[2].floor("1h")) & (table["regime"] == 1)].iloc[0]
    assert second_hour_r1["support"] == 2
    assert second_hour_r1["alerts"] == 2
    assert second_hour_r1["alert_rate"] == pytest.approx(1.0)


def test_alert_heatmap_table_reconciles_with_totals():
    df = make_frame(n_per_asset=30)
    table = alert_heatmap_table(df, freq="10min")

    assert table["support"].sum() == len(df)
    assert table["alerts"].sum() == int(df["is_anomaly_primary"].sum())


def test_alert_heatmap_table_empty_cell_has_undefined_rate_not_zero():
    # un buco di un'ora intera: l'intervallo centrale non ha alcuna riga
    ts = [
        pd.Timestamp("2025-02-08 00:00", tz="UTC"),
        pd.Timestamp("2025-02-08 02:00", tz="UTC"),
    ]
    df = pd.DataFrame(
        {"asset_id": ["A", "A"], "timestamp": ts, "regime": [0, 0], "is_anomaly_primary": [False, False]}
    )
    table = alert_heatmap_table(df, freq="1h")

    empty = table[table["bin_start"] == pd.Timestamp("2025-02-08 01:00", tz="UTC")].iloc[0]
    assert empty["support"] == 0
    assert empty["alerts"] == 0
    assert np.isnan(empty["alert_rate"])

    observed = table[table["bin_start"] == ts[0]].iloc[0]
    assert observed["alert_rate"] == pytest.approx(0.0)  # supporto presente, nessun alert: 0% reale


def test_alert_heatmap_table_declared_group_absent_from_data_is_kept_as_unavailable():
    df = make_frame(n_per_asset=3)
    table = alert_heatmap_table(df, freq="1h", group_values=[0, 1, 2, 3])

    missing = table[table["regime"] == 3]
    assert len(missing) > 0
    assert (missing["support"] == 0).all()
    assert missing["alert_rate"].isna().all()


def test_alert_heatmap_table_rejects_group_value_not_declared():
    df = make_frame(n_per_asset=6)
    with pytest.raises(ValueError, match="non dichiarati"):
        alert_heatmap_table(df, freq="1h", group_values=[0, 1])


def test_alert_heatmap_table_rejects_duplicate_group_values():
    df = make_frame(n_per_asset=6)
    with pytest.raises(ValueError, match="duplicati"):
        alert_heatmap_table(df, freq="1h", group_values=[0, 1, 2, 2])


def test_alert_heatmap_table_single_group_and_single_bin():
    ts = pd.date_range("2025-02-08 00:00", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"asset_id": ["A", "A", "A"], "timestamp": ts, "regime": [0, 0, 0],
         "is_anomaly_primary": [True, False, False]}
    )
    table = alert_heatmap_table(df, freq="1h")

    assert len(table) == 1
    assert table.iloc[0]["support"] == 3
    assert table.iloc[0]["alerts"] == 1


def test_alert_heatmap_table_is_invariant_to_row_order():
    df = make_frame(n_per_asset=20)
    shuffled = df.sample(frac=1, random_state=7)

    pd.testing.assert_frame_equal(alert_heatmap_table(df, freq="5min"), alert_heatmap_table(shuffled, freq="5min"))


def test_alert_heatmap_table_counts_duplicate_timestamps_once_per_row():
    ts = pd.Timestamp("2025-02-08 00:00", tz="UTC")
    df = pd.DataFrame(
        {"asset_id": ["A", "B"], "timestamp": [ts, ts], "regime": [0, 0], "is_anomaly_primary": [True, False]}
    )
    table = alert_heatmap_table(df, freq="1h")

    assert table.iloc[0]["support"] == 2
    assert table.iloc[0]["alerts"] == 1


def test_alert_heatmap_table_rejects_empty_frame():
    df = make_frame(n_per_asset=3).iloc[0:0]
    with pytest.raises(ValueError, match="non può essere vuoto"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_missing_column():
    df = make_frame(n_per_asset=3).drop(columns=["regime"])
    with pytest.raises(ValueError, match="non contiene le colonne"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_naive_timestamps():
    df = make_frame(n_per_asset=3)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_nat_timestamps():
    df = make_frame(n_per_asset=3)
    df.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_non_datetime_timestamp_column():
    df = make_frame(n_per_asset=3)
    df["timestamp"] = df["timestamp"].astype(str)
    with pytest.raises(ValueError, match="datetime"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_non_binary_alert_column():
    df = make_frame(n_per_asset=3)
    df["is_anomaly_primary"] = [2] * len(df)
    with pytest.raises(ValueError, match="diversi da 0/1"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_missing_group_value():
    df = make_frame(n_per_asset=3)
    df["regime"] = df["regime"].astype("float")
    df.loc[0, "regime"] = np.nan
    with pytest.raises(ValueError, match="valori mancanti"):
        alert_heatmap_table(df)


def test_alert_heatmap_table_rejects_non_positive_freq():
    df = make_frame(n_per_asset=3)
    with pytest.raises(ValueError):
        alert_heatmap_table(df, freq="0h")


# --- stratified_sample ---------------------------------------------------------------------


def test_stratified_sample_is_deterministic_for_the_same_seed():
    df = make_frame(n_per_asset=40)
    first = stratified_sample(df, group_col="cluster", size=5, seed=42)
    second = stratified_sample(df, group_col="cluster", size=5, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_stratified_sample_is_invariant_to_row_order():
    df = make_frame(n_per_asset=40)
    shuffled = df.sample(frac=1, random_state=3)

    keys_ordered = stratified_sample(df, group_col="cluster", size=5, seed=42)[["asset_id", "timestamp"]]
    keys_shuffled = stratified_sample(shuffled, group_col="cluster", size=5, seed=42)[["asset_id", "timestamp"]]

    pd.testing.assert_frame_equal(keys_ordered.reset_index(drop=True), keys_shuffled.reset_index(drop=True))


def test_stratified_sample_different_seeds_select_different_rows():
    df = make_frame(n_per_asset=60)
    first = stratified_sample(df, group_col="cluster", size=10, seed=1)[["asset_id", "timestamp"]]
    second = stratified_sample(df, group_col="cluster", size=10, seed=2)[["asset_id", "timestamp"]]

    assert not first.reset_index(drop=True).equals(second.reset_index(drop=True))


def test_stratified_sample_takes_whole_group_when_size_exceeds_population():
    df = make_frame(n_per_asset=6)
    sample = stratified_sample(df, group_col="cluster", size=10_000, seed=42)

    assert len(sample) == len(df)


def test_stratified_sample_fraction_keeps_group_proportions():
    df = make_frame(n_per_asset=20)  # 40 righe, 20 per cluster
    sample = stratified_sample(df, group_col="cluster", fraction=0.5, seed=42)

    counts = sample["cluster"].value_counts().to_dict()
    assert counts == {0: 10, 1: 10}


def test_stratified_sample_fraction_keeps_at_least_one_row_per_group():
    df = make_frame(n_per_asset=4)
    sample = stratified_sample(df, group_col="cluster", fraction=0.01, seed=42)

    assert set(sample["cluster"].unique()) == set(df["cluster"].unique())


def test_stratified_sample_with_a_single_group():
    df = make_frame(n_per_asset=10)
    df["cluster"] = 0
    sample = stratified_sample(df, group_col="cluster", size=4, seed=42)

    assert len(sample) == 4


def test_stratified_sample_does_not_change_group_assignment():
    df = make_frame(n_per_asset=30)
    sample = stratified_sample(df, group_col="cluster", size=8, seed=42)

    merged = sample[["asset_id", "timestamp", "cluster"]].merge(
        df[["asset_id", "timestamp", "cluster"]], on=["asset_id", "timestamp"], suffixes=("_sample", "_source")
    )
    assert (merged["cluster_sample"] == merged["cluster_source"]).all()


def test_stratified_sample_rejects_both_fraction_and_size():
    df = make_frame(n_per_asset=6)
    with pytest.raises(ValueError, match="esattamente una"):
        stratified_sample(df, group_col="cluster", fraction=0.5, size=2)


def test_stratified_sample_rejects_neither_fraction_nor_size():
    df = make_frame(n_per_asset=6)
    with pytest.raises(ValueError, match="esattamente una"):
        stratified_sample(df, group_col="cluster")


@pytest.mark.parametrize("bad_fraction", [0, -0.1, 1.5, float("nan")])
def test_stratified_sample_rejects_invalid_fraction(bad_fraction):
    df = make_frame(n_per_asset=6)
    with pytest.raises(ValueError, match="fraction"):
        stratified_sample(df, group_col="cluster", fraction=bad_fraction)


@pytest.mark.parametrize("bad_size", [0, -3, 2.5, True])
def test_stratified_sample_rejects_invalid_size(bad_size):
    df = make_frame(n_per_asset=6)
    with pytest.raises(ValueError, match="size"):
        stratified_sample(df, group_col="cluster", size=bad_size)


def test_stratified_sample_rejects_duplicate_keys():
    df = pd.concat([make_frame(n_per_asset=4), make_frame(n_per_asset=4)])
    with pytest.raises(ValueError, match="chiavi duplicate"):
        stratified_sample(df, group_col="cluster", size=2)


def test_stratified_sample_rejects_empty_frame():
    df = make_frame(n_per_asset=4).iloc[0:0]
    with pytest.raises(ValueError, match="non può essere vuoto"):
        stratified_sample(df, group_col="cluster", size=2)


def test_stratified_sample_rejects_missing_group_column():
    df = make_frame(n_per_asset=4).drop(columns=["cluster"])
    with pytest.raises(ValueError, match="non contiene le colonne"):
        stratified_sample(df, group_col="cluster", size=2)


# --- split_background_and_alerts -----------------------------------------------------------


def test_split_keeps_every_alert():
    df = make_frame(n_per_asset=40)
    background, alerts = split_background_and_alerts(df, fraction=0.1, seed=42)

    assert len(alerts) == int(df["is_anomaly_primary"].sum())
    assert alerts["is_anomaly_primary"].all()


def test_split_background_never_contains_an_alert():
    df = make_frame(n_per_asset=40)
    background, alerts = split_background_and_alerts(df, fraction=0.5, seed=42)

    assert not background["is_anomaly_primary"].any()
    keys = ["asset_id", "timestamp"]
    overlap = background[keys].merge(alerts[keys], on=keys)
    assert len(overlap) == 0


def test_split_without_any_alert_returns_empty_alerts():
    df = make_frame(n_per_asset=10)
    df["is_anomaly_primary"] = False
    background, alerts = split_background_and_alerts(df, fraction=0.5, seed=42)

    assert len(alerts) == 0
    assert len(background) > 0


def test_split_with_only_alerts_returns_empty_background():
    df = make_frame(n_per_asset=10)
    df["is_anomaly_primary"] = True
    background, alerts = split_background_and_alerts(df, fraction=0.5, seed=42)

    assert len(background) == 0
    assert len(alerts) == len(df)


def test_split_is_deterministic_for_the_same_seed():
    df = make_frame(n_per_asset=40)
    first_bg, first_alerts = split_background_and_alerts(df, fraction=0.2, seed=42)
    second_bg, second_alerts = split_background_and_alerts(df, fraction=0.2, seed=42)

    pd.testing.assert_frame_equal(first_bg, second_bg)
    pd.testing.assert_frame_equal(first_alerts, second_alerts)


def test_split_rejects_non_binary_alert_column():
    df = make_frame(n_per_asset=10)
    df["is_anomaly_primary"] = 5
    with pytest.raises(ValueError, match="diversi da 0/1"):
        split_background_and_alerts(df, fraction=0.5, seed=42)


# --- distances_by_flag ---------------------------------------------------------------------


def test_distances_by_flag_splits_and_counts():
    result = distances_by_flag([1.0, 2.0, 3.0, 4.0], [False, True, False, True])

    assert list(result["normal"]) == [1.0, 3.0]
    assert list(result["anomalous"]) == [2.0, 4.0]
    assert result["n_normal"] == 2
    assert result["n_anomalous"] == 2
    assert result["n_normal"] + result["n_anomalous"] == 4


def test_distances_by_flag_without_anomalies():
    result = distances_by_flag([1.0, 2.0], [False, False])

    assert result["n_anomalous"] == 0
    assert len(result["anomalous"]) == 0


def test_distances_by_flag_rejects_length_mismatch():
    with pytest.raises(ValueError, match="lunghezze diverse"):
        distances_by_flag([1.0, 2.0, 3.0], [True, False])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_distances_by_flag_rejects_non_finite_distances(bad):
    with pytest.raises(ValueError, match="non finiti"):
        distances_by_flag([1.0, bad], [True, False])


def test_distances_by_flag_rejects_negative_distances():
    with pytest.raises(ValueError, match="negativi"):
        distances_by_flag([1.0, -2.0], [True, False])


def test_distances_by_flag_rejects_empty_input():
    with pytest.raises(ValueError, match="non può essere vuoto"):
        distances_by_flag([], [])


def test_distances_by_flag_rejects_two_dimensional_distances():
    with pytest.raises(ValueError, match="1 dimensione"):
        distances_by_flag([[1.0], [2.0]], [True, False])


# --- pair_plot_frame -----------------------------------------------------------------------


def test_pair_plot_frame_returns_keys_features_and_group():
    df = make_frame(n_per_asset=30)
    frame = pair_plot_frame(df, size=5, seed=42)

    assert list(frame.columns) == ["asset_id", "timestamp"] + PAIR_PLOT_SIGNALS + ["cluster"]
    assert len(frame) == 10  # 5 righe per ciascuno dei due cluster


def test_pair_plot_frame_is_deterministic_for_the_same_seed():
    df = make_frame(n_per_asset=30)
    first = pair_plot_frame(df, size=5, seed=42)[["asset_id", "timestamp"]]
    second = pair_plot_frame(df, size=5, seed=42)[["asset_id", "timestamp"]]

    pd.testing.assert_frame_equal(first, second)


def test_pair_plot_frame_takes_whole_group_when_size_exceeds_population():
    df = make_frame(n_per_asset=4)
    frame = pair_plot_frame(df, size=1_000, seed=42)

    assert len(frame) == len(df)


def test_pair_plot_frame_rejects_non_finite_feature():
    df = make_frame(n_per_asset=10)
    df.loc[0, "rpm"] = np.nan
    with pytest.raises(ValueError, match="non finiti"):
        pair_plot_frame(df, size=3, seed=42)


def test_pair_plot_frame_rejects_missing_feature_column():
    df = make_frame(n_per_asset=10).drop(columns=["vib_rms"])
    with pytest.raises(ValueError, match="non contiene le colonne"):
        pair_plot_frame(df, size=3, seed=42)


def test_pair_plot_frame_rejects_empty_feature_list():
    df = make_frame(n_per_asset=10)
    with pytest.raises(ValueError, match="non può essere vuoto"):
        pair_plot_frame(df, feature_cols=[], size=3, seed=42)


def test_pair_plot_frame_rejects_duplicate_feature_columns():
    df = make_frame(n_per_asset=10)
    with pytest.raises(ValueError, match="duplicate"):
        pair_plot_frame(df, feature_cols=["rpm", "rpm"], size=3, seed=42)


# --- regressioni dal gate indipendente: unicita della chiave nella heatmap ---


def test_alert_heatmap_table_rejects_duplicate_keys():
    """Due righe con lo stesso asset e lo stesso istante sono la stessa osservazione contata due
    volte: devono essere rifiutate, non aggregate in silenzio."""
    ts = pd.Timestamp("2025-02-08 00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "timestamp": [ts, ts],
            "regime": [0, 0],
            "is_anomaly_primary": [True, True],
        }
    )
    with pytest.raises(ValueError, match="chiavi duplicate"):
        alert_heatmap_table(df, freq="1h")


def test_alert_heatmap_table_accepts_same_timestamp_across_assets():
    """Asset diversi allo stesso istante restano due osservazioni distinte e valide."""
    ts = pd.Timestamp("2025-02-08 00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "asset_id": ["A", "B"],
            "timestamp": [ts, ts],
            "regime": [0, 0],
            "is_anomaly_primary": [True, False],
        }
    )
    table = alert_heatmap_table(df, freq="1h")

    assert len(table) == 1
    assert table.iloc[0]["support"] == 2
    assert table.iloc[0]["alerts"] == 1


def test_alert_heatmap_table_rejects_missing_key_column():
    """Senza la colonna che identifica l'entita non e verificabile l'unicita: errore esplicito."""
    df = make_frame(n_per_asset=4).drop(columns=["asset_id"])
    with pytest.raises(ValueError, match="non contiene le colonne"):
        alert_heatmap_table(df, freq="1h")
