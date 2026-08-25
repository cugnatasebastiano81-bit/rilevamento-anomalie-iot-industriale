"""Unit test per src/validation.py (Fase 5). Dati sintetici piccoli, nessun I/O reale."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_fscore_support

from src.validation import (
    MIN_RELIABLE_COUNT,
    Z_95,
    alerts_near_fault_windows,
    build_fault_targets,
    classification_metrics,
    classification_metrics_with_intervals,
    confusion_counts,
    event_alert_coverage,
    merge_predictions_and_labels,
    metrics_by_group,
    recall_by_fault_type,
    segment_fault_events,
    select_confusion_cases,
    to_binary_array,
    wilson_interval,
)

# --- to_binary_array ---


def test_to_binary_array_accepts_int_0_1():
    result = to_binary_array(np.array([0, 1, 1, 0]), "x")
    assert result.dtype == bool
    assert list(result) == [False, True, True, False]


def test_to_binary_array_accepts_bool():
    result = to_binary_array(np.array([True, False]), "x")
    assert list(result) == [True, False]


def test_to_binary_array_rejects_nan():
    with pytest.raises(ValueError):
        to_binary_array(pd.Series([0, 1, np.nan]), "x")


def test_to_binary_array_rejects_infinite():
    with pytest.raises(ValueError):
        to_binary_array(np.array([0.0, np.inf]), "x")


def test_to_binary_array_rejects_non_binary_values():
    with pytest.raises(ValueError):
        to_binary_array(np.array([0, 1, 2]), "x")


def test_to_binary_array_rejects_strings():
    with pytest.raises(ValueError):
        to_binary_array(np.array(["0", "1"], dtype=object), "x")


def test_to_binary_array_rejects_2d():
    with pytest.raises(ValueError):
        to_binary_array(np.array([[0, 1], [1, 0]]), "x")


def test_to_binary_array_rejects_empty():
    with pytest.raises(ValueError):
        to_binary_array(np.array([]), "x")


# --- build_fault_targets: costruzione validata dei due target di validazione ---


def test_build_fault_targets_happy_path():
    fault_known, anomaly_partial = build_fault_targets(
        fault_code_true=np.array([0, 1, 0, 3]),
        anomaly_label=np.array([0, 1, 1, 0]),
    )
    assert list(fault_known) == [0, 1, 0, 1]
    assert list(anomaly_partial) == [0, 1, 1, 0]


def test_build_fault_targets_rejects_nan_fault_code():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0.0, np.nan]), np.array([0, 1]))


def test_build_fault_targets_does_not_treat_nan_as_no_fault():
    # un NaN in fault_code_true non deve mai tradursi silenziosamente in "nessun guasto" (0)
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0.0, np.nan]), np.array([0, 0]))


def test_build_fault_targets_rejects_infinite_fault_code():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0.0, np.inf]), np.array([0, 1]))


def test_build_fault_targets_rejects_unexpected_fault_code():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0, 99]), np.array([0, 1]))


def test_build_fault_targets_rejects_non_binary_anomaly_label():
    # anomaly_label=2 non deve mai essere silenziosamente trattato come "non anomalo" (0)
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0, 1]), np.array([0, 2]))


def test_build_fault_targets_rejects_empty():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([]), np.array([]))


def test_build_fault_targets_rejects_2d():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([[0, 1]]), np.array([[0, 1]]))


def test_build_fault_targets_rejects_mismatched_length():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0, 1, 0]), np.array([0, 1]))


def test_build_fault_targets_rejects_non_numeric_fault_code():
    with pytest.raises(ValueError):
        build_fault_targets(np.array(["0", "1"], dtype=object), np.array([0, 1]))


def test_build_fault_targets_rejects_non_integer_fault_code():
    with pytest.raises(ValueError):
        build_fault_targets(np.array([0.0, 1.5]), np.array([0, 1]))


def test_build_fault_targets_custom_admitted_codes():
    fault_known, anomaly_partial = build_fault_targets(
        np.array([0, 7]), np.array([0, 1]), admitted_fault_codes=(0, 7),
    )
    assert list(fault_known) == [0, 1]


# --- confusion_counts / classification_metrics: caso asimmetrico calcolabile a mano ---

Y_TRUE_HAND = np.array([1, 1, 1, 0, 0, 0, 0, 0])
Y_PRED_HAND = np.array([1, 0, 1, 1, 0, 0, 0, 0])
# TP=2 (idx0,2), FN=1 (idx1), FP=1 (idx3), TN=4 (idx4..7)


def test_confusion_counts_hand_example():
    counts = confusion_counts(Y_TRUE_HAND, Y_PRED_HAND)
    assert counts == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}


def test_confusion_counts_matches_sklearn_labels_0_1():
    from sklearn.metrics import confusion_matrix

    counts = confusion_counts(Y_TRUE_HAND, Y_PRED_HAND)
    tn, fp, fn, tp = confusion_matrix(Y_TRUE_HAND, Y_PRED_HAND, labels=[0, 1]).ravel()
    assert counts == {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def test_confusion_counts_rejects_mismatched_length():
    with pytest.raises(ValueError):
        confusion_counts(np.array([0, 1, 1]), np.array([0, 1]))


def test_classification_metrics_hand_example_matches_sklearn():
    metrics = classification_metrics(Y_TRUE_HAND, Y_PRED_HAND)
    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_TRUE_HAND, Y_PRED_HAND, labels=[0, 1], zero_division=0
    )
    assert metrics["precision"] == pytest.approx(precision[1], abs=1e-12)
    assert metrics["recall"] == pytest.approx(recall[1], abs=1e-12)
    assert metrics["f1"] == pytest.approx(f1[1], abs=1e-12)
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-12)
    assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-12)
    assert metrics["support"] == 8
    assert metrics["positives_real"] == 3
    assert metrics["positives_pred"] == 3
    assert metrics["prevalence"] == pytest.approx(3 / 8, abs=1e-12)
    assert metrics["alert_rate"] == pytest.approx(3 / 8, abs=1e-12)
    assert metrics["precision_undefined"] is False
    assert metrics["recall_undefined"] is False
    assert metrics["f1_undefined"] is False


def test_classification_metrics_no_real_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["recall_undefined"] is True
    assert metrics["recall"] == 0.0
    assert metrics["precision_undefined"] is False
    assert metrics["precision"] == 0.0  # TP=0, FP=1: legittimo zero, non un denominatore nullo
    assert metrics["f1_undefined"] is True
    assert metrics["f1"] == 0.0


def test_classification_metrics_no_predicted_positives():
    y_true = np.array([0, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["precision_undefined"] is True
    assert metrics["precision"] == 0.0
    assert metrics["recall_undefined"] is False
    assert metrics["recall"] == 0.0  # TP=0, positives_real=1: legittimo zero
    assert metrics["f1_undefined"] is True


def test_classification_metrics_no_positives_in_either():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["precision_undefined"] is True
    assert metrics["recall_undefined"] is True
    assert metrics["f1_undefined"] is True
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.0


def test_classification_metrics_all_rows_positive():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["precision"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-12)
    assert metrics["precision_undefined"] is False
    assert metrics["f1_undefined"] is False


def test_classification_metrics_single_row():
    metrics = classification_metrics(np.array([1]), np.array([1]))
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == pytest.approx(1.0, abs=1e-12)


def test_classification_metrics_empty_subgroup_raises():
    with pytest.raises(ValueError):
        classification_metrics(np.array([]), np.array([]))


# --- wilson_interval ---


def _wilson_reference(successes, n, z=Z_95):
    """Formula di Wilson riarrangiata in modo indipendente (usa i conteggi interi, non p_hat
    calcolato separatamente), per una verifica incrociata della stessa quantità matematica.
    """
    center = (successes + z**2 / 2) / (n + z**2)
    margin = z / (n + z**2) * math.sqrt(successes * (n - successes) / n + z**2 / 4)
    return max(0.0, center - margin), min(1.0, center + margin)


@pytest.mark.parametrize("successes,n", [(50, 100), (1, 100), (99, 100), (1000, 1500), (7, 7)])
def test_wilson_interval_matches_independent_reference(successes, n):
    lo, hi = wilson_interval(successes, n)
    ref_lo, ref_hi = _wilson_reference(successes, n)
    assert lo == pytest.approx(ref_lo, abs=1e-12)
    assert hi == pytest.approx(ref_hi, abs=1e-12)


def test_wilson_interval_zero_successes():
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi > 0.0


def test_wilson_interval_n_zero_returns_none():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(0, -1)


def test_classification_metrics_with_intervals_no_f1_interval():
    metrics = classification_metrics_with_intervals(Y_TRUE_HAND, Y_PRED_HAND)
    assert "alert_rate_ci95" in metrics
    assert "precision_ci95" in metrics
    assert "recall_ci95" in metrics
    assert "f1_ci95" not in metrics


def test_classification_metrics_with_intervals_none_when_denominator_zero():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = classification_metrics_with_intervals(y_true, y_pred)
    assert metrics["precision_ci95"] is None
    assert metrics["recall_ci95"] is None
    assert metrics["alert_rate_ci95"] is not None  # support > 0


# --- merge_predictions_and_labels: allineamento per chiave, non posizionale ---


def _toy_predictions():
    return pd.DataFrame({
        "asset_id": ["A", "A", "B"],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"], utc=True),
        "y_pred": [1, 0, 1],
    })


def _toy_labels():
    return pd.DataFrame({
        "asset_id": ["A", "A", "B"],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"], utc=True),
        "y_true": [1, 0, 0],
    })


def test_merge_predictions_and_labels_happy_path():
    merged = merge_predictions_and_labels(_toy_predictions(), _toy_labels())
    assert len(merged) == 3
    assert list(merged["y_pred"]) == [1, 0, 1]
    assert list(merged["y_true"]) == [1, 0, 0]


def test_merge_predictions_and_labels_realigns_by_key_not_position():
    predictions = _toy_predictions()
    labels = _toy_labels().sample(frac=1, random_state=0).reset_index(drop=True)  # righe mescolate
    merged = merge_predictions_and_labels(predictions, labels)
    merged = merged.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    expected = merge_predictions_and_labels(predictions, _toy_labels())
    expected = expected.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, expected)


def test_merge_predictions_and_labels_rejects_duplicate_keys():
    predictions = _toy_predictions()
    predictions.loc[2, "timestamp"] = predictions.loc[0, "timestamp"]
    predictions.loc[2, "asset_id"] = predictions.loc[0, "asset_id"]
    with pytest.raises(ValueError):
        merge_predictions_and_labels(predictions, _toy_labels())


def test_merge_predictions_and_labels_rejects_cardinality_mismatch():
    labels = _toy_labels().iloc[:2]
    with pytest.raises(ValueError):
        merge_predictions_and_labels(_toy_predictions(), labels)


def test_merge_predictions_and_labels_rejects_mismatched_keys():
    labels = _toy_labels()
    labels.loc[0, "asset_id"] = "Z"
    with pytest.raises(ValueError):
        merge_predictions_and_labels(_toy_predictions(), labels)


def test_merge_predictions_and_labels_rejects_missing_key_column():
    labels = _toy_labels().drop(columns=["asset_id"])
    with pytest.raises(ValueError):
        merge_predictions_and_labels(_toy_predictions(), labels)


def test_merge_predictions_and_labels_rejects_naive_timestamps():
    predictions = _toy_predictions()
    predictions["timestamp"] = predictions["timestamp"].dt.tz_localize(None)
    labels = _toy_labels()
    labels["timestamp"] = labels["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        merge_predictions_and_labels(predictions, labels)


def test_merge_predictions_and_labels_rejects_naive_predictions_only():
    predictions = _toy_predictions()
    predictions["timestamp"] = predictions["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        merge_predictions_and_labels(predictions, _toy_labels())


def test_merge_predictions_and_labels_rejects_nat_timestamps():
    predictions = _toy_predictions()
    predictions.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError):
        merge_predictions_and_labels(predictions, _toy_labels())


def test_merge_predictions_and_labels_rejects_non_datetime_timestamp_column():
    predictions = _toy_predictions()
    predictions["timestamp"] = ["2025-01-01", "2025-01-02", "2025-01-01"]
    with pytest.raises(ValueError):
        merge_predictions_and_labels(predictions, _toy_labels())


def test_merge_predictions_and_labels_rejects_mismatched_timezones():
    predictions = _toy_predictions()
    labels = _toy_labels()
    labels["timestamp"] = labels["timestamp"].dt.tz_convert("Europe/Rome")
    with pytest.raises(ValueError):
        merge_predictions_and_labels(predictions, labels)


# --- metrics_by_group: partizione completa, alert rate identico tra target ---


def _subgroup_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "cluster": rng.integers(0, 3, size=n),
        "y_true_a": rng.integers(0, 2, size=n),
        "y_true_b": rng.integers(0, 2, size=n),
        "y_pred": rng.integers(0, 2, size=n),
    })


def test_metrics_by_group_sums_to_global_confusion():
    df = _subgroup_df()
    grouped = metrics_by_group(df, "cluster", "y_true_a", "y_pred")
    global_counts = confusion_counts(df["y_true_a"], df["y_pred"])
    assert grouped["tn"].sum() == global_counts["tn"]
    assert grouped["fp"].sum() == global_counts["fp"]
    assert grouped["fn"].sum() == global_counts["fn"]
    assert grouped["tp"].sum() == global_counts["tp"]
    assert grouped["n"].sum() == len(df)


def test_metrics_by_group_alert_rate_same_across_targets():
    df = _subgroup_df()
    grouped_a = metrics_by_group(df, "cluster", "y_true_a", "y_pred")
    grouped_b = metrics_by_group(df, "cluster", "y_true_b", "y_pred")
    pd.testing.assert_series_equal(
        grouped_a["alert_rate"].sort_index(), grouped_b["alert_rate"].sort_index()
    )


def test_metrics_by_group_fragile_flag_boundary():
    # 19 positivi reali E predetti (< MIN_RELIABLE_COUNT=20): fragile. 20 di entrambi: non fragile.
    y_true_fragile = np.array([1] * 19 + [0] * 81)
    y_pred_fragile = np.array([1] * 19 + [0] * 81)
    y_true_ok = np.array([1] * 20 + [0] * 80)
    y_pred_ok = np.array([1] * 20 + [0] * 80)
    df = pd.DataFrame({
        "group": ["fragile"] * 100 + ["ok"] * 100,
        "y_true": np.concatenate([y_true_fragile, y_true_ok]),
        "y_pred": np.concatenate([y_pred_fragile, y_pred_ok]),
    })
    grouped = metrics_by_group(df, "group", "y_true", "y_pred", min_count=MIN_RELIABLE_COUNT)
    assert bool(grouped.loc["fragile", "fragile"]) is True
    assert bool(grouped.loc["ok", "fragile"]) is False


def test_metrics_by_group_rejects_missing_column():
    df = _subgroup_df()
    with pytest.raises(ValueError):
        metrics_by_group(df, "not_a_column", "y_true_a", "y_pred")


# --- recall_by_fault_type ---


def test_recall_by_fault_type_hand_example():
    df = pd.DataFrame({
        "fault_type_true": ["bearing_wear", "bearing_wear", "overheating", "overheating", "overheating"],
        "y_pred": [1, 0, 1, 1, 0],
    })
    result = recall_by_fault_type(df, "fault_type_true", "y_pred", min_count=2)
    assert result.loc["bearing_wear", "recall"] == pytest.approx(0.5, abs=1e-12)
    assert result.loc["overheating", "recall"] == pytest.approx(2 / 3, abs=1e-12)
    assert result.loc["bearing_wear", "n_detected"] == 1
    # con min_count=2 e n=2, 2 < 2 è False: non fragile
    assert bool(result.loc["bearing_wear", "fragile"]) is False
    assert bool(result.loc["overheating", "fragile"]) is False  # n=3 >= min_count=2


def test_recall_by_fault_type_rejects_empty():
    df = pd.DataFrame({"fault_type_true": [], "y_pred": []})
    with pytest.raises(ValueError):
        recall_by_fault_type(df, "fault_type_true", "y_pred")


def test_recall_by_fault_type_rejects_missing_column():
    df = pd.DataFrame({"y_pred": [1, 0]})
    with pytest.raises(ValueError):
        recall_by_fault_type(df, "fault_type_true", "y_pred")


# --- select_confusion_cases: ordine deterministico ---


def test_select_confusion_cases_orders_by_ratio_then_key():
    df = pd.DataFrame({
        "asset_id": ["B", "A", "C", "A"],
        "timestamp": [1, 1, 1, 2],
        "y_true": [0, 0, 0, 0],
        "y_pred": [1, 1, 1, 1],
        "ratio": [2.0, 3.0, 3.0, 1.0],
    })
    result = select_confusion_cases(
        df, "y_true", "y_pred", "ratio", ["asset_id", "timestamp"], category="fp", n=10
    )
    # ratio decrescente: 3.0 (A o C, tie-break su asset_id: A prima di C), poi 2.0 (B), poi 1.0 (A)
    assert list(result["asset_id"]) == ["A", "C", "B", "A"]
    assert list(result["ratio"]) == [3.0, 3.0, 2.0, 1.0]


def test_select_confusion_cases_respects_n_limit():
    df = pd.DataFrame({
        "asset_id": ["A"] * 15,
        "timestamp": range(15),
        "y_true": [0] * 15,
        "y_pred": [1] * 15,
        "ratio": list(range(15)),
    })
    result = select_confusion_cases(df, "y_true", "y_pred", "ratio", ["asset_id", "timestamp"], "fp", n=10)
    assert len(result) == 10


def test_select_confusion_cases_fn_category():
    df = pd.DataFrame({
        "asset_id": ["A", "B"],
        "timestamp": [1, 2],
        "y_true": [1, 1],
        "y_pred": [0, 1],
        "ratio": [0.5, 0.9],
    })
    result = select_confusion_cases(df, "y_true", "y_pred", "ratio", ["asset_id", "timestamp"], "fn")
    assert list(result["asset_id"]) == ["A"]


def test_select_confusion_cases_rejects_invalid_category():
    df = pd.DataFrame({
        "asset_id": ["A"], "timestamp": [1], "y_true": [0], "y_pred": [1], "ratio": [1.0],
    })
    with pytest.raises(ValueError):
        select_confusion_cases(df, "y_true", "y_pred", "ratio", ["asset_id"], "wrong")


# --- segment_fault_events: nessun attraversamento di confine tra asset ---


def _events_input(asset_id, fault_flags):
    n = len(fault_flags)
    return pd.DataFrame({
        "asset_id": [asset_id] * n,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
        "fault_code_true": fault_flags,
    })


def test_segment_fault_events_single_run_in_middle():
    df = _events_input("A", [0, 0, 3, 3, 3, 0, 0])
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["n_rows"] == 3
    assert ev["truncated_start"] is np.False_ or ev["truncated_start"] is False
    assert ev["truncated_end"] is np.False_ or ev["truncated_end"] is False


def test_segment_fault_events_truncated_at_boundaries():
    df = _events_input("A", [5, 5, 0, 0, 0, 7])
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 2
    first, second = events.iloc[0], events.iloc[1]
    assert bool(first["truncated_start"]) is True
    assert bool(first["truncated_end"]) is False
    assert bool(second["truncated_start"]) is False
    assert bool(second["truncated_end"]) is True


def test_segment_fault_events_no_crossing_between_assets():
    df_a = _events_input("A", [0, 0, 0, 1])  # evento troncato alla fine dell'asset A
    df_b = _events_input("B", [1, 0, 0, 0])  # evento troncato all'inizio dell'asset B
    df = pd.concat([df_a, df_b], ignore_index=True)
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 2
    assert set(events["asset_id"]) == {"A", "B"}
    for _, ev in events.iterrows():
        assert ev["n_rows"] == 1  # nessuna fusione tra la coda di A e la testa di B


def test_segment_fault_events_no_fault_returns_empty():
    df = _events_input("A", [0, 0, 0])
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 0
    assert list(events.columns) == [
        "asset_id", "start", "end", "n_rows", "truncated_start", "truncated_end"
    ]


def test_segment_fault_events_rejects_missing_columns():
    df = pd.DataFrame({"asset_id": ["A"], "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")]})
    with pytest.raises(ValueError):
        segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")


def test_segment_fault_events_consecutive_rows_stay_one_event():
    df = _events_input("A", [3, 3, 3])  # delta di 1 minuto tra righe, pari a expected_freq
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 1
    assert events.iloc[0]["n_rows"] == 3


def test_segment_fault_events_splits_on_temporal_gap():
    timestamps = pd.to_datetime([
        "2025-01-01 00:00:00", "2025-01-01 00:01:00",  # primo evento, righe contigue
        "2025-01-01 00:10:00", "2025-01-01 00:11:00",  # secondo evento, dopo un gap non osservato
    ], utc=True)
    df = pd.DataFrame({
        "asset_id": ["A"] * 4,
        "timestamp": timestamps,
        "fault_code_true": [3, 3, 3, 3],
    })
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 2
    assert list(events["n_rows"]) == [2, 2]
    assert events.iloc[0]["end"] == timestamps[1]
    assert events.iloc[1]["start"] == timestamps[2]
    # entrambi troncati alla fine/inizio dell'osservazione dell'asset, non uno solo dell'evento fuso
    assert bool(events.iloc[0]["truncated_start"]) is True
    assert bool(events.iloc[0]["truncated_end"]) is False
    assert bool(events.iloc[1]["truncated_start"]) is False
    assert bool(events.iloc[1]["truncated_end"]) is True


def test_segment_fault_events_gap_across_three_runs():
    timestamps = pd.to_datetime([
        "2025-01-01 00:00:00",
        "2025-01-01 00:05:00",
        "2025-01-01 00:10:00",
    ], utc=True)
    df = pd.DataFrame({
        "asset_id": ["A"] * 3,
        "timestamp": timestamps,
        "fault_code_true": [3, 3, 3],
    })
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    assert len(events) == 3
    assert list(events["n_rows"]) == [1, 1, 1]


def test_segment_fault_events_custom_expected_freq():
    # frequenza dichiarata di 5 minuti: un delta di 5 minuti resta contiguo con questo parametro
    timestamps = pd.to_datetime([
        "2025-01-01 00:00:00", "2025-01-01 00:05:00", "2025-01-01 00:10:00",
    ], utc=True)
    df = pd.DataFrame({
        "asset_id": ["A"] * 3,
        "timestamp": timestamps,
        "fault_code_true": [3, 3, 3],
    })
    events = segment_fault_events(
        df, "asset_id", "timestamp", "fault_code_true", expected_freq=pd.Timedelta(minutes=5)
    )
    assert len(events) == 1
    assert events.iloc[0]["n_rows"] == 3


def test_segment_fault_events_rejects_non_positive_expected_freq():
    df = _events_input("A", [3, 3, 3])
    with pytest.raises(ValueError):
        segment_fault_events(
            df, "asset_id", "timestamp", "fault_code_true", expected_freq=pd.Timedelta(0)
        )


# --- event_alert_coverage ---


def test_event_alert_coverage_hand_example():
    df = _events_input("A", [3, 3, 3, 3, 0])
    df["is_anomaly"] = [False, False, True, True, False]
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    coverage = event_alert_coverage(events, df, "asset_id", "timestamp", "is_anomaly")
    row = coverage.iloc[0]
    assert bool(row["has_alert"]) is True
    assert row["delay_minutes"] == pytest.approx(2.0, abs=1e-12)  # alert 2 minuti dopo l'inizio
    assert row["coverage"] == pytest.approx(0.5, abs=1e-12)  # 2 alert su 4 righe dell'evento


def test_event_alert_coverage_no_alert_in_window():
    df = _events_input("A", [3, 3, 0])
    df["is_anomaly"] = [False, False, False]
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    coverage = event_alert_coverage(events, df, "asset_id", "timestamp", "is_anomaly")
    row = coverage.iloc[0]
    assert bool(row["has_alert"]) is False
    assert math.isnan(row["delay_minutes"])
    assert row["coverage"] == 0.0


# --- alerts_near_fault_windows ---


def test_alerts_near_fault_windows_hand_example():
    df = _events_input("A", [0, 0, 3, 3, 0, 0, 0, 0, 0, 0])
    # evento: righe 2-3 (minuti 02:00-02:01). Alert a minuto 5 (7 min dopo la fine, dentro +-15) e
    # minuto 20 fuori dal df -> uso solo alert dentro il range disponibile, uno vicino uno lontano.
    df["is_anomaly"] = False
    df.loc[df.index[5], "is_anomaly"] = True  # 3 minuti dopo la fine dell'evento: vicino
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    result = alerts_near_fault_windows(df, "asset_id", "timestamp", "is_anomaly", events, window_minutes=15)
    assert result["n_alerts"] == 1
    assert result["n_near"] == 1
    assert result["fraction_near"] == pytest.approx(1.0, abs=1e-12)


def test_alerts_near_fault_windows_far_alert_not_counted():
    n = 40
    df = _events_input("A", [0, 0] + [3, 3] + [0] * (n - 4))
    df["is_anomaly"] = False
    df.loc[df.index[-1], "is_anomaly"] = True  # ultimo minuto, ben oltre 15 min dalla fine dell'evento
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    result = alerts_near_fault_windows(df, "asset_id", "timestamp", "is_anomaly", events, window_minutes=15)
    assert result["n_alerts"] == 1
    assert result["n_near"] == 0
    assert result["fraction_near"] == 0.0


def test_alerts_near_fault_windows_no_alerts():
    df = _events_input("A", [0, 3, 0])
    df["is_anomaly"] = False
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    result = alerts_near_fault_windows(df, "asset_id", "timestamp", "is_anomaly", events, window_minutes=15)
    assert result["n_alerts"] == 0
    assert result["n_near"] == 0
    assert math.isnan(result["fraction_near"])


def test_alerts_near_fault_windows_asset_without_events_ignored():
    df_a = _events_input("A", [3, 3, 0])  # con evento
    df_b = _events_input("B", [0, 0, 0])  # senza evento, ma con un alert
    df = pd.concat([df_a, df_b], ignore_index=True)
    df["is_anomaly"] = False
    df.loc[df["asset_id"] == "B", "is_anomaly"] = [True, False, False]
    events = segment_fault_events(df, "asset_id", "timestamp", "fault_code_true")
    result = alerts_near_fault_windows(df, "asset_id", "timestamp", "is_anomaly", events, window_minutes=15)
    assert result["n_alerts"] == 1
    assert result["n_near"] == 0  # l'alert è sull'asset B, che non ha eventi


# --- sentinella anti-tuning: le label non toccano mai artefatti operativi ---


def test_labels_never_mutate_operational_arrays():
    distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    thresholds = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
    flags = distances > thresholds
    distances_before, thresholds_before, flags_before = distances.copy(), thresholds.copy(), flags.copy()

    rng = np.random.default_rng(0)
    for _ in range(5):
        permuted_labels = rng.integers(0, 2, size=len(flags))
        classification_metrics(permuted_labels, flags.astype(int))

    assert np.array_equal(distances, distances_before)
    assert np.array_equal(thresholds, thresholds_before)
    assert np.array_equal(flags, flags_before)


def test_reordering_predictions_and_labels_together_preserves_metrics():
    predictions = _toy_predictions()
    labels = _toy_labels()
    merged = merge_predictions_and_labels(predictions, labels)
    metrics_original = classification_metrics(merged["y_true"], merged["y_pred"])

    shuffled_predictions = predictions.sample(frac=1, random_state=1).reset_index(drop=True)
    shuffled_labels = labels.sample(frac=1, random_state=2).reset_index(drop=True)
    merged_shuffled = merge_predictions_and_labels(shuffled_predictions, shuffled_labels)
    metrics_shuffled = classification_metrics(merged_shuffled["y_true"], merged_shuffled["y_pred"])

    assert metrics_original == metrics_shuffled
