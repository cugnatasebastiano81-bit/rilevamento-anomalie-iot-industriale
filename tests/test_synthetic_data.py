import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import normalized_mutual_info_score

from src.clustering import fit_final_kmeans
from src.features import (
    CONTEXT_NUM_COLS,
    add_regime_onehot,
    add_rolling_and_diff,
    derived_feature_cols,
    fit_pca,
    fit_scaler,
    regime_onehot_cols,
)
from src.synthetic_data import (
    DATASET_COLUMNS,
    FAULT_TYPES,
    SENSOR_COLUMNS,
    SyntheticIoTConfig,
    generate_synthetic_iot_data,
    write_synthetic_iot_csv,
)


def small_config(**overrides):
    values = {
        "days": 2,
        "n_assets": 4,
        "seed": 1234,
        "missing_rate": 0.0038,
        "fault_events_per_asset": 4,
    }
    values.update(overrides)
    return SyntheticIoTConfig(**values)


def test_default_contract_has_expected_public_volume():
    config = SyntheticIoTConfig()
    assert config.periods_per_asset == 14_400
    assert config.expected_rows == 230_400


def test_generator_is_deterministic_for_same_seed():
    first = generate_synthetic_iot_data(small_config())
    second = generate_synthetic_iot_data(small_config())
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_sensor_values():
    first = generate_synthetic_iot_data(small_config(seed=10))
    second = generate_synthetic_iot_data(small_config(seed=11))
    assert not np.array_equal(
        first[SENSOR_COLUMNS].fillna(-1).to_numpy(),
        second[SENSOR_COLUMNS].fillna(-1).to_numpy(),
    )


def test_schema_keys_and_frequency_are_exact():
    config = small_config()
    frame = generate_synthetic_iot_data(config)

    assert list(frame.columns) == DATASET_COLUMNS
    assert len(frame) == config.expected_rows
    assert frame["asset_id"].nunique() == config.n_assets
    assert not frame.duplicated(["asset_id", "timestamp"]).any()
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"

    deltas = frame.groupby("asset_id")["timestamp"].apply(lambda values: values.diff().dropna().unique())
    assert all(list(values) == [pd.Timedelta(minutes=1)] for values in deltas)


def test_missing_values_are_limited_to_sensor_columns_and_match_rate():
    config = small_config()
    frame = generate_synthetic_iot_data(config)
    non_sensor = [column for column in frame.columns if column not in SENSOR_COLUMNS]

    assert not frame[non_sensor].isna().any().any()
    expected_missing = round(len(frame) * len(SENSOR_COLUMNS) * config.missing_rate)
    assert int(frame[SENSOR_COLUMNS].isna().sum().sum()) == expected_missing


def test_fault_contract_and_partial_label_are_valid():
    frame = generate_synthetic_iot_data(small_config())

    assert set(frame["fault_code_true"].unique()) == set(FAULT_TYPES)
    assert set(frame["fault_type_true"].unique()) == set(FAULT_TYPES.values())
    assert set(frame["anomaly_label"].unique()) <= {0, 1}
    assert not ((frame["fault_code_true"] == 0) & (frame["anomaly_label"] == 1)).any()
    for code, name in FAULT_TYPES.items():
        assert (frame.loc[frame["fault_code_true"] == code, "fault_type_true"] == name).all()


def test_regimes_have_equal_daily_support_per_asset():
    frame = generate_synthetic_iot_data(small_config(missing_rate=0.0))
    counts = frame.groupby(["asset_id", "regime"]).size().unstack(fill_value=0)

    assert list(counts.columns) == [0, 1, 2]
    assert (counts.nunique(axis=1) == 1).all()


def test_physical_bounds_hold_for_observed_values():
    frame = generate_synthetic_iot_data(small_config())

    for column in ["rpm", "current_a", "pressure_bar", "flow_lpm", "vib_rms"]:
        assert frame[column].dropna().ge(0).all()
    for column in ["load_pct", "humidity_pct"]:
        assert frame[column].dropna().between(0, 100).all()


def test_fault_injection_has_expected_directional_effects():
    frame = generate_synthetic_iot_data(small_config(missing_rate=0.0))
    normal = frame[frame["fault_code_true"] == 0]

    assert frame.loc[frame["fault_code_true"] == 1, "vib_rms"].median() > normal["vib_rms"].median()
    assert frame.loc[frame["fault_code_true"] == 3, "temp_c"].median() > normal["temp_c"].median()
    assert frame.loc[frame["fault_code_true"] == 4, "vib_kurtosis"].median() > normal["vib_kurtosis"].median()


def test_three_operating_regimes_remain_distinct_in_project_pipeline():
    """K=3 deve rappresentare i tre regimi, non isolare un piccolo cluster di fault."""

    config = small_config(n_assets=16, missing_rate=0.0, fault_events_per_asset=1)
    frame = generate_synthetic_iot_data(config)
    categories = [0, 1, 2]
    featured = add_regime_onehot(frame, categories)
    featured = add_rolling_and_diff(featured)
    feature_columns = (
        regime_onehot_cols(categories)
        + CONTEXT_NUM_COLS
        + derived_feature_cols()
    )
    featured = featured.dropna(subset=feature_columns)

    scaler = fit_scaler(featured, feature_columns)
    scaled = scaler.transform(featured[feature_columns])
    pca = fit_pca(scaled, n_components=10, random_state=42)
    reduced = pca.transform(scaled)
    labels = fit_final_kmeans(
        reduced,
        n_clusters=3,
        random_state=42,
        n_init=10,
    ).labels_

    cluster_shares = pd.Series(labels).value_counts(normalize=True)
    assert cluster_shares.min() >= 0.20
    assert normalized_mutual_info_score(featured["regime"], labels) >= 0.80


def test_csv_bytes_are_reproducible(tmp_path):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    config = small_config()

    write_synthetic_iot_csv(first_path, config)
    write_synthetic_iot_csv(second_path, config)

    first_digest = hashlib.sha256(first_path.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second_path.read_bytes()).hexdigest()
    assert first_digest == second_digest
    assert first_path.read_bytes() == second_path.read_bytes()


def test_cli_runs_from_repository_root(tmp_path):
    output = tmp_path / "generated.csv"
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_synthetic_iot_data.py",
            "--output",
            str(output),
            "--days",
            "1",
            "--assets",
            "2",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["rows"] == 2_880
    assert summary["columns"] == len(DATASET_COLUMNS)
    assert output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("days", 0),
        ("n_assets", 0),
        ("frequency_minutes", 7),
        ("missing_rate", -0.1),
        ("missing_rate", 0.25),
        ("fault_events_per_asset", -1),
        ("partial_label_probability", 1.1),
    ],
)
def test_invalid_config_rejected(field, value):
    config = small_config(**{field: value})
    with pytest.raises(ValueError):
        generate_synthetic_iot_data(config)
