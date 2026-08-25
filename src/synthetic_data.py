"""Generazione riproducibile di serie temporali IoT industriali sintetiche.

Il generatore e' originale e non legge, campiona o trasforma dataset esterni. Produce lo schema
pubblico usato dal progetto, con regimi operativi, fault sintetici e missing controllati.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "ambient_temp_c",
    "humidity_pct",
    "load_pct",
    "rpm",
    "current_a",
    "pressure_bar",
    "flow_lpm",
    "temp_c",
    "vib_rms",
    "vib_crest",
    "vib_kurtosis",
]

DATASET_COLUMNS = [
    "timestamp",
    "asset_id",
    "site_id",
    "line_id",
    "regime",
    *SENSOR_COLUMNS,
    "fault_code_true",
    "fault_type_true",
    "anomaly_label",
]

FAULT_TYPES = {
    0: "none",
    1: "bearing_wear",
    2: "imbalance",
    3: "overheating",
    4: "shock",
}


@dataclass(frozen=True)
class SyntheticIoTConfig:
    """Parametri pubblici del generatore.

    I default producono 230.400 righe: 16 asset, 10 giorni, frequenza al minuto.
    """

    start: str = "2025-02-01 00:00:00"
    days: int = 10
    n_assets: int = 16
    frequency_minutes: int = 1
    seed: int = 20260825
    missing_rate: float = 0.0038
    fault_events_per_asset: int = 4
    partial_label_probability: float = 0.60

    @property
    def periods_per_asset(self):
        return self.days * 24 * 60 // self.frequency_minutes

    @property
    def expected_rows(self):
        return self.periods_per_asset * self.n_assets


def _validate_config(config):
    if config.days <= 0:
        raise ValueError("days deve essere positivo")
    if config.n_assets <= 0:
        raise ValueError("n_assets deve essere positivo")
    if config.frequency_minutes <= 0:
        raise ValueError("frequency_minutes deve essere positivo")
    if (24 * 60) % config.frequency_minutes != 0:
        raise ValueError("frequency_minutes deve dividere esattamente 1.440 minuti")
    if not 0 <= config.missing_rate < 0.25:
        raise ValueError("missing_rate deve essere compreso tra 0 incluso e 0,25 escluso")
    if config.fault_events_per_asset < 0:
        raise ValueError("fault_events_per_asset non può essere negativo")
    if not 0 <= config.partial_label_probability <= 1:
        raise ValueError("partial_label_probability deve essere compresa tra 0 e 1")
    if config.fault_events_per_asset and config.periods_per_asset < 20 * config.fault_events_per_asset:
        raise ValueError("intervallo troppo corto per distribuire gli eventi di guasto senza sovrapposizioni")


def _fault_windows(config, asset_index, rng):
    periods = config.periods_per_asset
    n_events = config.fault_events_per_asset
    if n_events == 0:
        return []

    segment = periods // n_events
    requested_durations = (90, 120, 150, 180)
    windows = []
    for event_index in range(n_events):
        segment_start = event_index * segment
        segment_end = periods if event_index == n_events - 1 else (event_index + 1) * segment
        duration = min(
            requested_durations[(asset_index + event_index) % len(requested_durations)],
            max(2, (segment_end - segment_start) // 3),
        )
        low = segment_start + 1
        high = max(low, segment_end - duration - 1)
        start = int(rng.integers(low, high + 1))
        code = (asset_index + event_index) % 4 + 1
        windows.append((start, start + duration, code))
    return windows


def _inject_faults(frame, config, rng):
    periods = config.periods_per_asset
    fault_code = np.zeros(len(frame), dtype=np.int8)

    for asset_index in range(config.n_assets):
        offset = asset_index * periods
        for start, end, code in _fault_windows(config, asset_index, rng):
            rows = np.arange(offset + start, offset + end)
            local = np.arange(len(rows), dtype=float)
            progress = local / max(1, len(rows) - 1)
            fault_code[rows] = code

            if code == 1:  # usura cuscinetto: deriva progressiva
                frame.loc[rows, "vib_rms"] += 0.20 + 1.00 * progress
                frame.loc[rows, "vib_crest"] += 0.10 + 0.60 * progress
                frame.loc[rows, "vib_kurtosis"] += 0.40 + 1.80 * progress
                frame.loc[rows, "temp_c"] += 0.20 + 2.30 * progress
            elif code == 2:  # sbilanciamento: oscillazione quasi-periodica
                severity = 0.35 + 0.85 * np.abs(np.sin(2 * np.pi * local / 12.0))
                frame.loc[rows, "vib_rms"] += severity
                frame.loc[rows, "current_a"] += 0.30 + 1.20 * severity
                frame.loc[rows, "vib_crest"] += 0.20 + 0.40 * severity
            elif code == 3:  # surriscaldamento: crescita lenta
                frame.loc[rows, "temp_c"] += 2.00 + 10.00 * progress
                frame.loc[rows, "current_a"] += 0.50 + 2.00 * progress
                frame.loc[rows, "pressure_bar"] += 0.10 + 0.50 * progress
            else:  # urti: picchi brevi entro una finestra degradata
                pulse = ((local.astype(int) % 17) < 2).astype(float)
                frame.loc[rows, "vib_rms"] += 0.15 + 2.00 * pulse
                frame.loc[rows, "vib_crest"] += 0.20 + 3.00 * pulse
                frame.loc[rows, "vib_kurtosis"] += 0.50 + 8.00 * pulse

    frame["fault_code_true"] = fault_code
    frame["fault_type_true"] = pd.Series(fault_code).map(FAULT_TYPES).astype("string")

    annotated = (fault_code != 0) & (rng.random(len(frame)) < config.partial_label_probability)
    frame["anomaly_label"] = annotated.astype(np.int8)


def _inject_missing_values(frame, config, rng):
    n_rows = len(frame)
    n_sensors = len(SENSOR_COLUMNS)
    mask = np.zeros((n_rows, n_sensors), dtype=bool)
    periods = config.periods_per_asset

    if config.missing_rate == 0:
        return

    for asset_index in range(config.n_assets):
        for gap_index in range(2):
            sensor_index = (asset_index + gap_index * 3) % n_sensors
            local_start = ((gap_index + 1) * periods // 3 + asset_index * 97) % max(1, periods - 10)
            start = asset_index * periods + local_start
            mask[start : start + 9, sensor_index] = True

    target = int(round(n_rows * n_sensors * config.missing_rate))
    already_selected = int(mask.sum())
    if target < already_selected:
        raise ValueError(
            "missing_rate troppo basso per includere i gap lunghi deterministici: "
            f"servono almeno {already_selected / (n_rows * n_sensors):.6f}"
        )
    remaining = target - already_selected
    if remaining:
        candidates = np.flatnonzero(~mask.ravel())
        chosen = rng.choice(candidates, size=remaining, replace=False)
        mask.ravel()[chosen] = True

    frame.loc[:, SENSOR_COLUMNS] = frame[SENSOR_COLUMNS].mask(mask)


def generate_synthetic_iot_data(config=None):
    """Genera un dataframe deterministico conforme allo schema pubblico del progetto."""

    config = config or SyntheticIoTConfig()
    _validate_config(config)
    rng = np.random.default_rng(config.seed)

    periods = config.periods_per_asset
    timestamps = pd.date_range(
        config.start,
        periods=periods,
        freq=f"{config.frequency_minutes}min",
        tz="UTC",
    )
    asset_index = np.repeat(np.arange(config.n_assets), periods)
    minute_index = np.tile(np.arange(periods), config.n_assets)
    timestamp_values = np.tile(timestamps.to_numpy(dtype="datetime64[ns]"), config.n_assets)
    n_rows = len(asset_index)

    minute_of_day = minute_index % (24 * 60 // config.frequency_minutes)
    shifted_minute = (
        minute_of_day * config.frequency_minutes + asset_index * 17
    ) % (24 * 60)
    hour = shifted_minute / 60.0
    regime = np.where(hour < 8, 0, np.where(hour < 16, 1, 2)).astype(np.int8)
    previous_regime = (regime - 1) % 3
    minutes_in_regime = shifted_minute % (8 * 60)
    transition_progress = np.clip(minutes_in_regime / 45.0, 0.0, 1.0)
    transition_weight = 0.5 - 0.5 * np.cos(np.pi * transition_progress)

    def smooth_regime_reference(values):
        values = np.asarray(values, dtype=float)
        return values[previous_regime] * (1 - transition_weight) + values[regime] * transition_weight

    daily_phase = 2 * np.pi * (minute_of_day * config.frequency_minutes) / (24 * 60)
    asset_offset = asset_index - (config.n_assets - 1) / 2
    # Tre modi fisicamente diversi e non collineari: standby, produzione ad alta
    # velocita' e produzione ad alto carico/coppia. La non collinearita' evita che
    # K-Means fonda due regimi e usi il terzo cluster soltanto per i fault.
    load_reference = smooth_regime_reference([7.0, 46.0, 91.0])

    ambient_temp = 22.0 + 5.5 * np.sin(daily_phase - np.pi / 2) + 0.08 * asset_offset + rng.normal(0, 0.45, n_rows)
    humidity = 58.0 - 13.0 * np.sin(daily_phase - np.pi / 2) + rng.normal(0, 1.8, n_rows)
    load = (
        load_reference
        + 2.5 * np.sin(daily_phase + asset_index / 5)
        + 0.25 * asset_offset
        + rng.normal(0, 2.0, n_rows)
    )
    load = np.clip(load, 0, 100)

    load_delta = load - load_reference
    rpm = np.clip(
        smooth_regime_reference([90.0, 2_650.0, 1_750.0])
        + 13.0 * load_delta
        + 8.0 * asset_offset
        + rng.normal(0, 28.0, n_rows),
        0,
        None,
    )
    current = np.clip(
        smooth_regime_reference([2.5, 14.0, 30.0])
        + 0.10 * load_delta
        + 0.10 * asset_offset
        + rng.normal(0, 0.50, n_rows),
        0,
        None,
    )
    pressure = np.clip(
        smooth_regime_reference([1.2, 4.6, 8.2])
        + 0.018 * load_delta
        + 0.015 * asset_offset
        + rng.normal(0, 0.12, n_rows),
        0,
        None,
    )
    flow = np.clip(
        smooth_regime_reference([6.0, 112.0, 64.0])
        + 0.35 * load_delta
        + 0.35 * asset_offset
        + rng.normal(0, 1.8, n_rows),
        0,
        None,
    )
    machine_temp = (
        ambient_temp
        + smooth_regime_reference([4.0, 19.0, 43.0])
        + 0.10 * load_delta
        + 0.12 * asset_offset
        + rng.normal(0, 0.65, n_rows)
    )
    vib_rms = np.clip(
        smooth_regime_reference([0.14, 0.58, 1.32])
        + 0.004 * load_delta
        + 0.010 * asset_index
        + rng.normal(0, 0.055, n_rows),
        0,
        None,
    )
    vib_crest = np.clip(
        smooth_regime_reference([2.15, 3.35, 2.75])
        + rng.normal(0, 0.12, n_rows),
        0,
        None,
    )
    vib_kurtosis = np.clip(
        smooth_regime_reference([2.8, 3.3, 4.9])
        + rng.normal(0, 0.17, n_rows),
        0,
        None,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamp_values, utc=True),
            "asset_id": asset_index + 1,
            "site_id": np.char.add("S", np.char.zfill((asset_index // 4 + 1).astype(str), 2)),
            "line_id": np.char.add("L", np.char.zfill((asset_index // 2 + 1).astype(str), 2)),
            "regime": regime,
            "ambient_temp_c": ambient_temp,
            "humidity_pct": np.clip(humidity, 0, 100),
            "load_pct": load,
            "rpm": rpm,
            "current_a": current,
            "pressure_bar": pressure,
            "flow_lpm": flow,
            "temp_c": machine_temp,
            "vib_rms": vib_rms,
            "vib_crest": vib_crest,
            "vib_kurtosis": vib_kurtosis,
        }
    )

    _inject_faults(frame, config, rng)
    _inject_missing_values(frame, config, rng)
    frame[SENSOR_COLUMNS] = frame[SENSOR_COLUMNS].round(6)
    return frame[DATASET_COLUMNS]


def write_synthetic_iot_csv(path, config=None):
    """Genera e scrive il CSV con UTF-8 e terminatori LF; ritorna il dataframe scritto."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_iot_data(config)
    frame.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z")
    return frame
