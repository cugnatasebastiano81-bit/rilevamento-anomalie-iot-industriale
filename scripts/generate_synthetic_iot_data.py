"""CLI per generare il dataset IoT sintetico riproducibile del progetto."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import SENSOR_COLUMNS, SyntheticIoTConfig, write_synthetic_iot_csv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/iot_synth_kaggle_generated.csv"),
        help="Percorso del CSV da creare.",
    )
    parser.add_argument("--seed", type=int, default=SyntheticIoTConfig.seed)
    parser.add_argument("--days", type=int, default=SyntheticIoTConfig.days)
    parser.add_argument("--assets", type=int, default=SyntheticIoTConfig.n_assets)
    parser.add_argument("--missing-rate", type=float, default=SyntheticIoTConfig.missing_rate)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    config = SyntheticIoTConfig(
        seed=args.seed,
        days=args.days,
        n_assets=args.assets,
        missing_rate=args.missing_rate,
    )
    frame = write_synthetic_iot_csv(args.output, config)
    summary = {
        "output": args.output.as_posix(),
        "rows": len(frame),
        "columns": len(frame.columns),
        "assets": int(frame["asset_id"].nunique()),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "missing_sensor_fraction": float(frame[SENSOR_COLUMNS].isna().to_numpy().mean()),
        "fault_fraction": float((frame["fault_code_true"] != 0).mean()),
        "partial_label_fraction": float(frame["anomaly_label"].mean()),
        "seed": config.seed,
        "sha256": sha256_file(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
