import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def mean_and_error(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    stderr = float(values.std(ddof=1) / math.sqrt(values.size))
    return mean, stderr


def rms_flow(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(values * values))))


def analyze(args: argparse.Namespace) -> dict[str, float]:
    rows = read_rows(args.input)
    mean_pt = np.asarray([row["mean_pt"] for row in rows], dtype=np.float64)
    v2 = np.asarray([row["v2"] for row in rows], dtype=np.float64)
    v3 = np.asarray([row["v3"] for row in rows], dtype=np.float64)
    cf_weight = np.asarray([row["mean_cooper_frye_weight"] for row in rows], dtype=np.float64)

    mean_pt_mean, mean_pt_stderr = mean_and_error(mean_pt)
    v2_mean, v2_stderr = mean_and_error(v2)
    v3_mean, v3_stderr = mean_and_error(v3)
    cf_weight_mean, cf_weight_stderr = mean_and_error(cf_weight)

    return {
        "events": float(len(rows)),
        "mean_pt": mean_pt_mean,
        "mean_pt_stderr": mean_pt_stderr,
        "v2_mean": v2_mean,
        "v2_stderr": v2_stderr,
        "v2_rms": rms_flow(v2),
        "v3_mean": v3_mean,
        "v3_stderr": v3_stderr,
        "v3_rms": rms_flow(v3),
        "mean_cooper_frye_weight": cf_weight_mean,
        "mean_cooper_frye_weight_stderr": cf_weight_stderr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an event-pipeline summary CSV.")
    parser.add_argument("--input", type=Path, default=Path("outputs/pipeline_summary.csv"), help="Pipeline summary CSV.")
    parser.add_argument("--output", type=Path, default=Path("outputs/pipeline_analysis.csv"), help="Analysis output CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Pipeline summary not found: {args.input}")
    summary = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved pipeline analysis: {args.output}")
    print(
        f"events={summary['events']:.0f}, mean_pt={summary['mean_pt']:.6g} +/- {summary['mean_pt_stderr']:.3g}, "
        f"v2_rms={summary['v2_rms']:.6g}, v3_rms={summary['v3_rms']:.6g}"
    )


if __name__ == "__main__":
    main()
