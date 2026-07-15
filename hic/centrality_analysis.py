import argparse
import csv
import math
from pathlib import Path

import numpy as np


DEFAULT_BINS = "0-5,5-10,10-20,20-30,30-40,40-50,50-60,60-70,70-80"


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def parse_bins(bin_text: str) -> list[tuple[float, float]]:
    bins = []
    for item in bin_text.split(","):
        lower, upper = item.split("-", maxsplit=1)
        bins.append((float(lower), float(upper)))
    return bins


def mean_and_error(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    return mean, float(values.std(ddof=1) / math.sqrt(values.size))


def summarize_bin(rows: list[dict[str, float]], lower: float, upper: float) -> dict[str, float | str]:
    selected = [row for row in rows if lower <= row["centrality_percentile"] < upper]
    summary: dict[str, float | str] = {
        "centrality_bin": f"{lower:g}-{upper:g}",
        "events": float(len(selected)),
    }
    for key in ("impact_parameter", "n_part", "n_coll", "total_source", "eps2", "eps3", "rms_radius"):
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        mean, stderr = mean_and_error(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_stderr"] = stderr
    return summary


def analyze(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows = read_rows(args.input)
    bins = parse_bins(args.bins)
    return [summarize_bin(rows, lower, upper) for lower, upper in bins]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Glauber ensemble observables by centrality bin.")
    parser.add_argument("--input", type=Path, default=Path("outputs/glauber_ensemble.csv"), help="Glauber ensemble CSV.")
    parser.add_argument("--bins", default=DEFAULT_BINS, help="Comma-separated centrality bins, e.g. 0-5,5-10,10-20.")
    parser.add_argument("--output", type=Path, default=Path("outputs/centrality_analysis.csv"), help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Glauber ensemble not found: {args.input}")
    summaries = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "centrality_bin",
        "events",
        "impact_parameter_mean",
        "impact_parameter_stderr",
        "n_part_mean",
        "n_part_stderr",
        "n_coll_mean",
        "n_coll_stderr",
        "total_source_mean",
        "total_source_stderr",
        "eps2_mean",
        "eps2_stderr",
        "eps3_mean",
        "eps3_stderr",
        "rms_radius_mean",
        "rms_radius_stderr",
    ]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved centrality analysis: {args.output}")
    for row in summaries:
        print(
            f"{row['centrality_bin']}%: events={row['events']:.0f}, "
            f"<Npart>={row['n_part_mean']:.2f}, <Ncoll>={row['n_coll_mean']:.2f}, "
            f"<eps2>={row['eps2_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
