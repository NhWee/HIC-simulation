import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def read_summary(path: Path) -> dict[str, float]:
    with path.open(newline="") as input_file:
        row = next(csv.DictReader(input_file))
    return {key: float(value) for key, value in row.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small end-to-end HIC event pipeline.")
    parser.add_argument("--events", type=int, default=3, help="Number of events.")
    parser.add_argument("--particles", type=int, default=2000, help="Particles sampled per event.")
    parser.add_argument("--grid-size", type=int, default=60, help="Hydro grid size.")
    parser.add_argument("--steps", type=int, default=60, help="Hydro steps.")
    parser.add_argument("--freezeout-energy", type=float, default=0.5, help="Freeze-out threshold.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline"), help="Per-event output directory.")
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/pipeline_summary.csv"), help="Aggregated summary CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    python = sys.executable
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for event in range(args.events):
        seed = args.seed + event
        hydro_output = args.output_dir / f"event_{event:04d}_hydro.csv"
        freezeout_output = args.output_dir / f"event_{event:04d}_freezeout.csv"
        particle_output = args.output_dir / f"event_{event:04d}_particles.csv"
        particle_summary = args.output_dir / f"event_{event:04d}_observables.csv"

        run_command(
            [
                python,
                "-m",
                "hic.ideal_hydro_2d",
                "--bjorken-expansion",
                "--grid-size",
                str(args.grid_size),
                "--steps",
                str(args.steps),
                "--seed",
                str(seed),
                "--freezeout-energy",
                str(args.freezeout_energy),
                "--output",
                str(hydro_output),
                "--freezeout-output",
                str(freezeout_output),
            ]
        )
        run_command(
            [
                python,
                "-m",
                "hic.particlization",
                "--freezeout-surface",
                str(freezeout_output),
                "--particles",
                str(args.particles),
                "--seed",
                str(seed),
                "--output",
                str(particle_output),
                "--summary-output",
                str(particle_summary),
            ]
        )
        summary = read_summary(particle_summary)
        summary["event"] = event
        summary["seed"] = seed
        summaries.append(summary)

    fieldnames = ["event", "seed", "n_particles", "mean_pt", "v2", "v3", "mean_cooper_frye_weight"]
    with args.summary_output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved pipeline summary: {args.summary_output}")


if __name__ == "__main__":
    main()
