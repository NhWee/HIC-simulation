import argparse
import csv
from pathlib import Path

from hic.hydro_validation import build_parser as build_validation_parser
from hic.hydro_validation import run_validation


def validation_namespace(args: argparse.Namespace, grid_size: int, cfl: float) -> argparse.Namespace:
    # Start from the validation suite's own defaults so that new validation options never go
    # missing here, then override the options this scan controls.
    namespace = build_validation_parser().parse_args([])
    for key, value in vars(args).items():
        if hasattr(namespace, key) and key != "output":
            setattr(namespace, key, value)
    namespace.grid_size = grid_size
    namespace.cfl = cfl
    return namespace


def run_scan(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows = []
    for grid_size in args.grid_sizes:
        for cfl in args.cfl_values:
            print(f"Validation scan: grid_size={grid_size}, cfl={cfl}")
            validation_args = validation_namespace(args, grid_size, cfl)
            results = run_validation(validation_args)
            row: dict[str, float | str] = {
                "grid_size": float(grid_size),
                "cfl": float(cfl),
                "status": "PASS" if all(result["status"] != "FAIL" for result in results) else "FAIL",
            }
            for result in results:
                test = str(result["test"])
                metric = str(result["metric"])
                row[f"{test}_{metric}"] = float(result["value"])
                row[f"{test}_status"] = str(result["status"])
            rows.append(row)
    return rows


def parse_list(text: str | list[str], cast_type: type) -> list:
    items = text if isinstance(text, list) else [text]
    parsed = []
    for item in items:
        parsed.extend(cast_type(value.strip()) for value in item.split(",") if value.strip())
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hydro validation over grid-size and CFL scans.")
    parser.add_argument("--grid-sizes", nargs="+", default=["40,60,80"], help="Grid sizes, comma-separated or space-separated.")
    parser.add_argument("--cfl-values", nargs="+", default=["0.15,0.25,0.35"], help="CFL values, comma-separated or space-separated.")
    parser.add_argument("--extent", type=float, default=12.0, help="Half-width of the transverse plane.")
    parser.add_argument("--steps", type=int, default=30, help="Validation steps per run.")
    parser.add_argument("--eos", choices=("conformal", "crossover", "table"), default="conformal", help="Equation of state.")
    parser.add_argument("--eos-table", type=Path, default=Path("data/eos_table.csv"), help="EoS table for --eos table.")
    parser.add_argument("--speed-of-sound2", type=float, default=1.0 / 3.0, help="Conformal sound speed squared.")
    parser.add_argument("--low-speed-of-sound2", type=float, default=0.15, help="Crossover low-density sound speed squared.")
    parser.add_argument("--high-speed-of-sound2", type=float, default=1.0 / 3.0, help="Crossover high-density sound speed squared.")
    parser.add_argument("--transition-energy", type=float, default=1.0, help="Crossover midpoint energy density.")
    parser.add_argument("--transition-width", type=float, default=0.25, help="Crossover width.")
    parser.add_argument("--energy-floor", type=float, default=1.0e-8, help="Minimum energy density.")
    parser.add_argument("--tau0", type=float, default=0.6, help="Initial proper time.")
    parser.add_argument("--boost-velocity", type=float, default=0.2, help="Uniform boost velocity.")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0, help="Gaussian fireball width.")
    parser.add_argument("--uniform-tolerance", type=float, default=1.0e-6, help="Uniform test tolerance.")
    parser.add_argument("--gaussian-energy-tolerance", type=float, default=2.0e-3, help="Gaussian energy drift tolerance.")
    parser.add_argument("--bjorken-tolerance", type=float, default=1.0e-5, help="Bjorken analytic tolerance.")
    parser.add_argument(
        "--crossover-bjorken-tolerance",
        type=float,
        default=5.0e-3,
        help="Coarse/fine relative tolerance for crossover Bjorken cooling.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output", type=Path, default=Path("outputs/hydro_convergence.csv"), help="Output CSV path.")
    parsed = parser.parse_args()
    parsed.grid_sizes = parse_list(parsed.grid_sizes, int)
    parsed.cfl_values = parse_list(parsed.cfl_values, float)
    return parsed


def main() -> None:
    args = parse_args()
    rows = run_scan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(1 for row in rows if row["status"] == "FAIL")
    print(f"Saved hydro convergence scan: {args.output}")
    print(f"runs={len(rows)}, failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
