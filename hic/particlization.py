import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def read_freezeout_surface(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def sample_thermal_pt(rng: np.random.Generator, temperature: float, mass: float, size: int) -> np.ndarray:
    samples = []
    max_weight = max(temperature * math.exp(-mass / max(temperature, 1.0e-12)), 1.0e-12)
    while len(samples) < size:
        batch = max(256, 4 * (size - len(samples)))
        pt = rng.exponential(scale=2.0 * temperature, size=batch)
        mt = np.sqrt(pt * pt + mass * mass)
        accept_prob = pt * np.exp(-mt / max(temperature, 1.0e-12)) / max_weight
        accepted = pt[rng.random(batch) < np.clip(accept_prob, 0.0, 1.0)]
        samples.extend(accepted.tolist())
    return np.asarray(samples[:size], dtype=np.float64)


def sample_particles(args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, float]]:
    surface = read_freezeout_surface(args.freezeout_surface)
    if not surface:
        return [], {
            "n_particles": 0,
            "mean_pt": 0.0,
            "v2": 0.0,
            "v3": 0.0,
        }

    rng = np.random.default_rng(args.seed)
    weights = np.asarray([max(row["d_sigma_tau"], 0.0) * max(row["energy_density"], 0.0) * max(row["tau"], 1.0e-12) for row in surface])
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(surface), dtype=np.float64)
    weights = weights / weights.sum()

    cell_indices = rng.choice(len(surface), size=args.particles, replace=True, p=weights)
    surface_temperatures = np.asarray([row.get("temperature", args.temperature) for row in surface], dtype=np.float64)
    sampled_temperatures = surface_temperatures[cell_indices]
    local_pt = np.asarray(
        [sample_thermal_pt(rng, max(float(temperature), 1.0e-6), args.mass, 1)[0] for temperature in sampled_temperatures],
        dtype=np.float64,
    )
    local_phi = rng.uniform(0.0, 2.0 * math.pi, size=args.particles)

    rows = []
    pt_values = []
    phi_values = []
    for particle_id, cell_index in enumerate(cell_indices.tolist()):
        cell = surface[cell_index]
        temperature = cell.get("temperature", args.temperature)
        ux = cell["velocity_x"]
        uy = cell["velocity_y"]
        flow_speed = min(math.sqrt(ux * ux + uy * uy), 0.95)
        flow_phi = math.atan2(uy, ux) if flow_speed > 1.0e-12 else 0.0

        px = local_pt[particle_id] * math.cos(local_phi[particle_id]) + args.flow_kick * args.mass * ux
        py = local_pt[particle_id] * math.sin(local_phi[particle_id]) + args.flow_kick * args.mass * uy
        pt = math.sqrt(px * px + py * py)
        phi = math.atan2(py, px)
        mt = math.sqrt(pt * pt + args.mass * args.mass)
        gamma = 1.0 / max(math.sqrt(1.0 - flow_speed * flow_speed), 1.0e-12)
        p_dot_u = gamma * (mt - px * ux - py * uy)
        p_dot_dsigma = mt * cell.get("d_sigma_tau", 0.0) + px * cell.get("d_sigma_x", 0.0) + py * cell.get("d_sigma_y", 0.0)
        cooper_frye_weight = max(p_dot_dsigma, 0.0) * math.exp(-p_dot_u / max(temperature, 1.0e-12))

        pt_values.append(pt)
        phi_values.append(phi)
        rows.append(
            {
                "particle_id": particle_id,
                "cell_index": cell_index,
                "tau": cell["tau"],
                "x": cell["x"],
                "y": cell["y"],
                "mass": args.mass,
                "temperature": temperature,
                "pt": pt,
                "phi": phi,
                "px": px,
                "py": py,
                "p_dot_u": p_dot_u,
                "p_dot_dsigma": p_dot_dsigma,
                "cooper_frye_weight": cooper_frye_weight,
                "flow_speed": flow_speed,
                "flow_phi": flow_phi,
            }
        )

    pt_array = np.asarray(pt_values, dtype=np.float64)
    phi_array = np.asarray(phi_values, dtype=np.float64)
    summary = {
        "n_particles": args.particles,
        "mean_pt": float(pt_array.mean()),
        "v2": float(np.mean(np.cos(2.0 * phi_array))),
        "v3": float(np.mean(np.cos(3.0 * phi_array))),
        "mean_cooper_frye_weight": float(np.mean([row["cooper_frye_weight"] for row in rows])),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample pion-like particles from freeze-out candidate cells.")
    parser.add_argument("--freezeout-surface", type=Path, default=Path("outputs/freezeout_surface.csv"), help="Freeze-out surface CSV.")
    parser.add_argument("--particles", type=int, default=10000, help="Number of sampled particles.")
    parser.add_argument("--temperature", type=float, default=0.145, help="Freeze-out temperature in GeV-like units.")
    parser.add_argument("--mass", type=float, default=0.13957, help="Sampled hadron mass, pion by default.")
    parser.add_argument("--flow-kick", type=float, default=1.0, help="Strength of approximate transverse-flow momentum shift.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("outputs/particles.csv"), help="Output particle CSV path.")
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/particle_observables.csv"), help="Output summary CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.freezeout_surface.exists():
        raise SystemExit(f"Freeze-out surface not found: {args.freezeout_surface}")

    rows, summary = sample_particles(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    particle_fieldnames = [
        "particle_id",
        "cell_index",
        "tau",
        "x",
        "y",
        "mass",
        "temperature",
        "pt",
        "phi",
        "px",
        "py",
        "p_dot_u",
        "p_dot_dsigma",
        "cooper_frye_weight",
        "flow_speed",
        "flow_phi",
    ]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=particle_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with args.summary_output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["n_particles", "mean_pt", "v2", "v3", "mean_cooper_frye_weight"])
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved sampled particles: {args.output}")
    print(f"Saved particle observables: {args.summary_output}")
    print(
        f"n={summary['n_particles']}, mean_pt={summary['mean_pt']:.6g}, "
        f"v2={summary['v2']:.6g}, v3={summary['v3']:.6g}, "
        f"mean_cf_weight={summary['mean_cooper_frye_weight']:.6g}"
    )


if __name__ == "__main__":
    main()
