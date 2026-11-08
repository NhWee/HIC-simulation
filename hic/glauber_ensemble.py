import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise SystemExit(
        "PyTorch is not installed in this Python environment.\n"
        "Set up the project environment first:\n"
        r"  .\setup.ps1"
        "\nThen use the project launcher:\n"
        r"  .\run_glauber_ensemble.ps1 --events 200 --output outputs\glauber_ensemble.csv"
    ) from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hic.nuclear_collision_2d import (
    gaussian_point_thickness,
    make_grid,
    sample_woods_saxon_nucleons,
    tag_glauber_collisions,
)


def eccentricity_n(e: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor, harmonic: int) -> float:
    weight = torch.clamp(e, min=0.0)
    r = torch.sqrt(xx * xx + yy * yy)
    phi = torch.atan2(yy, xx)
    numerator_real = torch.sum(weight * torch.pow(r, harmonic) * torch.cos(harmonic * phi))
    numerator_imag = torch.sum(weight * torch.pow(r, harmonic) * torch.sin(harmonic * phi))
    denominator = torch.sum(weight * torch.pow(r, harmonic)) + 1.0e-8
    eps_n = torch.sqrt(numerator_real * numerator_real + numerator_imag * numerator_imag) / denominator
    return float(eps_n.detach().cpu())


def rms_radius(e: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor) -> float:
    weight = torch.clamp(e, min=0.0)
    total = torch.sum(weight) + 1.0e-8
    r2 = xx * xx + yy * yy
    return float(torch.sqrt(torch.sum(weight * r2) / total).detach().cpu())


def sample_impact_parameter(rng: np.random.Generator, b_min: float, b_max: float) -> float:
    return float(np.sqrt(rng.uniform(b_min * b_min, b_max * b_max)))


def build_initial_source(
    xx: torch.Tensor,
    yy: torch.Tensor,
    impact_parameter: float,
    mass_number: int,
    radius: float,
    skin_depth: float,
    nucleon_width: float,
    inelastic_cross_section: float,
    binary_fraction: float,
    norm: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    ax, ay = sample_woods_saxon_nucleons(mass_number, radius, skin_depth, rng)
    bx, by = sample_woods_saxon_nucleons(mass_number, radius, skin_depth, rng)
    participant_a, participant_b, coll_x, coll_y = tag_glauber_collisions(
        ax, ay, bx, by, impact_parameter, inelastic_cross_section
    )

    participants_a = (
        torch.as_tensor(ax[participant_a], device=device),
        torch.as_tensor(ay[participant_a], device=device),
    )
    participants_b = (
        torch.as_tensor(bx[participant_b], device=device),
        torch.as_tensor(by[participant_b], device=device),
    )
    collision_centers = (
        torch.as_tensor(coll_x, device=device),
        torch.as_tensor(coll_y, device=device),
    )
    center_x = torch.tensor(0.0, device=device)
    center_a_y = torch.tensor(-0.5 * impact_parameter, device=device)
    center_b_y = torch.tensor(0.5 * impact_parameter, device=device)
    center_y = torch.tensor(0.0, device=device)

    participant_a_thickness = gaussian_point_thickness(
        xx,
        yy,
        participants_a[0],
        participants_a[1],
        center_x,
        center_a_y,
        nucleon_width,
        max(participants_a[0].numel(), 1),
    )
    participant_b_thickness = gaussian_point_thickness(
        xx,
        yy,
        participants_b[0],
        participants_b[1],
        center_x,
        center_b_y,
        nucleon_width,
        max(participants_b[0].numel(), 1),
    )
    binary = gaussian_point_thickness(
        xx,
        yy,
        collision_centers[0],
        collision_centers[1],
        center_x,
        center_y,
        nucleon_width,
        max(collision_centers[0].numel(), 1),
    )

    wounded = participant_a_thickness + participant_b_thickness
    source = norm * ((1.0 - binary_fraction) * wounded + binary_fraction * binary)
    n_part = int(participant_a.sum() + participant_b.sum())
    n_coll = int(coll_x.shape[0])
    return source, n_part, n_coll


def run_ensemble(args: argparse.Namespace) -> list[dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    xx, yy, dx = make_grid(args.grid_size, args.extent, device)
    rng = np.random.default_rng(args.seed)
    rows = []

    for event_index in range(args.events):
        impact_parameter = sample_impact_parameter(rng, args.b_min, args.b_max)
        source, n_part, n_coll = build_initial_source(
            xx=xx,
            yy=yy,
            impact_parameter=impact_parameter,
            mass_number=args.mass_number,
            radius=args.radius,
            skin_depth=args.skin_depth,
            nucleon_width=args.nucleon_width,
            inelastic_cross_section=args.inelastic_cross_section,
            binary_fraction=args.binary_fraction,
            norm=args.norm,
            rng=rng,
            device=device,
        )
        total_source = float((torch.sum(source) * dx * dx).detach().cpu())
        rows.append(
            {
                "event": event_index,
                "seed": args.seed,
                "impact_parameter": impact_parameter,
                "n_part": n_part,
                "n_coll": n_coll,
                "total_source": total_source,
                "eps2": eccentricity_n(source, xx, yy, 2),
                "eps3": eccentricity_n(source, xx, yy, 3),
                "rms_radius": rms_radius(source, xx, yy),
            }
        )

    ranked = sorted(rows, key=lambda row: row["total_source"], reverse=True)
    denominator = max(len(ranked), 1)
    for rank, row in enumerate(ranked):
        row["centrality_percentile"] = 100.0 * (rank + 0.5) / denominator

    return sorted(ranked, key=lambda row: row["event"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an event-by-event Monte Carlo Glauber ensemble.")
    parser.add_argument("--events", type=int, default=200, help="Number of Monte Carlo Glauber events.")
    parser.add_argument("--grid-size", type=int, default=120, help="2D transverse grid size.")
    parser.add_argument("--extent", type=float, default=12.0, help="Half-width of the transverse plane in fm-like units.")
    parser.add_argument("--b-min", type=float, default=0.0, help="Minimum sampled impact parameter.")
    parser.add_argument("--b-max", type=float, default=15.0, help="Maximum sampled impact parameter.")
    parser.add_argument("--mass-number", type=int, default=197, help="Nuclear mass number.")
    parser.add_argument("--radius", type=float, default=6.4, help="Nuclear radius.")
    parser.add_argument("--skin-depth", type=float, default=0.55, help="Woods-Saxon surface diffuseness.")
    parser.add_argument("--nucleon-width", type=float, default=0.45, help="Gaussian nucleon width.")
    parser.add_argument(
        "--inelastic-cross-section",
        type=float,
        default=4.2,
        help="Nucleon-nucleon inelastic cross section in fm^2.",
    )
    parser.add_argument("--binary-fraction", type=float, default=0.16, help="Wounded/binary source mixing fraction.")
    parser.add_argument("--norm", type=float, default=2.5, help="Overall source normalization.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output", type=Path, default=Path("outputs/glauber_ensemble.csv"), help="Output CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = run_ensemble(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event",
        "seed",
        "impact_parameter",
        "centrality_percentile",
        "n_part",
        "n_coll",
        "total_source",
        "eps2",
        "eps3",
        "rms_radius",
    ]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved Glauber ensemble: {args.output}")


if __name__ == "__main__":
    main()
