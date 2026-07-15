import argparse
import csv
import sys
from pathlib import Path

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
        r"  .\run_hydro_shock_tube.ps1"
    ) from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hic.eos import add_eos_arguments, build_eos
from hic.ideal_hydro_2d import (
    conserved_from_primitive,
    conserved_totals,
    primitive_from_conserved,
    rusanov_step,
)
from hic.nuclear_collision_2d import make_grid


def build_shock_tube_state(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    xx, _, dx = make_grid(args.grid_size, args.extent, device)
    eos = build_eos(args)
    left = xx < args.discontinuity_x
    energy_density = torch.where(
        left,
        torch.full_like(xx, args.left_energy_density),
        torch.full_like(xx, args.right_energy_density),
    )
    velocity_x = torch.zeros_like(energy_density)
    velocity_y = torch.zeros_like(energy_density)
    state = conserved_from_primitive(energy_density, velocity_x, velocity_y, eos)
    return xx, state, dx


def run_shock_tube(args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    xx, state, dx = build_shock_tube_state(args, device)
    eos = build_eos(args)
    initial_energy, initial_px, initial_py = conserved_totals(state, dx)
    dt = args.cfl * dx / max(1.0, eos.max_sound_speed())

    for _ in range(args.steps):
        state = rusanov_step(state, dx, dt, eos, args.energy_floor)

    energy_density, pressure, velocity_x, velocity_y = primitive_from_conserved(
        state[0], state[1], state[2], eos, args.energy_floor
    )
    final_energy, final_px, final_py = conserved_totals(state, dx)
    mid_y = args.grid_size // 2

    center_energy_density = energy_density[mid_y]
    center_velocity_x = velocity_x[mid_y]
    center_velocity_y = velocity_y[mid_y]
    rows = []
    for index in range(args.grid_size):
        rows.append(
            {
                "x": float(xx[mid_y, index].detach().cpu()),
                "energy_density": float(center_energy_density[index].detach().cpu()),
                "pressure": float(pressure[mid_y, index].detach().cpu()),
                "velocity_x": float(center_velocity_x[index].detach().cpu()),
                "velocity_y": float(center_velocity_y[index].detach().cpu()),
            }
        )

    center_speed = torch.sqrt(center_velocity_x * center_velocity_x + center_velocity_y * center_velocity_y)
    diagnostics = {
        "time": args.steps * dt,
        "eos": eos.kind,
        "initial_total_energy": initial_energy,
        "final_total_energy": final_energy,
        "relative_energy_drift": abs(final_energy - initial_energy) / max(abs(initial_energy), 1.0e-12),
        "initial_px": initial_px,
        "final_px": final_px,
        "initial_py": initial_py,
        "final_py": final_py,
        "center_min_energy_density": float(torch.min(center_energy_density).detach().cpu()),
        "center_max_velocity": float(torch.max(center_speed).detach().cpu()),
    }
    return rows, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a relativistic ideal-hydro shock-tube benchmark.")
    parser.add_argument("--grid-size", type=int, default=200, help="2D grid size; the center row is written as 1D output.")
    parser.add_argument("--extent", type=float, default=10.0, help="Half-width of the domain.")
    parser.add_argument("--steps", type=int, default=80, help="Hydrodynamic time steps.")
    parser.add_argument("--cfl", type=float, default=0.2, help="Courant factor.")
    add_eos_arguments(parser)
    parser.add_argument("--energy-floor", type=float, default=1.0e-8, help="Minimum energy density.")
    parser.add_argument("--left-energy-density", type=float, default=10.0, help="Left-state energy density.")
    parser.add_argument("--right-energy-density", type=float, default=1.0, help="Right-state energy density.")
    parser.add_argument("--discontinuity-x", type=float, default=0.0, help="Initial discontinuity location.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output", type=Path, default=Path("outputs/hydro_shock_tube.csv"), help="Output profile CSV path.")
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=Path("outputs/hydro_shock_tube_diagnostics.csv"),
        help="Output diagnostics CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, diagnostics = run_shock_tube(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["x", "energy_density", "pressure", "velocity_x", "velocity_y"])
        writer.writeheader()
        writer.writerows(rows)

    with args.diagnostics_output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(diagnostics.keys()))
        writer.writeheader()
        writer.writerow(diagnostics)

    print(
        "Shock tube diagnostics: "
        f"t={diagnostics['time']:.4f}, "
        f"relative_energy_drift={diagnostics['relative_energy_drift']:.6g}, "
        f"center_max_velocity={diagnostics['center_max_velocity']:.6g}"
    )
    print(f"Saved shock-tube profile: {args.output}")
    print(f"Saved shock-tube diagnostics: {args.diagnostics_output}")


if __name__ == "__main__":
    main()
