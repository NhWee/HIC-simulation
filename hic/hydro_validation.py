import argparse
import csv
import math
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
        r"  .\run_hydro_validation.ps1"
    ) from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hic.eos import EquationOfState, add_eos_arguments, build_eos
from hic.ideal_hydro_2d import (
    apply_bjorken_cooling,
    apply_interface_fluxes,
    boundary_outflow,
    conserved_from_primitive,
    conserved_totals,
    primitive_from_conserved,
    rusanov_interface_fluxes,
    rusanov_step,
)
from hic.nuclear_collision_2d import make_grid


def evolve(
    state: torch.Tensor,
    dx: float,
    steps: int,
    cfl: float,
    eos,
    energy_floor: float,
    bjorken_expansion: bool = False,
    tau0: float = 0.6,
) -> torch.Tensor:
    dt = cfl * dx / max(1.0, eos.max_sound_speed())
    for step in range(steps):
        state = rusanov_step(state, dx, dt, eos, energy_floor)
        if bjorken_expansion:
            state = apply_bjorken_cooling(state, tau0 + step * dt, dt, eos, energy_floor)
    return state


def uniform_rest_test(args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    _, _, dx = make_grid(args.grid_size, args.extent, device)
    eos = build_eos(args)
    energy_density = torch.full((args.grid_size, args.grid_size), 1.0, device=device)
    velocity = torch.zeros_like(energy_density)
    initial = conserved_from_primitive(energy_density, velocity, velocity, eos)
    final = evolve(initial, dx, args.steps, args.cfl, eos, args.energy_floor)
    max_abs_delta = float(torch.max(torch.abs(final - initial)).detach().cpu())
    return {
        "test": "uniform_rest",
        "metric": "max_abs_conserved_delta",
        "value": max_abs_delta,
        "tolerance": args.uniform_tolerance,
        "status": "PASS" if max_abs_delta <= args.uniform_tolerance else "FAIL",
    }


def uniform_boost_test(args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    _, _, dx = make_grid(args.grid_size, args.extent, device)
    eos = build_eos(args)
    energy_density = torch.full((args.grid_size, args.grid_size), 1.0, device=device)
    velocity_x = torch.full_like(energy_density, args.boost_velocity)
    velocity_y = torch.zeros_like(energy_density)
    initial = conserved_from_primitive(energy_density, velocity_x, velocity_y, eos)
    final = evolve(initial, dx, args.steps, args.cfl, eos, args.energy_floor)
    max_abs_delta = float(torch.max(torch.abs(final - initial)).detach().cpu())
    return {
        "test": "uniform_boost",
        "metric": "max_abs_conserved_delta",
        "value": max_abs_delta,
        "tolerance": args.uniform_tolerance,
        "status": "PASS" if max_abs_delta <= args.uniform_tolerance else "FAIL",
    }


def gaussian_energy_test(args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    """Energy conservation of the finite-volume update for an expanding Gaussian.

    Energy that crosses the fixed outer cells is counted as outflow, so the
    check measures conservation of the scheme rather than how much of a
    numerically diffused profile has reached the grid edge.
    """
    xx, yy, dx = make_grid(args.grid_size, args.extent, device)
    eos = build_eos(args)
    r2 = xx * xx + yy * yy
    energy_density = args.energy_floor + torch.exp(-r2 / (2.0 * args.gaussian_sigma * args.gaussian_sigma))
    velocity = torch.zeros_like(energy_density)
    initial = conserved_from_primitive(energy_density, velocity, velocity, eos)
    dt = args.cfl * dx / max(1.0, eos.max_sound_speed())
    state = initial
    outflow = torch.zeros(3, dtype=initial.dtype, device=initial.device)
    for _ in range(args.steps):
        interface_x, interface_y = rusanov_interface_fluxes(state, eos, args.energy_floor)
        outflow = outflow + boundary_outflow(interface_x, interface_y, dx, dt)
        state = apply_interface_fluxes(state, interface_x, interface_y, dx, dt, args.energy_floor)
    initial_energy, _, _ = conserved_totals(initial, dx)
    final_energy, final_px, final_py = conserved_totals(state, dx)
    energy_outflow = float(outflow[0].detach().cpu())
    scale = max(abs(initial_energy), 1.0e-12)
    conservation_error = abs(final_energy + energy_outflow - initial_energy) / scale
    momentum_norm = (final_px * final_px + final_py * final_py) ** 0.5
    status = "PASS" if conservation_error <= args.gaussian_energy_tolerance else "FAIL"
    return {
        "test": "gaussian_fireball",
        "metric": "relative_energy_conservation_error",
        "value": conservation_error,
        "tolerance": args.gaussian_energy_tolerance,
        "status": status,
        "final_momentum_norm": momentum_norm,
        "boundary_outflow_fraction": energy_outflow / scale,
    }


def bjorken_uniform_cooling_test(args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    _, _, dx = make_grid(args.grid_size, args.extent, device)
    eos = build_eos(args)
    if eos.kind != "conformal":
        return {
            "test": "bjorken_uniform_cooling",
            "metric": "relative_energy_density_error",
            "value": 0.0,
            "tolerance": args.bjorken_tolerance,
            "status": "SKIP",
        }
    energy_density = torch.full((args.grid_size, args.grid_size), 1.0, device=device)
    velocity = torch.zeros_like(energy_density)
    initial = conserved_from_primitive(energy_density, velocity, velocity, eos)
    final = evolve(
        initial,
        dx,
        args.steps,
        args.cfl,
        eos,
        args.energy_floor,
        bjorken_expansion=True,
        tau0=args.tau0,
    )
    final_energy_density, _, _, _ = primitive_from_conserved(
        final[0], final[1], final[2], eos, args.energy_floor
    )
    dt = args.cfl * dx / max(1.0, eos.max_sound_speed())
    exact = eos.bjorken_cooling_factor(args.tau0, args.steps * dt)
    relative_error = float(
        (torch.max(torch.abs(final_energy_density - exact)) / max(abs(exact), 1.0e-12)).detach().cpu()
    )
    return {
        "test": "bjorken_uniform_cooling",
        "metric": "relative_energy_density_error",
        "value": relative_error,
        "tolerance": args.bjorken_tolerance,
        "status": "PASS" if relative_error <= args.bjorken_tolerance else "FAIL",
    }


def bjorken_crossover_convergence_test(args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    _, _, dx = make_grid(args.grid_size, args.extent, device)
    eos = EquationOfState(
        kind="crossover",
        speed_of_sound2=1.0 / 3.0,
        low_speed_of_sound2=0.15,
        high_speed_of_sound2=1.0 / 3.0,
        transition_energy=1.0,
        transition_width=0.25,
    )
    energy_density = torch.full((args.grid_size, args.grid_size), 1.0, device=device)
    velocity = torch.zeros_like(energy_density)
    initial = conserved_from_primitive(energy_density, velocity, velocity, eos)
    coarse = evolve(
        initial,
        dx,
        args.steps,
        args.cfl,
        eos,
        args.energy_floor,
        bjorken_expansion=True,
        tau0=args.tau0,
    )
    reference = evolve(
        initial,
        dx,
        4 * args.steps,
        args.cfl / 4.0,
        eos,
        args.energy_floor,
        bjorken_expansion=True,
        tau0=args.tau0,
    )
    coarse_energy, _, _, _ = primitive_from_conserved(coarse[0], coarse[1], coarse[2], eos, args.energy_floor)
    reference_energy, _, _, _ = primitive_from_conserved(
        reference[0], reference[1], reference[2], eos, args.energy_floor
    )
    relative_error = float(
        (
            torch.max(torch.abs(coarse_energy - reference_energy))
            / torch.clamp(torch.max(torch.abs(reference_energy)), min=1.0e-12)
        )
        .detach()
        .cpu()
    )
    cooled = float(torch.max(coarse_energy).detach().cpu()) < 1.0
    status = "PASS" if cooled and relative_error <= args.crossover_bjorken_tolerance else "FAIL"
    return {
        "test": "bjorken_crossover_convergence",
        "metric": "relative_substep_error",
        "value": relative_error,
        "tolerance": args.crossover_bjorken_tolerance,
        "status": status,
    }


def gubser_profile(
    tau: float,
    xx: torch.Tensor,
    yy: torch.Tensor,
    q: float,
    reference_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic ideal conformal Gubser flow (S. S. Gubser, Phys. Rev. D 82, 085027 (2010)).

    e(tau, r) ~ tau^(-4/3) [1 + 2 q^2 (tau^2 + r^2) + q^4 (tau^2 - r^2)^2]^(-4/3),
    v_r = 2 q^2 tau r / (1 + q^2 tau^2 + q^2 r^2),
    normalized so that e(reference_tau, r = 0) = 1. The flow solves the
    boost-invariant ideal-hydro equations for p = e/3 with transverse
    expansion, so it tests the coupling of the transverse fluxes to the
    Milne-frame source terms.
    """
    r2 = xx * xx + yy * yy
    q2 = q * q
    denominator = 1.0 + 2.0 * q2 * (tau * tau + r2) + q2 * q2 * (tau * tau - r2) ** 2
    normalization = reference_tau ** (4.0 / 3.0) * (1.0 + q2 * reference_tau * reference_tau) ** (8.0 / 3.0)
    energy_density = normalization / tau ** (4.0 / 3.0) / denominator ** (4.0 / 3.0)
    velocity_over_radius = 2.0 * q2 * tau / (1.0 + q2 * tau * tau + q2 * r2)
    return energy_density, velocity_over_radius * xx, velocity_over_radius * yy


def gubser_flow_error(
    grid_size: int,
    extent: float,
    tau0: float,
    tau1: float,
    q: float,
    cfl: float,
    eos: EquationOfState,
    energy_floor: float,
    device: torch.device,
) -> float:
    """Relative L1 error of e(tau1) after evolving the analytic Gubser flow from tau0."""
    xx, yy, dx = make_grid(grid_size, extent, device)
    energy_density, velocity_x, velocity_y = gubser_profile(tau0, xx, yy, q, tau0)
    state = conserved_from_primitive(energy_density + energy_floor, velocity_x, velocity_y, eos)
    steps = max(1, math.ceil((tau1 - tau0) / (cfl * dx)))
    dt = (tau1 - tau0) / steps
    tau = tau0
    for _ in range(steps):
        state = rusanov_step(state, dx, dt, eos, energy_floor)
        state = apply_bjorken_cooling(state, tau, dt, eos, energy_floor)
        tau += dt
    numerical, _, _, _ = primitive_from_conserved(state[0], state[1], state[2], eos, energy_floor)
    exact, _, _ = gubser_profile(tau1, xx, yy, q, tau0)
    region = exact > 1.0e-3 * torch.max(exact)
    error = torch.sum(torch.abs(numerical - exact)[region]) / torch.sum(exact[region])
    return float(error.detach().cpu())


def gubser_flow_test(args: argparse.Namespace, device: torch.device) -> dict[str, float | str]:
    eos = build_eos(args)
    row: dict[str, float | str] = {
        "test": "gubser_flow",
        "metric": "relative_l1_energy_error",
        "value": 0.0,
        "tolerance": args.gubser_tolerance,
        "status": "SKIP",
    }
    if eos.kind != "conformal" or abs(eos.speed_of_sound2 - 1.0 / 3.0) > 1.0e-9:
        return row
    fine_grid = args.gubser_grid_size
    coarse_grid = (fine_grid - 1) // 2 + 1
    errors = [
        gubser_flow_error(
            grid_size,
            args.gubser_extent,
            args.gubser_tau0,
            args.gubser_tau1,
            args.gubser_q,
            args.cfl,
            eos,
            args.energy_floor,
            device,
        )
        for grid_size in (coarse_grid, fine_grid)
    ]
    # The Rusanov update is first order, so halving dx should roughly halve the
    # error. A wrong source term leaves a resolution-independent error instead.
    order = math.log2(errors[0] / max(errors[1], 1.0e-300))
    passed = errors[1] <= args.gubser_tolerance and order >= args.gubser_min_order
    row.update(
        {
            "value": errors[1],
            "status": "PASS" if passed else "FAIL",
            "coarse_value": errors[0],
            "convergence_order": order,
        }
    )
    return row


def run_validation(args: argparse.Namespace) -> list[dict[str, float | str]]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return [
        uniform_rest_test(args, device),
        uniform_boost_test(args, device),
        gaussian_energy_test(args, device),
        bjorken_uniform_cooling_test(args, device),
        bjorken_crossover_convergence_test(args, device),
        gubser_flow_test(args, device),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the ideal hydrodynamics finite-volume update.")
    parser.add_argument("--grid-size", type=int, default=80, help="2D transverse grid size.")
    parser.add_argument("--extent", type=float, default=12.0, help="Half-width of the transverse plane.")
    parser.add_argument("--steps", type=int, default=40, help="Validation time steps.")
    parser.add_argument("--cfl", type=float, default=0.25, help="Courant factor.")
    add_eos_arguments(parser)
    parser.add_argument("--energy-floor", type=float, default=1.0e-8, help="Minimum energy density.")
    parser.add_argument("--tau0", type=float, default=0.6, help="Initial proper time for Bjorken cooling validation.")
    parser.add_argument("--boost-velocity", type=float, default=0.2, help="Uniform boost velocity for the regression test.")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0, help="Gaussian fireball width.")
    parser.add_argument("--uniform-tolerance", type=float, default=1.0e-6, help="Tolerance for uniform exact-solution tests.")
    parser.add_argument(
        "--gaussian-energy-tolerance",
        type=float,
        default=1.0e-4,
        help="Relative energy-conservation tolerance (boundary outflow included) for the Gaussian fireball.",
    )
    parser.add_argument(
        "--bjorken-tolerance",
        type=float,
        default=1.0e-5,
        help="Relative tolerance for the uniform Bjorken cooling analytic solution.",
    )
    parser.add_argument(
        "--crossover-bjorken-tolerance",
        type=float,
        default=5.0e-3,
        help="Coarse/fine relative tolerance for crossover Bjorken cooling.",
    )
    parser.add_argument("--gubser-grid-size", type=int, default=161, help="Fine grid size of the Gubser-flow test (odd).")
    parser.add_argument("--gubser-extent", type=float, default=8.0, help="Half-width of the Gubser-flow grid in fm.")
    parser.add_argument("--gubser-q", type=float, default=1.0, help="Gubser-flow scale q in 1/fm.")
    parser.add_argument("--gubser-tau0", type=float, default=1.0, help="Initial proper time of the Gubser-flow test.")
    parser.add_argument("--gubser-tau1", type=float, default=2.0, help="Final proper time of the Gubser-flow test.")
    parser.add_argument(
        "--gubser-tolerance",
        type=float,
        default=0.15,
        help="Relative L1 energy-density tolerance on the fine Gubser grid.",
    )
    parser.add_argument(
        "--gubser-min-order",
        type=float,
        default=0.8,
        help="Minimum observed convergence order between the coarse and fine Gubser grids.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output", type=Path, default=Path("outputs/hydro_validation.csv"), help="Output CSV path.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = run_validation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "test",
        "metric",
        "value",
        "tolerance",
        "status",
        "final_momentum_norm",
        "boundary_outflow_fraction",
        "coarse_value",
        "convergence_order",
    ]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['status']} {row['test']}: {row['metric']}={row['value']:.6g}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)
    print(f"Saved hydro validation report: {args.output}")


if __name__ == "__main__":
    main()
