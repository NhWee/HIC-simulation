import argparse
import csv
import math
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
        r"  .\run_ideal_hydro_2d.ps1"
    ) from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hic.eos import EquationOfState, add_eos_arguments, build_eos
from hic.glauber_ensemble import build_initial_source, eccentricity_n, rms_radius
from hic.nuclear_collision_2d import make_grid


def primitive_from_conserved(
    energy: torch.Tensor,
    momentum_x: torch.Tensor,
    momentum_y: torch.Tensor,
    eos: EquationOfState,
    energy_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    energy = torch.clamp(energy, min=energy_floor)
    momentum = torch.sqrt(momentum_x * momentum_x + momentum_y * momentum_y)
    small = momentum < 1.0e-12

    if eos.kind == "conformal" and eos.speed_of_sound2 > 0.0:
        c = eos.speed_of_sound2
        discriminant = (1.0 + c) * (1.0 + c) * energy * energy - 4.0 * c * momentum * momentum
        discriminant = torch.clamp(discriminant, min=0.0)
        velocity = 2.0 * momentum / ((1.0 + c) * energy + torch.sqrt(discriminant))
        velocity = torch.where(small, torch.zeros_like(velocity), velocity)
        velocity = torch.clamp(velocity, min=0.0, max=0.999)
    else:
        # For a tabulated or crossover EoS solve
        #   v = |M| / (T00 + p(e)),  e = T00 - |M| v
        # by fixed-point iteration. Using a constant sound speed here makes the
        # recovered primitive state inconsistent with the pressure flux.
        pressure_guess = eos.pressure(energy)
        velocity = torch.clamp(momentum / torch.clamp(energy + pressure_guess, min=energy_floor), 0.0, 0.999)
        for _ in range(12):
            energy_guess = torch.clamp(energy - momentum * velocity, min=energy_floor)
            pressure_guess = eos.pressure(energy_guess)
            updated_velocity = torch.clamp(
                momentum / torch.clamp(energy + pressure_guess, min=energy_floor),
                min=0.0,
                max=0.999,
            )
            velocity = 0.5 * (velocity + updated_velocity)
        velocity = torch.where(small, torch.zeros_like(velocity), velocity)

    energy_density = torch.clamp(energy - momentum * velocity, min=energy_floor)
    pressure = eos.pressure(energy_density)
    velocity_x = torch.where(small, torch.zeros_like(momentum_x), velocity * momentum_x / torch.clamp(momentum, min=1.0e-12))
    velocity_y = torch.where(small, torch.zeros_like(momentum_y), velocity * momentum_y / torch.clamp(momentum, min=1.0e-12))
    return energy_density, pressure, velocity_x, velocity_y


def conserved_from_primitive(
    energy_density: torch.Tensor,
    velocity_x: torch.Tensor,
    velocity_y: torch.Tensor,
    eos: EquationOfState,
) -> torch.Tensor:
    velocity2 = velocity_x * velocity_x + velocity_y * velocity_y
    gamma2 = 1.0 / torch.clamp(1.0 - velocity2, min=1.0e-8)
    pressure = eos.pressure(energy_density)
    enthalpy_gamma2 = (energy_density + pressure) * gamma2
    return torch.stack(
        (
            enthalpy_gamma2 - pressure,
            enthalpy_gamma2 * velocity_x,
            enthalpy_gamma2 * velocity_y,
        )
    )


def apply_bjorken_cooling(
    state: torch.Tensor,
    tau: float,
    dt: float,
    eos: EquationOfState,
    energy_floor: float,
    max_log_step: float = 0.02,
) -> torch.Tensor:
    """Integrate the boost-invariant (Milne) source terms from tau to tau + dt.

    With u^eta = 0 the longitudinal expansion adds, on top of the transverse fluxes,

        d(T^tt)/dtau = -(T^tt + p) / tau,     d(T^ti)/dtau = -T^ti / tau,

    where p = tau^2 T^etaeta is the local pressure. The momentum equation is solved exactly
    (tau * T^ti is constant). The energy equation is integrated with classical RK4 in
    s = ln(tau), sub-stepped so that each step spans at most ``max_log_step`` in s, while the
    pressure is recovered from the full (T^tt, T^ti) state at every stage. For a fluid at
    rest this reduces to de/dtau = -(e + p)/tau.
    """
    start_tau = max(tau, 1.0e-8)
    end_tau = max(tau + dt, 1.0e-8)
    total_log = math.log(end_tau / start_tau)
    substeps = max(1, math.ceil(abs(total_log) / max_log_step))
    step = total_log / substeps
    momentum_x0, momentum_y0 = state[1], state[2]

    def energy_rate(energy: torch.Tensor, log_tau: float) -> torch.Tensor:
        scale = math.exp(-log_tau)  # tau0 / tau, from tau * T^ti = const
        energy = torch.clamp(energy, min=energy_floor)
        _, pressure, _, _ = primitive_from_conserved(
            energy, momentum_x0 * scale, momentum_y0 * scale, eos, energy_floor
        )
        return -(energy + pressure)

    energy = state[0]
    for index in range(substeps):
        log_tau = index * step
        k1 = energy_rate(energy, log_tau)
        k2 = energy_rate(energy + 0.5 * step * k1, log_tau + 0.5 * step)
        k3 = energy_rate(energy + 0.5 * step * k2, log_tau + 0.5 * step)
        k4 = energy_rate(energy + step * k3, log_tau + step)
        energy = energy + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    scale = start_tau / end_tau
    return torch.stack(
        (
            torch.clamp(energy, min=energy_floor),
            momentum_x0 * scale,
            momentum_y0 * scale,
        )
    )


def fluxes(
    state: torch.Tensor,
    eos: EquationOfState,
    energy_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    energy, momentum_x, momentum_y = state
    energy_density, pressure, velocity_x, velocity_y = primitive_from_conserved(
        energy, momentum_x, momentum_y, eos, energy_floor
    )
    flux_x = torch.stack(
        (
            momentum_x,
            momentum_x * velocity_x + pressure,
            momentum_y * velocity_x,
        )
    )
    flux_y = torch.stack(
        (
            momentum_y,
            momentum_x * velocity_y,
            momentum_y * velocity_y + pressure,
        )
    )
    return flux_x, flux_y, velocity_x, velocity_y


def rusanov_step(
    state: torch.Tensor,
    dx: float,
    dt: float,
    eos: EquationOfState,
    energy_floor: float,
) -> torch.Tensor:
    flux_x, flux_y, velocity_x, velocity_y = fluxes(state, eos, energy_floor)
    sound_speed = eos.max_sound_speed()

    left = state[:, :, :-1]
    right = state[:, :, 1:]
    flux_left = flux_x[:, :, :-1]
    flux_right = flux_x[:, :, 1:]
    abs_velocity_x = torch.abs(velocity_x)
    signal_x = (abs_velocity_x + sound_speed) / (1.0 + abs_velocity_x * sound_speed)
    speed_x = torch.maximum(signal_x[:, :-1], signal_x[:, 1:])
    interface_x = 0.5 * (flux_left + flux_right) - 0.5 * speed_x.unsqueeze(0) * (right - left)

    bottom = state[:, :-1, :]
    top = state[:, 1:, :]
    flux_bottom = flux_y[:, :-1, :]
    flux_top = flux_y[:, 1:, :]
    abs_velocity_y = torch.abs(velocity_y)
    signal_y = (abs_velocity_y + sound_speed) / (1.0 + abs_velocity_y * sound_speed)
    speed_y = torch.maximum(signal_y[:-1, :], signal_y[1:, :])
    interface_y = 0.5 * (flux_bottom + flux_top) - 0.5 * speed_y.unsqueeze(0) * (top - bottom)

    next_state = state.clone()
    next_state[:, :, 1:-1] -= dt / dx * (interface_x[:, :, 1:] - interface_x[:, :, :-1])
    next_state[:, 1:-1, :] -= dt / dx * (interface_y[:, 1:, :] - interface_y[:, :-1, :])
    next_state[0] = torch.clamp(next_state[0], min=energy_floor)
    return next_state


def laplacian_no_wrap(field: torch.Tensor, dx: float) -> torch.Tensor:
    laplacian = torch.zeros_like(field)
    laplacian[1:-1, 1:-1] = (
        field[1:-1, :-2]
        + field[1:-1, 2:]
        + field[:-2, 1:-1]
        + field[2:, 1:-1]
        - 4.0 * field[1:-1, 1:-1]
    ) / (dx * dx)
    return laplacian


def apply_shear_viscous_momentum_diffusion(
    state: torch.Tensor,
    dx: float,
    dt: float,
    shear_viscosity: float,
) -> torch.Tensor:
    if shear_viscosity <= 0.0:
        return state
    next_state = state.clone()
    next_state[1] = next_state[1] + shear_viscosity * laplacian_no_wrap(state[1], dx) * dt
    next_state[2] = next_state[2] + shear_viscosity * laplacian_no_wrap(state[2], dx) * dt
    return next_state


def conserved_totals(state: torch.Tensor, dx: float) -> tuple[float, float, float]:
    area = dx * dx
    total_energy = float((torch.sum(state[0]) * area).detach().cpu())
    total_px = float((torch.sum(state[1]) * area).detach().cpu())
    total_py = float((torch.sum(state[2]) * area).detach().cpu())
    return total_energy, total_px, total_py


def collect_freezeout_crossings(
    previous_energy_density: torch.Tensor,
    current_energy_density: torch.Tensor,
    pressure: torch.Tensor,
    temperature: torch.Tensor,
    velocity_x: torch.Tensor,
    velocity_y: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    previous_tau: float,
    tau: float,
    dx: float,
    freezeout_energy: float,
) -> list[dict[str, float]]:
    crossed = (previous_energy_density >= freezeout_energy) & (current_energy_density < freezeout_energy)
    indices = torch.nonzero(crossed, as_tuple=False)
    grad_y, grad_x = torch.gradient(current_energy_density, spacing=dx)
    rows = []
    for row_index, col_index in indices.detach().cpu().tolist():
        previous_e = float(previous_energy_density[row_index, col_index].detach().cpu())
        current_e = float(current_energy_density[row_index, col_index].detach().cpu())
        fraction = (previous_e - freezeout_energy) / max(previous_e - current_e, 1.0e-12)
        fraction = min(max(fraction, 0.0), 1.0)
        freezeout_tau = previous_tau + fraction * (tau - previous_tau)
        normal_x = -float(grad_x[row_index, col_index].detach().cpu())
        normal_y = -float(grad_y[row_index, col_index].detach().cpu())
        normal_norm = max((1.0 + normal_x * normal_x + normal_y * normal_y) ** 0.5, 1.0e-12)
        cell_area = dx * dx
        rows.append(
            {
                "tau": freezeout_tau,
                "x": float(xx[row_index, col_index].detach().cpu()),
                "y": float(yy[row_index, col_index].detach().cpu()),
                "energy_density": freezeout_energy,
                "post_step_energy_density": current_e,
                "pressure": float(pressure[row_index, col_index].detach().cpu()),
                "temperature": float(temperature[row_index, col_index].detach().cpu()),
                "velocity_x": float(velocity_x[row_index, col_index].detach().cpu()),
                "velocity_y": float(velocity_y[row_index, col_index].detach().cpu()),
                "d_sigma_tau": cell_area / normal_norm,
                "d_sigma_x": cell_area * normal_x / normal_norm,
                "d_sigma_y": cell_area * normal_y / normal_norm,
            }
        )
    return rows


def run_hydro(args: argparse.Namespace) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    xx, yy, dx = make_grid(args.grid_size, args.extent, device)
    eos = build_eos(args)
    rng = np.random.default_rng(args.seed)
    source, n_part, n_coll = build_initial_source(
        xx=xx,
        yy=yy,
        impact_parameter=args.impact_parameter,
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

    state = conserved_from_primitive(torch.clamp(source, min=args.energy_floor), torch.zeros_like(source), torch.zeros_like(source), eos)
    dt = args.cfl * dx / max(1.0, eos.max_sound_speed())
    rows = []
    freezeout_rows = []
    previous_energy_density, _, _, _ = primitive_from_conserved(state[0], state[1], state[2], eos, args.energy_floor)

    print(f"Initial condition: N_part={n_part}, N_coll={n_coll}, dt={dt:.5f}")
    for step in range(args.steps + 1):
        if step % args.sample_every == 0 or step == args.steps:
            energy_density, _, velocity_x, velocity_y = primitive_from_conserved(
                state[0], state[1], state[2], eos, args.energy_floor
            )
            total_energy, total_px, total_py = conserved_totals(state, dx)
            rows.append(
                {
                    "step": step,
                    "time": args.tau0 + step * dt,
                    "bjorken_expansion": int(args.bjorken_expansion),
                    "eos": eos.kind,
                    "n_part": n_part,
                    "n_coll": n_coll,
                    "total_energy": total_energy,
                    "total_px": total_px,
                    "total_py": total_py,
                    "max_energy_density": float(torch.max(energy_density).detach().cpu()),
                    "max_velocity": float(torch.max(torch.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)).detach().cpu()),
                    "eps2": eccentricity_n(energy_density, xx, yy, 2),
                    "eps3": eccentricity_n(energy_density, xx, yy, 3),
                    "rms_radius": rms_radius(energy_density, xx, yy),
                }
            )
        if step < args.steps:
            state = rusanov_step(state, dx, dt, eos, args.energy_floor)
            state = apply_shear_viscous_momentum_diffusion(state, dx, dt, args.shear_viscosity)
            if args.bjorken_expansion:
                state = apply_bjorken_cooling(
                    state,
                    tau=args.tau0 + step * dt,
                    dt=dt,
                    eos=eos,
                    energy_floor=args.energy_floor,
                )
            if args.freezeout_output is not None:
                energy_density, pressure, velocity_x, velocity_y = primitive_from_conserved(
                    state[0], state[1], state[2], eos, args.energy_floor
                )
                temperature = eos.temperature(energy_density, args.freezeout_temperature)
                freezeout_rows.extend(
                    collect_freezeout_crossings(
                        previous_energy_density,
                        energy_density,
                        pressure,
                        temperature,
                        velocity_x,
                        velocity_y,
                        xx,
                        yy,
                        previous_tau=args.tau0 + step * dt,
                        tau=args.tau0 + (step + 1) * dt,
                        dx=dx,
                        freezeout_energy=args.freezeout_energy,
                    )
                )
                previous_energy_density = energy_density

    return rows, freezeout_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a conservative 2+1D ideal hydrodynamics prototype.")
    parser.add_argument("--grid-size", type=int, default=120, help="2D transverse grid size.")
    parser.add_argument("--extent", type=float, default=12.0, help="Half-width of the transverse plane in fm-like units.")
    parser.add_argument("--steps", type=int, default=120, help="Hydrodynamic time steps.")
    parser.add_argument("--sample-every", type=int, default=5, help="Write diagnostics every N steps.")
    parser.add_argument("--cfl", type=float, default=0.25, help="Courant factor.")
    add_eos_arguments(parser)
    parser.add_argument("--energy-floor", type=float, default=1.0e-8, help="Minimum energy density.")
    parser.add_argument(
        "--shear-viscosity",
        type=float,
        default=0.0,
        help="Prototype momentum-diffusion coefficient. This is not full Israel-Stewart viscous hydrodynamics.",
    )
    parser.add_argument("--tau0", type=float, default=0.6, help="Initial proper time for boost-invariant evolution.")
    parser.add_argument(
        "--bjorken-expansion",
        action="store_true",
        help="Apply boost-invariant longitudinal Bjorken cooling after each transverse hydro step.",
    )
    parser.add_argument("--impact-parameter", type=float, default=7.0, help="Impact parameter.")
    parser.add_argument("--mass-number", type=int, default=197, help="Nuclear mass number.")
    parser.add_argument("--radius", type=float, default=6.4, help="Nuclear radius.")
    parser.add_argument("--skin-depth", type=float, default=0.55, help="Woods-Saxon surface diffuseness.")
    parser.add_argument("--nucleon-width", type=float, default=0.45, help="Gaussian nucleon width.")
    parser.add_argument("--inelastic-cross-section", type=float, default=4.2, help="Nucleon-nucleon cross section in fm^2.")
    parser.add_argument("--binary-fraction", type=float, default=0.16, help="Wounded/binary source mixing fraction.")
    parser.add_argument("--norm", type=float, default=2.5, help="Overall source normalization.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output", type=Path, default=Path("outputs/ideal_hydro_2d.csv"), help="Output CSV path.")
    parser.add_argument("--freezeout-energy", type=float, default=0.35, help="Energy-density threshold for freeze-out candidates.")
    parser.add_argument("--freezeout-temperature", type=float, default=0.145, help="Fallback freeze-out temperature for non-tabulated EoS.")
    parser.add_argument("--freezeout-output", type=Path, default=None, help="Optional output CSV for freeze-out candidate cells.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, freezeout_rows = run_hydro(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "time",
        "bjorken_expansion",
        "eos",
        "n_part",
        "n_coll",
        "total_energy",
        "total_px",
        "total_py",
        "max_energy_density",
        "max_velocity",
        "eps2",
        "eps3",
        "rms_radius",
    ]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ideal hydro diagnostics: {args.output}")

    if args.freezeout_output is not None:
        args.freezeout_output.parent.mkdir(parents=True, exist_ok=True)
        freezeout_fieldnames = [
            "tau",
            "x",
            "y",
            "energy_density",
            "post_step_energy_density",
            "pressure",
            "temperature",
            "velocity_x",
            "velocity_y",
            "d_sigma_tau",
            "d_sigma_x",
            "d_sigma_y",
        ]
        with args.freezeout_output.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=freezeout_fieldnames)
            writer.writeheader()
            writer.writerows(freezeout_rows)
        print(f"Saved freeze-out candidates: {args.freezeout_output} ({len(freezeout_rows)} cells)")


if __name__ == "__main__":
    main()
