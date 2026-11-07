import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as torch_functional
except ModuleNotFoundError as error:
    raise SystemExit("PyTorch is required. Run .\\setup.ps1 first.") from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hic.eos import EquationOfState, read_eos_table
from hic.glauber_ensemble import eccentricity_n
from hic.ideal_hydro_2d import (
    apply_bjorken_cooling,
    conserved_from_primitive,
    primitive_from_conserved,
    rusanov_step,
)
from hic.nuclear_collision_2d import make_grid, tag_glauber_collisions


SPECIES_NAMES = np.asarray(("pion", "kaon", "proton"))
SPECIES_MASSES = np.asarray((0.13957, 0.49368, 0.93827), dtype=np.float64)
SPECIES_DEGENERACIES = np.asarray((3.0, 4.0, 4.0), dtype=np.float64)


@dataclass
class EventGeometry:
    nucleus_a_xyz: np.ndarray
    nucleus_b_xyz: np.ndarray
    participant_a: np.ndarray
    participant_b: np.ndarray
    collision_centers: np.ndarray
    source: torch.Tensor
    n_part: int
    n_coll: int


@dataclass
class PreEquilibriumResult:
    tau: np.ndarray
    energy_density: np.ndarray
    momentum_x: np.ndarray
    momentum_y: np.ndarray
    stress_xx: np.ndarray
    stress_xy: np.ndarray
    stress_yy: np.ndarray
    landau_energy_density: np.ndarray
    effective_temperature: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    landau_residual: np.ndarray
    final_energy_density: torch.Tensor
    final_velocity_x: torch.Tensor
    final_velocity_y: torch.Tensor


@dataclass
class HydroResult:
    tau: np.ndarray
    energy_density: np.ndarray
    temperature: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    max_temperature: np.ndarray
    max_velocity: np.ndarray
    total_energy_per_rapidity: np.ndarray
    eccentricity2: np.ndarray
    freezeout_tau: np.ndarray
    freezeout_velocity_x: np.ndarray
    freezeout_velocity_y: np.ndarray
    freezeout_energy: np.ndarray


@dataclass
class SurfaceResult:
    tau: np.ndarray
    x: np.ndarray
    y: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    d_sigma_tau: np.ndarray
    d_sigma_x: np.ndarray
    d_sigma_y: np.ndarray


@dataclass
class ParticleResult:
    species: np.ndarray
    mass: np.ndarray
    tau: np.ndarray
    eta_s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    energy: np.ndarray
    momentum_longitudinal: np.ndarray
    momentum_x: np.ndarray
    momentum_y: np.ndarray
    cooper_frye_weight: np.ndarray
    mean_pt: float
    v2: float
    v3: float


def sample_woods_saxon_3d(
    mass_number: int,
    radius: float,
    skin_depth: float,
    rng: np.random.Generator,
) -> np.ndarray:
    max_radius = radius + 8.0 * skin_depth
    positions: list[list[float]] = []
    while len(positions) < mass_number:
        batch = max(1024, 10 * (mass_number - len(positions)))
        xyz = rng.uniform(-max_radius, max_radius, size=(batch, 3))
        radius_values = np.linalg.norm(xyz, axis=1)
        density = 1.0 / (1.0 + np.exp((radius_values - radius) / max(skin_depth, 1.0e-8)))
        accepted = (radius_values <= max_radius) & (rng.random(batch) < density)
        positions.extend(xyz[accepted].tolist())
    result = np.asarray(positions[:mass_number], dtype=np.float32)
    result -= np.mean(result, axis=0, keepdims=True)
    return result


def deposit_gaussians(
    xx: torch.Tensor,
    yy: torch.Tensor,
    points_x: np.ndarray,
    points_y: np.ndarray,
    width: float,
) -> torch.Tensor:
    if points_x.size == 0:
        return torch.zeros_like(xx)
    sigma2 = max(width * width, 1.0e-8)
    normalization = 1.0 / (2.0 * math.pi * sigma2)
    result = torch.zeros_like(xx)
    for start in range(0, points_x.size, 64):
        end = min(start + 64, points_x.size)
        px = torch.as_tensor(points_x[start:end], dtype=xx.dtype, device=xx.device).view(-1, 1, 1)
        py = torch.as_tensor(points_y[start:end], dtype=yy.dtype, device=yy.device).view(-1, 1, 1)
        distance2 = (xx.unsqueeze(0) - px) ** 2 + (yy.unsqueeze(0) - py) ** 2
        result += torch.exp(-distance2 / (2.0 * sigma2)).sum(dim=0) * normalization
    return result


def build_event_geometry(
    args: argparse.Namespace,
    xx: torch.Tensor,
    yy: torch.Tensor,
    rng: np.random.Generator,
) -> EventGeometry:
    nucleus_a = sample_woods_saxon_3d(args.mass_number, args.radius, args.skin_depth, rng)
    nucleus_b = sample_woods_saxon_3d(args.mass_number, args.radius, args.skin_depth, rng)
    participant_a, participant_b, collision_x, collision_y = tag_glauber_collisions(
        nucleus_a[:, 0],
        nucleus_a[:, 1],
        nucleus_b[:, 0],
        nucleus_b[:, 1],
        args.impact_parameter,
        args.inelastic_cross_section,
    )

    participant_a_x = nucleus_a[participant_a, 0]
    participant_a_y = nucleus_a[participant_a, 1] - 0.5 * args.impact_parameter
    participant_b_x = nucleus_b[participant_b, 0]
    participant_b_y = nucleus_b[participant_b, 1] + 0.5 * args.impact_parameter

    wounded_a = deposit_gaussians(xx, yy, participant_a_x, participant_a_y, args.nucleon_width)
    wounded_b = deposit_gaussians(xx, yy, participant_b_x, participant_b_y, args.nucleon_width)
    binary = deposit_gaussians(xx, yy, collision_x, collision_y, args.nucleon_width)
    source = 0.5 * (1.0 - args.binary_fraction) * (wounded_a + wounded_b) + args.binary_fraction * binary
    source_max = float(torch.max(source).detach().cpu())
    if source_max <= 0.0:
        raise RuntimeError("The sampled Glauber event has no binary collisions. Change the seed or impact parameter.")
    source = source / source_max * args.initial_peak_energy

    collision_centers = np.column_stack((collision_x, collision_y)).astype(np.float32)
    return EventGeometry(
        nucleus_a_xyz=nucleus_a,
        nucleus_b_xyz=nucleus_b,
        participant_a=participant_a,
        participant_b=participant_b,
        collision_centers=collision_centers,
        source=source,
        n_part=int(participant_a.sum() + participant_b.sum()),
        n_coll=int(collision_x.size),
    )


def build_table_eos(path: Path) -> EquationOfState:
    energy, pressure, sound_speed2, temperature = read_eos_table(path)
    if not temperature:
        raise ValueError(f"The event pipeline needs a temperature column in the EoS table: {path}")
    return EquationOfState(
        kind="table",
        speed_of_sound2=1.0 / 3.0,
        low_speed_of_sound2=min(sound_speed2),
        high_speed_of_sound2=max(sound_speed2),
        transition_energy=1.0,
        transition_width=0.25,
        table_energy=energy,
        table_pressure=pressure,
        table_sound_speed2=sound_speed2,
        table_temperature=temperature,
    )


def physical_temperature(eos: EquationOfState, energy_density: torch.Tensor) -> torch.Tensor:
    """Temperature from the tabulated EoS, including its consistent extrapolation.

    Beyond the table the EoS continues p(e) linearly and T follows
    T ~ (e + p)^(k / (1 + k)), so no separate high-energy rule is needed.
    """
    return eos.temperature(energy_density)


def landau_match_transverse_tensor(
    energy: torch.Tensor,
    momentum_x: torch.Tensor,
    momentum_y: torch.Tensor,
    stress_xx: torch.Tensor,
    stress_xy: torch.Tensor,
    stress_yy: torch.Tensor,
    energy_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the timelike eigenvector of T^mu_nu in the transverse plane."""
    active = energy > max(100.0 * energy_floor, 1.0e-7)
    lower_energy = torch.zeros_like(energy)
    upper_energy = torch.clamp(energy, min=energy_floor)
    for _ in range(36):
        trial_energy = 0.5 * (lower_energy + upper_energy)
        coefficient_xx = stress_xx + trial_energy
        coefficient_yy = stress_yy + trial_energy
        determinant = coefficient_xx * coefficient_yy - stress_xy.square()
        safe_determinant = torch.where(
            determinant.abs() > 1.0e-12,
            determinant,
            torch.full_like(determinant, 1.0e-12),
        )
        trial_velocity_x = (momentum_x * coefficient_yy - momentum_y * stress_xy) / safe_determinant
        trial_velocity_y = (momentum_y * coefficient_xx - momentum_x * stress_xy) / safe_determinant
        eigenvalue_error = trial_energy - (
            energy - momentum_x * trial_velocity_x - momentum_y * trial_velocity_y
        )
        upper_energy = torch.where(eigenvalue_error > 0.0, trial_energy, upper_energy)
        lower_energy = torch.where(eigenvalue_error > 0.0, lower_energy, trial_energy)

    rest_frame_energy = torch.clamp(0.5 * (lower_energy + upper_energy), min=energy_floor)
    coefficient_xx = stress_xx + rest_frame_energy
    coefficient_yy = stress_yy + rest_frame_energy
    determinant = coefficient_xx * coefficient_yy - stress_xy.square()
    safe_determinant = torch.where(
        determinant.abs() > 1.0e-12,
        determinant,
        torch.full_like(determinant, 1.0e-12),
    )
    velocity_x = (momentum_x * coefficient_yy - momentum_y * stress_xy) / safe_determinant
    velocity_y = (momentum_y * coefficient_xx - momentum_x * stress_xy) / safe_determinant

    velocity_x = torch.where(active, velocity_x, 0.0)
    velocity_y = torch.where(active, velocity_y, 0.0)
    rest_frame_energy = torch.where(active, rest_frame_energy, torch.clamp(energy, min=energy_floor))

    speed = torch.sqrt(velocity_x**2 + velocity_y**2)
    limiter = torch.clamp(0.999 / torch.clamp(speed, min=1.0e-12), max=1.0)
    velocity_x = velocity_x * limiter
    velocity_y = velocity_y * limiter
    rest_frame_energy = torch.clamp(rest_frame_energy, min=energy_floor)

    gamma = torch.rsqrt(torch.clamp(1.0 - velocity_x**2 - velocity_y**2, min=1.0e-8))
    u0 = gamma
    ux = gamma * velocity_x
    uy = gamma * velocity_y
    residual_0 = energy * u0 - momentum_x * ux - momentum_y * uy - rest_frame_energy * u0
    residual_x = momentum_x * u0 - stress_xx * ux - stress_xy * uy - rest_frame_energy * ux
    residual_y = momentum_y * u0 - stress_xy * ux - stress_yy * uy - rest_frame_energy * uy
    residual_scale = torch.clamp(torch.maximum(energy.abs(), rest_frame_energy.abs()), min=1.0e-6)
    residual = torch.sqrt(residual_0**2 + residual_x**2 + residual_y**2) / residual_scale
    residual = torch.where(active, residual, 0.0)
    return rest_frame_energy, velocity_x, velocity_y, residual


def evolve_pre_equilibrium(
    args: argparse.Namespace,
    source: torch.Tensor,
    eos: EquationOfState,
) -> PreEquilibriumResult:
    height, width = source.shape
    axis_y = torch.linspace(-1.0, 1.0, height, dtype=source.dtype, device=source.device)
    axis_x = torch.linspace(-1.0, 1.0, width, dtype=source.dtype, device=source.device)
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)

    angles = torch.arange(args.preeq_angles, dtype=source.dtype, device=source.device)
    angles = 2.0 * math.pi * angles / args.preeq_angles
    direction_x = torch.cos(angles)
    direction_y = torch.sin(angles)
    times = np.linspace(args.tau_initial, args.tau0, args.preeq_snapshots, dtype=np.float64)

    raw_energy: list[torch.Tensor] = []
    raw_momentum_x: list[torch.Tensor] = []
    raw_momentum_y: list[torch.Tensor] = []
    raw_stress_xx: list[torch.Tensor] = []
    raw_stress_xy: list[torch.Tensor] = []
    raw_stress_yy: list[torch.Tensor] = []
    source_batch = source.view(1, 1, height, width).expand(args.preeq_angles, -1, -1, -1)
    for tau in times:
        propagation_time = float(tau - args.tau_initial)
        normalized_shift = propagation_time / args.extent
        sample_grid = base_grid.unsqueeze(0).expand(args.preeq_angles, -1, -1, -1).clone()
        sample_grid[..., 0] -= normalized_shift * direction_x.view(-1, 1, 1)
        sample_grid[..., 1] -= normalized_shift * direction_y.view(-1, 1, 1)
        streams = torch_functional.grid_sample(
            source_batch,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        longitudinal_dilution = args.tau_initial / max(float(tau), args.tau_initial)
        energy = longitudinal_dilution * torch.mean(streams, dim=0)
        momentum_x = longitudinal_dilution * torch.mean(streams * direction_x.view(-1, 1, 1), dim=0)
        momentum_y = longitudinal_dilution * torch.mean(streams * direction_y.view(-1, 1, 1), dim=0)
        stress_xx = longitudinal_dilution * torch.mean(streams * direction_x.square().view(-1, 1, 1), dim=0)
        stress_xy = longitudinal_dilution * torch.mean(
            streams * (direction_x * direction_y).view(-1, 1, 1), dim=0
        )
        stress_yy = longitudinal_dilution * torch.mean(streams * direction_y.square().view(-1, 1, 1), dim=0)
        raw_energy.append(energy)
        raw_momentum_x.append(momentum_x)
        raw_momentum_y.append(momentum_y)
        raw_stress_xx.append(stress_xx)
        raw_stress_xy.append(stress_xy)
        raw_stress_yy.append(stress_yy)

    raw_matched_energy, _, _, _ = landau_match_transverse_tensor(
        raw_energy[-1],
        raw_momentum_x[-1],
        raw_momentum_y[-1],
        raw_stress_xx[-1],
        raw_stress_xy[-1],
        raw_stress_yy[-1],
        args.energy_floor,
    )
    scale = args.initial_peak_energy / max(float(torch.max(raw_matched_energy).detach().cpu()), 1.0e-12)

    energy_frames: list[np.ndarray] = []
    momentum_x_frames: list[np.ndarray] = []
    momentum_y_frames: list[np.ndarray] = []
    stress_xx_frames: list[np.ndarray] = []
    stress_xy_frames: list[np.ndarray] = []
    stress_yy_frames: list[np.ndarray] = []
    landau_energy_frames: list[np.ndarray] = []
    effective_temperature_frames: list[np.ndarray] = []
    velocity_x_frames: list[np.ndarray] = []
    velocity_y_frames: list[np.ndarray] = []
    residual_frames: list[np.ndarray] = []
    final_energy_density = torch.empty_like(source)
    final_velocity_x = torch.empty_like(source)
    final_velocity_y = torch.empty_like(source)
    tensor_frames = zip(
        raw_energy,
        raw_momentum_x,
        raw_momentum_y,
        raw_stress_xx,
        raw_stress_xy,
        raw_stress_yy,
    )
    for index, (energy, momentum_x, momentum_y, stress_xx, stress_xy, stress_yy) in enumerate(tensor_frames):
        scaled_energy = energy * scale
        scaled_momentum_x = momentum_x * scale
        scaled_momentum_y = momentum_y * scale
        scaled_stress_xx = stress_xx * scale
        scaled_stress_xy = stress_xy * scale
        scaled_stress_yy = stress_yy * scale
        matched_energy, velocity_x, velocity_y, residual = landau_match_transverse_tensor(
            scaled_energy,
            scaled_momentum_x,
            scaled_momentum_y,
            scaled_stress_xx,
            scaled_stress_xy,
            scaled_stress_yy,
            args.energy_floor,
        )
        effective_temperature = physical_temperature(eos, matched_energy)
        energy_frames.append(scaled_energy.detach().cpu().numpy().astype(np.float32))
        momentum_x_frames.append(scaled_momentum_x.detach().cpu().numpy().astype(np.float32))
        momentum_y_frames.append(scaled_momentum_y.detach().cpu().numpy().astype(np.float32))
        stress_xx_frames.append(scaled_stress_xx.detach().cpu().numpy().astype(np.float32))
        stress_xy_frames.append(scaled_stress_xy.detach().cpu().numpy().astype(np.float32))
        stress_yy_frames.append(scaled_stress_yy.detach().cpu().numpy().astype(np.float32))
        landau_energy_frames.append(matched_energy.detach().cpu().numpy().astype(np.float32))
        effective_temperature_frames.append(effective_temperature.detach().cpu().numpy().astype(np.float32))
        velocity_x_frames.append(velocity_x.detach().cpu().numpy().astype(np.float32))
        velocity_y_frames.append(velocity_y.detach().cpu().numpy().astype(np.float32))
        residual_frames.append(residual.detach().cpu().numpy().astype(np.float32))
        if index == len(raw_energy) - 1:
            final_energy_density = matched_energy
            final_velocity_x = velocity_x
            final_velocity_y = velocity_y

    print(
        f"Pre-equilibrium: tau={times[0]:.3f}->{times[-1]:.3f} fm/c, "
        f"angles={args.preeq_angles}, max preflow="
        f"{float(torch.max(torch.sqrt(final_velocity_x**2 + final_velocity_y**2)).detach().cpu()):.3f}c"
    )
    return PreEquilibriumResult(
        tau=times.astype(np.float32),
        energy_density=np.asarray(energy_frames, dtype=np.float32),
        momentum_x=np.asarray(momentum_x_frames, dtype=np.float32),
        momentum_y=np.asarray(momentum_y_frames, dtype=np.float32),
        stress_xx=np.asarray(stress_xx_frames, dtype=np.float32),
        stress_xy=np.asarray(stress_xy_frames, dtype=np.float32),
        stress_yy=np.asarray(stress_yy_frames, dtype=np.float32),
        landau_energy_density=np.asarray(landau_energy_frames, dtype=np.float32),
        effective_temperature=np.asarray(effective_temperature_frames, dtype=np.float32),
        velocity_x=np.asarray(velocity_x_frames, dtype=np.float32),
        velocity_y=np.asarray(velocity_y_frames, dtype=np.float32),
        landau_residual=np.asarray(residual_frames, dtype=np.float32),
        final_energy_density=final_energy_density,
        final_velocity_x=final_velocity_x,
        final_velocity_y=final_velocity_y,
    )


def evolve_hydrodynamics(
    args: argparse.Namespace,
    preequilibrium: PreEquilibriumResult,
    eos: EquationOfState,
    xx: torch.Tensor,
    yy: torch.Tensor,
    dx: float,
) -> HydroResult:
    state = conserved_from_primitive(
        torch.clamp(preequilibrium.final_energy_density, min=args.energy_floor),
        preequilibrium.final_velocity_x,
        preequilibrium.final_velocity_y,
        eos,
    )
    base_dt = args.cfl * dx
    tau = args.tau0
    next_snapshot_tau = tau

    freezeout_tau = torch.full_like(preequilibrium.final_energy_density, float("nan"))
    freezeout_velocity_x = torch.zeros_like(preequilibrium.final_energy_density)
    freezeout_velocity_y = torch.zeros_like(preequilibrium.final_energy_density)
    freezeout_energy = torch.zeros_like(preequilibrium.final_energy_density)

    previous_energy, _, previous_vx, previous_vy = primitive_from_conserved(
        state[0], state[1], state[2], eos, args.energy_floor
    )
    previous_temperature = physical_temperature(eos, previous_energy)
    initially_hot = previous_temperature >= args.freezeout_temperature

    tau_frames: list[float] = []
    energy_frames: list[np.ndarray] = []
    temperature_frames: list[np.ndarray] = []
    vx_frames: list[np.ndarray] = []
    vy_frames: list[np.ndarray] = []
    max_temperature_values: list[float] = []
    max_velocity_values: list[float] = []
    total_energy_values: list[float] = []
    eccentricity_values: list[float] = []

    def append_snapshot(
        snapshot_tau: float,
        energy_density: torch.Tensor,
        velocity_x: torch.Tensor,
        velocity_y: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        if tau_frames and abs(tau_frames[-1] - snapshot_tau) < 1.0e-7:
            return
        speed = torch.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
        tau_frames.append(snapshot_tau)
        energy_frames.append(energy_density.detach().cpu().numpy().astype(np.float32))
        temperature_frames.append(temperature.detach().cpu().numpy().astype(np.float32))
        vx_frames.append(velocity_x.detach().cpu().numpy().astype(np.float32))
        vy_frames.append(velocity_y.detach().cpu().numpy().astype(np.float32))
        max_temperature_values.append(float(torch.max(temperature).detach().cpu()))
        max_velocity_values.append(float(torch.max(speed).detach().cpu()))
        total_energy_values.append(float(snapshot_tau * torch.sum(state[0]).detach().cpu() * dx * dx))
        eccentricity_values.append(eccentricity_n(energy_density, xx, yy, 2))

    append_snapshot(tau, previous_energy, previous_vx, previous_vy, previous_temperature)
    next_snapshot_tau += args.snapshot_dt
    steps = 0

    while tau < args.tau_max:
        dt = min(base_dt, args.tau_max - tau)
        previous_tau = tau
        state = rusanov_step(state, dx, dt, eos, args.energy_floor)
        state = apply_bjorken_cooling(state, tau=tau, dt=dt, eos=eos, energy_floor=args.energy_floor)
        tau += dt
        steps += 1

        energy_density, _, velocity_x, velocity_y = primitive_from_conserved(
            state[0], state[1], state[2], eos, args.energy_floor
        )
        temperature = physical_temperature(eos, energy_density)

        not_recorded = torch.isnan(freezeout_tau)
        crossed = initially_hot & not_recorded & (previous_temperature >= args.freezeout_temperature) & (
            temperature < args.freezeout_temperature
        )
        denominator = torch.clamp(previous_temperature - temperature, min=1.0e-12)
        fraction = torch.clamp((previous_temperature - args.freezeout_temperature) / denominator, 0.0, 1.0)
        interpolated_tau = previous_tau + fraction * dt
        freezeout_tau = torch.where(crossed, interpolated_tau, freezeout_tau)
        freezeout_velocity_x = torch.where(
            crossed,
            previous_vx + fraction * (velocity_x - previous_vx),
            freezeout_velocity_x,
        )
        freezeout_velocity_y = torch.where(
            crossed,
            previous_vy + fraction * (velocity_y - previous_vy),
            freezeout_velocity_y,
        )
        freezeout_energy = torch.where(
            crossed,
            previous_energy + fraction * (energy_density - previous_energy),
            freezeout_energy,
        )

        if tau + 1.0e-8 >= next_snapshot_tau:
            append_snapshot(tau, energy_density, velocity_x, velocity_y, temperature)
            while next_snapshot_tau <= tau + 1.0e-8:
                next_snapshot_tau += args.snapshot_dt

        previous_energy = energy_density
        previous_temperature = temperature
        previous_vx = velocity_x
        previous_vy = velocity_y

        if tau > args.tau0 + 0.5 and float(torch.max(temperature).detach().cpu()) < args.freezeout_temperature:
            append_snapshot(tau, energy_density, velocity_x, velocity_y, temperature)
            break

    print(
        f"Hydro: steps={steps}, snapshots={len(tau_frames)}, "
        f"tau={tau_frames[0]:.3f}->{tau_frames[-1]:.3f} fm/c"
    )
    return HydroResult(
        tau=np.asarray(tau_frames, dtype=np.float32),
        energy_density=np.asarray(energy_frames, dtype=np.float32),
        temperature=np.asarray(temperature_frames, dtype=np.float32),
        velocity_x=np.asarray(vx_frames, dtype=np.float32),
        velocity_y=np.asarray(vy_frames, dtype=np.float32),
        max_temperature=np.asarray(max_temperature_values, dtype=np.float32),
        max_velocity=np.asarray(max_velocity_values, dtype=np.float32),
        total_energy_per_rapidity=np.asarray(total_energy_values, dtype=np.float32),
        eccentricity2=np.asarray(eccentricity_values, dtype=np.float32),
        freezeout_tau=freezeout_tau.detach().cpu().numpy().astype(np.float32),
        freezeout_velocity_x=freezeout_velocity_x.detach().cpu().numpy().astype(np.float32),
        freezeout_velocity_y=freezeout_velocity_y.detach().cpu().numpy().astype(np.float32),
        freezeout_energy=freezeout_energy.detach().cpu().numpy().astype(np.float32),
    )


def masked_surface_gradient(values: np.ndarray, valid: np.ndarray, dx: float, axis: int) -> np.ndarray:
    gradient = np.zeros_like(values, dtype=np.float64)
    if axis == 1:
        center = valid[:, 1:-1]
        both = center & valid[:, :-2] & valid[:, 2:]
        right = center & ~valid[:, :-2] & valid[:, 2:]
        left = center & valid[:, :-2] & ~valid[:, 2:]
        middle = gradient[:, 1:-1]
        middle[both] = ((values[:, 2:] - values[:, :-2]) / (2.0 * dx))[both]
        middle[right] = ((values[:, 2:] - values[:, 1:-1]) / dx)[right]
        middle[left] = ((values[:, 1:-1] - values[:, :-2]) / dx)[left]
    else:
        center = valid[1:-1, :]
        both = center & valid[:-2, :] & valid[2:, :]
        top = center & ~valid[:-2, :] & valid[2:, :]
        bottom = center & valid[:-2, :] & ~valid[2:, :]
        middle = gradient[1:-1, :]
        middle[both] = ((values[2:, :] - values[:-2, :]) / (2.0 * dx))[both]
        middle[top] = ((values[2:, :] - values[1:-1, :]) / dx)[top]
        middle[bottom] = ((values[1:-1, :] - values[:-2, :]) / dx)[bottom]
    return gradient


def build_freezeout_surface(hydro: HydroResult, axis: np.ndarray, dx: float) -> SurfaceResult:
    valid = np.isfinite(hydro.freezeout_tau)
    if int(valid.sum()) == 0:
        raise RuntimeError("No cells crossed the requested freeze-out temperature.")
    safe_tau = np.where(valid, hydro.freezeout_tau, 0.0).astype(np.float64)
    grad_x = masked_surface_gradient(safe_tau, valid, dx, axis=1)
    grad_y = masked_surface_gradient(safe_tau, valid, dx, axis=0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    tau = safe_tau[valid]
    area = dx * dx
    d_sigma_tau = tau * area
    d_sigma_x = -tau * grad_x[valid] * area
    d_sigma_y = -tau * grad_y[valid] * area
    return SurfaceResult(
        tau=tau.astype(np.float32),
        x=xx[valid].astype(np.float32),
        y=yy[valid].astype(np.float32),
        velocity_x=hydro.freezeout_velocity_x[valid].astype(np.float32),
        velocity_y=hydro.freezeout_velocity_y[valid].astype(np.float32),
        d_sigma_tau=d_sigma_tau.astype(np.float32),
        d_sigma_x=d_sigma_x.astype(np.float32),
        d_sigma_y=d_sigma_y.astype(np.float32),
    )


def bessel_k2(z: float) -> float:
    integration_axis = np.linspace(0.0, 12.0, 6000)
    integrand = np.exp(-z * np.cosh(integration_axis)) * np.cosh(2.0 * integration_axis)
    return float(np.trapezoid(integrand, integration_axis))


def thermal_species_probabilities(temperature: float) -> np.ndarray:
    densities = []
    for mass, degeneracy in zip(SPECIES_MASSES, SPECIES_DEGENERACIES):
        densities.append(degeneracy * mass * mass * temperature * bessel_k2(mass / temperature))
    values = np.asarray(densities, dtype=np.float64)
    return values / values.sum()


def sample_thermal_magnitudes(
    masses: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(masses.size, dtype=np.float64)
    for mass in np.unique(masses):
        target = np.flatnonzero(masses == mass)
        accepted_values: list[float] = []
        while len(accepted_values) < target.size:
            batch = max(256, 5 * (target.size - len(accepted_values)))
            momentum = rng.gamma(shape=3.0, scale=temperature, size=batch)
            energy = np.sqrt(momentum * momentum + mass * mass)
            probability = np.exp(-(energy - momentum) / temperature)
            accepted_values.extend(momentum[rng.random(batch) < probability].tolist())
        result[target] = np.asarray(accepted_values[: target.size], dtype=np.float64)
    return result


def lorentz_boost(
    rest_energy: np.ndarray,
    rest_momentum: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    beta2 = np.sum(beta * beta, axis=1)
    too_fast = beta2 >= 0.998**2
    if np.any(too_fast):
        beta[too_fast] *= (0.998 / np.sqrt(beta2[too_fast]))[:, None]
        beta2 = np.sum(beta * beta, axis=1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta2, 1.0e-12))
    beta_dot_p = np.sum(beta * rest_momentum, axis=1)
    factor = np.zeros_like(beta2)
    moving = beta2 > 1.0e-14
    factor[moving] = (gamma[moving] - 1.0) * beta_dot_p[moving] / beta2[moving] + gamma[moving] * rest_energy[moving]
    lab_momentum = rest_momentum + factor[:, None] * beta
    lab_energy = gamma * (rest_energy + beta_dot_p)
    return lab_energy, lab_momentum


def sample_cooper_frye_particles(
    args: argparse.Namespace,
    surface: SurfaceResult,
    rng: np.random.Generator,
) -> ParticleResult:
    envelope = np.abs(surface.d_sigma_tau) + np.abs(surface.d_sigma_x) + np.abs(surface.d_sigma_y)
    envelope = np.maximum(envelope.astype(np.float64), 1.0e-14)
    surface_probability = envelope / envelope.sum()
    species_probability = thermal_species_probabilities(args.freezeout_temperature)
    # The displayed particle set stays compact, while a larger importance-
    # sampled pool stabilizes integrated pT and vn estimates.
    candidate_count = max(args.particles * 32, 12000)

    cell = rng.choice(surface.tau.size, size=candidate_count, replace=True, p=surface_probability)
    species = rng.choice(SPECIES_NAMES.size, size=candidate_count, replace=True, p=species_probability)
    masses = SPECIES_MASSES[species]
    momentum_magnitude = sample_thermal_magnitudes(masses, args.freezeout_temperature, rng)

    cosine_theta = rng.uniform(-1.0, 1.0, size=candidate_count)
    sine_theta = np.sqrt(np.maximum(1.0 - cosine_theta * cosine_theta, 0.0))
    azimuth = rng.uniform(0.0, 2.0 * math.pi, size=candidate_count)
    rest_momentum = np.column_stack(
        (
            momentum_magnitude * cosine_theta,
            momentum_magnitude * sine_theta * np.cos(azimuth),
            momentum_magnitude * sine_theta * np.sin(azimuth),
        )
    )
    rest_energy = np.sqrt(momentum_magnitude * momentum_magnitude + masses * masses)

    eta_s = rng.uniform(-args.eta_window, args.eta_window, size=candidate_count)
    cosh_eta = np.cosh(eta_s)
    beta = np.column_stack(
        (
            np.tanh(eta_s),
            surface.velocity_x[cell] / cosh_eta,
            surface.velocity_y[cell] / cosh_eta,
        )
    ).astype(np.float64)
    lab_energy, lab_momentum = lorentz_boost(rest_energy, rest_momentum, beta)
    momentum_longitudinal = lab_momentum[:, 0]
    momentum_x = lab_momentum[:, 1]
    momentum_y = lab_momentum[:, 2]
    momentum_tau = cosh_eta * lab_energy - np.sinh(eta_s) * momentum_longitudinal
    p_dot_sigma = (
        momentum_tau * surface.d_sigma_tau[cell]
        + momentum_x * surface.d_sigma_x[cell]
        + momentum_y * surface.d_sigma_y[cell]
    )
    importance = np.maximum(p_dot_sigma, 0.0) / (rest_energy * envelope[cell])
    if float(importance.sum()) <= 0.0:
        raise RuntimeError("The Cooper-Frye surface produced no positive contributions.")
    selection_probability = importance / importance.sum()
    selected = rng.choice(candidate_count, size=args.particles, replace=False, p=selection_probability)

    selected_px = momentum_x[selected]
    selected_py = momentum_y[selected]
    candidate_pt = np.sqrt(momentum_x * momentum_x + momentum_y * momentum_y)
    candidate_phi = np.arctan2(momentum_y, momentum_x)
    observable_weight = importance / importance.sum()
    return ParticleResult(
        species=species[selected].astype(np.int8),
        mass=masses[selected].astype(np.float32),
        tau=surface.tau[cell[selected]].astype(np.float32),
        eta_s=eta_s[selected].astype(np.float32),
        x=surface.x[cell[selected]].astype(np.float32),
        y=surface.y[cell[selected]].astype(np.float32),
        energy=lab_energy[selected].astype(np.float32),
        momentum_longitudinal=momentum_longitudinal[selected].astype(np.float32),
        momentum_x=selected_px.astype(np.float32),
        momentum_y=selected_py.astype(np.float32),
        cooper_frye_weight=importance[selected].astype(np.float32),
        mean_pt=float(np.sum(observable_weight * candidate_pt)),
        # The impact parameter is along +y in this event geometry.
        v2=float(np.sum(observable_weight * np.cos(2.0 * (candidate_phi - 0.5 * math.pi)))),
        v3=float(np.sum(observable_weight * np.cos(3.0 * candidate_phi))),
    )


def physics_audit(
    args: argparse.Namespace,
    geometry: EventGeometry,
    preequilibrium: PreEquilibriumResult,
    hydro: HydroResult,
    surface: SurfaceResult,
    particles: ParticleResult,
) -> dict[str, object]:
    finite_fields = bool(
        np.all(np.isfinite(hydro.energy_density))
        and np.all(np.isfinite(hydro.velocity_x))
        and np.all(np.isfinite(hydro.velocity_y))
        and np.all(np.isfinite(preequilibrium.landau_energy_density))
        and np.all(np.isfinite(preequilibrium.landau_residual))
    )
    temperature_rises = int(np.count_nonzero(np.diff(hydro.max_temperature) > 2.0e-3))
    preequilibrium_speed = np.sqrt(preequilibrium.velocity_x**2 + preequilibrium.velocity_y**2)
    preequilibrium_peak_energy = np.max(preequilibrium.energy_density, axis=(1, 2))
    final_preequilibrium_energy = np.maximum(preequilibrium.landau_energy_density[-1], 0.0)
    mean_preequilibrium_flow = float(
        np.sum(final_preequilibrium_energy * preequilibrium_speed[-1])
        / max(float(np.sum(final_preequilibrium_energy)), 1.0e-12)
    )
    qgp_cells_at_hydrodynamization = int(np.count_nonzero(hydro.temperature[0] >= 0.165))
    initial_preequilibrium_flow = float(np.max(preequilibrium_speed[0]))
    handoff_velocity_error = float(
        np.max(
            np.hypot(
                hydro.velocity_x[0] - preequilibrium.velocity_x[-1],
                hydro.velocity_y[0] - preequilibrium.velocity_y[-1],
            )
        )
    )
    handoff_peak_energy_error = abs(float(np.max(hydro.energy_density[0])) - args.initial_peak_energy)
    matched_medium = preequilibrium.effective_temperature >= args.freezeout_temperature
    max_landau_eigen_residual = float(
        np.max(preequilibrium.landau_residual[matched_medium]) if np.any(matched_medium) else np.inf
    )
    handoff_temperature_error = float(
        np.max(np.abs(hydro.temperature[0] - preequilibrium.effective_temperature[-1]))
    )
    audit = {
        "passed": bool(
            finite_fields
            and float(np.min(hydro.energy_density)) >= 0.0
            and float(np.max(hydro.max_velocity)) < 1.0
            and surface.tau.size >= 100
            and particles.species.size == args.particles
            and float(np.min(particles.cooper_frye_weight)) >= 0.0
            and temperature_rises <= 1
            and np.all(np.diff(preequilibrium.tau) > 0.0)
            and float(np.max(preequilibrium_speed)) < 1.0
            and int(np.count_nonzero(np.diff(preequilibrium_peak_energy) > 1.0e-3)) == 0
            and qgp_cells_at_hydrodynamization > 0
            and initial_preequilibrium_flow < 1.0e-5
            and handoff_velocity_error < 1.0e-3
            and handoff_peak_energy_error < 1.0e-3
            and handoff_temperature_error < 1.0e-5
            and max_landau_eigen_residual < 1.0e-3
        ),
        "finite_fields": finite_fields,
        "n_part": geometry.n_part,
        "n_coll": geometry.n_coll,
        "preequilibrium_snapshots": int(preequilibrium.tau.size),
        "preequilibrium_tau_start_fm_c": float(preequilibrium.tau[0]),
        "preequilibrium_tau_end_fm_c": float(preequilibrium.tau[-1]),
        "max_preequilibrium_flow": float(np.max(preequilibrium_speed)),
        "energy_weighted_mean_preequilibrium_flow": mean_preequilibrium_flow,
        "initial_preequilibrium_flow": initial_preequilibrium_flow,
        "initial_max_landau_effective_temperature_gev": float(np.max(preequilibrium.effective_temperature[0])),
        "handoff_max_landau_effective_temperature_gev": float(np.max(preequilibrium.effective_temperature[-1])),
        "landau_handoff_velocity_linf_error": handoff_velocity_error,
        "landau_handoff_peak_energy_error": handoff_peak_energy_error,
        "landau_handoff_temperature_linf_error_gev": handoff_temperature_error,
        "max_landau_eigen_residual": max_landau_eigen_residual,
        "qgp_cells_at_hydrodynamization": qgp_cells_at_hydrodynamization,
        "hydro_snapshots": int(hydro.tau.size),
        "initial_max_temperature_gev": float(hydro.max_temperature[0]),
        "final_max_temperature_gev": float(hydro.max_temperature[-1]),
        "max_transverse_velocity": float(np.max(hydro.max_velocity)),
        "max_temperature_rise_count": temperature_rises,
        "freezeout_cells": int(surface.tau.size),
        "sampled_particles": int(particles.species.size),
        "mean_pt_gev": particles.mean_pt,
        "v2_reaction_plane": particles.v2,
        "v3_reaction_plane": particles.v3,
    }
    return audit


def save_event_cache(
    args: argparse.Namespace,
    device: torch.device,
    geometry: EventGeometry,
    preequilibrium: PreEquilibriumResult,
    hydro: HydroResult,
    surface: SurfaceResult,
    particles: ParticleResult,
    axis: np.ndarray,
    audit: dict[str, object],
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format_version=np.asarray("hic_blender_event_v3"),
        physics_model=np.asarray(
            "MC-Glauber + transverse collisionless pre-equilibrium + Landau matching + "
            "boost-invariant 2+1D ideal relativistic hydrodynamics + Cooper-Frye"
        ),
        afterburner=np.asarray("none; sampled hadrons free-stream after particlization"),
        device=np.asarray(str(device)),
        beam_system=np.asarray("Au+Au"),
        sqrt_snn_gev=np.asarray(args.sqrt_snn, dtype=np.float32),
        impact_parameter=np.asarray(args.impact_parameter, dtype=np.float32),
        seed=np.asarray(args.seed, dtype=np.int32),
        tau_initial=np.asarray(args.tau_initial, dtype=np.float32),
        tau0=np.asarray(args.tau0, dtype=np.float32),
        freezeout_temperature=np.asarray(args.freezeout_temperature, dtype=np.float32),
        eta_window=np.asarray(args.eta_window, dtype=np.float32),
        grid_axis=axis.astype(np.float32),
        n_part=np.asarray(geometry.n_part, dtype=np.int16),
        n_coll=np.asarray(geometry.n_coll, dtype=np.int32),
        nucleus_a_xyz=geometry.nucleus_a_xyz.astype(np.float32),
        nucleus_b_xyz=geometry.nucleus_b_xyz.astype(np.float32),
        participant_a=geometry.participant_a.astype(np.uint8),
        participant_b=geometry.participant_b.astype(np.uint8),
        collision_centers=geometry.collision_centers.astype(np.float32),
        preeq_tau=preequilibrium.tau,
        preeq_energy_density=preequilibrium.energy_density,
        preeq_momentum_x=preequilibrium.momentum_x,
        preeq_momentum_y=preequilibrium.momentum_y,
        preeq_stress_xx=preequilibrium.stress_xx,
        preeq_stress_xy=preequilibrium.stress_xy,
        preeq_stress_yy=preequilibrium.stress_yy,
        preeq_landau_energy_density=preequilibrium.landau_energy_density,
        preeq_effective_temperature=preequilibrium.effective_temperature,
        preeq_velocity_x=preequilibrium.velocity_x,
        preeq_velocity_y=preequilibrium.velocity_y,
        preeq_landau_residual=preequilibrium.landau_residual,
        hydro_tau=hydro.tau,
        energy_density=hydro.energy_density,
        temperature=hydro.temperature,
        velocity_x=hydro.velocity_x,
        velocity_y=hydro.velocity_y,
        max_temperature=hydro.max_temperature,
        max_velocity=hydro.max_velocity,
        total_energy_per_rapidity=hydro.total_energy_per_rapidity,
        eccentricity2=hydro.eccentricity2,
        freezeout_tau_map=hydro.freezeout_tau,
        surface_tau=surface.tau,
        surface_x=surface.x,
        surface_y=surface.y,
        surface_velocity_x=surface.velocity_x,
        surface_velocity_y=surface.velocity_y,
        surface_d_sigma_tau=surface.d_sigma_tau,
        surface_d_sigma_x=surface.d_sigma_x,
        surface_d_sigma_y=surface.d_sigma_y,
        particle_species_names=SPECIES_NAMES,
        particle_species=particles.species,
        particle_mass=particles.mass,
        particle_tau=particles.tau,
        particle_eta_s=particles.eta_s,
        particle_x=particles.x,
        particle_y=particles.y,
        particle_energy=particles.energy,
        particle_momentum_longitudinal=particles.momentum_longitudinal,
        particle_momentum_x=particles.momentum_x,
        particle_momentum_y=particles.momentum_y,
        particle_cooper_frye_weight=particles.cooper_frye_weight,
        mean_pt=np.asarray(particles.mean_pt, dtype=np.float32),
        v2=np.asarray(particles.v2, dtype=np.float32),
        v3=np.asarray(particles.v3, dtype=np.float32),
    )
    audit_path = args.audit_output or args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="ascii")
    print(f"Saved event cache: {args.output}")
    print(f"Saved physics audit: {audit_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a physics-driven heavy-ion event cache for Blender.")
    parser.add_argument("--output", type=Path, default=Path("outputs/blender_physics_event.npz"))
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--eos-table", type=Path, default=Path("data/eos_table.csv"))
    parser.add_argument("--impact-parameter", type=float, default=5.7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sqrt-snn", type=float, default=200.0, help="Collision energy per nucleon pair in GeV.")
    parser.add_argument("--mass-number", type=int, default=197)
    parser.add_argument("--radius", type=float, default=6.38)
    parser.add_argument("--skin-depth", type=float, default=0.535)
    parser.add_argument("--inelastic-cross-section", type=float, default=4.2, help="fm^2")
    parser.add_argument("--nucleon-width", type=float, default=0.45, help="fm")
    parser.add_argument("--binary-fraction", type=float, default=0.14)
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument("--extent", type=float, default=12.0, help="Transverse half-width in fm.")
    parser.add_argument("--initial-peak-energy", type=float, default=6.0, help="GeV/fm^3")
    parser.add_argument("--tau-initial", type=float, default=0.1, help="Pre-equilibrium formation time in fm/c.")
    parser.add_argument("--tau0", type=float, default=0.6, help="fm/c")
    parser.add_argument("--preeq-snapshots", type=int, default=9)
    parser.add_argument("--preeq-angles", type=int, default=48)
    parser.add_argument("--tau-max", type=float, default=14.0, help="fm/c")
    parser.add_argument("--snapshot-dt", type=float, default=0.08, help="fm/c")
    parser.add_argument("--cfl", type=float, default=0.24)
    parser.add_argument("--freezeout-temperature", type=float, default=0.145, help="GeV")
    parser.add_argument("--eta-window", type=float, default=0.8)
    parser.add_argument("--particles", type=int, default=900)
    parser.add_argument("--energy-floor", type=float, default=1.0e-8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    eos = build_table_eos(args.eos_table)
    if not 0.0 <= args.impact_parameter <= 2.0 * args.radius + 2.0:
        raise SystemExit("impact parameter is outside the supported Au+Au geometry")
    if args.initial_peak_energy > eos.table_energy[-1]:
        raise SystemExit(
            f"initial peak energy exceeds the EoS table range ({eos.table_energy[-1]:g} GeV/fm^3); "
            "extend the table or lower --initial-peak-energy"
        )
    if not 0.0 < args.tau_initial < args.tau0:
        raise SystemExit("tau-initial must be positive and smaller than tau0")
    if args.preeq_snapshots < 2 or args.preeq_angles < 8:
        raise SystemExit("pre-equilibrium requires at least 2 snapshots and 8 angular streams")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    xx, yy, dx = make_grid(args.grid_size, args.extent, device)
    axis = np.linspace(-args.extent, args.extent, args.grid_size, dtype=np.float32)
    geometry = build_event_geometry(args, xx, yy, rng)
    print(f"Glauber event: b={args.impact_parameter:.2f} fm, Npart={geometry.n_part}, Ncoll={geometry.n_coll}")
    preequilibrium = evolve_pre_equilibrium(args, geometry.source, eos)
    hydro = evolve_hydrodynamics(args, preequilibrium, eos, xx, yy, dx)
    surface = build_freezeout_surface(hydro, axis, dx)
    particles = sample_cooper_frye_particles(args, surface, rng)
    audit = physics_audit(args, geometry, preequilibrium, hydro, surface, particles)
    save_event_cache(args, device, geometry, preequilibrium, hydro, surface, particles, axis, audit)
    print(
        f"Particlization: cells={surface.tau.size}, particles={particles.species.size}, "
        f"mean pT={particles.mean_pt:.3f} GeV, v2={particles.v2:.4f}"
    )
    if not audit["passed"]:
        raise SystemExit("Physics audit failed. Inspect the generated audit JSON.")


if __name__ == "__main__":
    main()
