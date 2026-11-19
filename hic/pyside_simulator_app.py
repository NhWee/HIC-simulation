import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


TC_GEV = 0.165
TFO_GEV = 0.118
MAX_NUCLEONS = 416
DEFAULT_HYDRO_CACHE = Path("outputs/pyside_hydro_cache_b5p7.npz")


@dataclass
class FieldState:
    progress: float
    impact_parameter: float
    temperature: np.ndarray
    freezeout: np.ndarray
    nuclei: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    particle_x: np.ndarray
    particle_y: np.ndarray
    particle_size: np.ndarray
    particle_species: np.ndarray
    stage: str
    tau: float
    radial_flow: float
    eccentricity: float
    multiplicity: int
    n_part: int
    n_coll: int
    spectators: int
    mean_cs2: float
    emitted_v2: float
    mean_pt: float
    pion_count: int
    kaon_count: int
    proton_count: int
    qgp_fraction: float
    mixed_fraction: float
    hadron_fraction: float
    centrality_percent: float
    physics_status: str


@dataclass
class ParticleSample:
    x: np.ndarray
    y: np.ndarray
    size: np.ndarray
    species: np.ndarray
    emitted_v2: float
    mean_pt: float
    pion_count: int
    kaon_count: int
    proton_count: int


@dataclass
class HydroFrameCache:
    path: Path
    impact_parameter: float
    mass_number: int
    eos_name: str
    progress: np.ndarray
    tau: np.ndarray
    temperature: np.ndarray
    freezeout: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    qgp_fraction: np.ndarray
    mixed_fraction: np.ndarray
    hadron_fraction: np.ndarray
    mean_cs2: np.ndarray
    radial_flow: np.ndarray
    eccentricity: np.ndarray
    n_part: int
    n_coll: int
    multiplicity: int

    @property
    def grid_size(self) -> int:
        return int(self.temperature.shape[-1])

    @classmethod
    def load(cls, path: Path) -> "HydroFrameCache":
        data = np.load(path, allow_pickle=False)
        eos_name = str(data["eos_name"].item()) if "eos_name" in data.files else "legacy conformal"
        return cls(
            path=path,
            impact_parameter=float(data["impact_parameter"]),
            mass_number=int(data["mass_number"]),
            eos_name=eos_name,
            progress=data["progress"].astype(np.float32),
            tau=data["tau"].astype(np.float32),
            temperature=data["temperature"].astype(np.float32),
            freezeout=data["freezeout"].astype(np.float32),
            vx=data["vx"].astype(np.float32),
            vy=data["vy"].astype(np.float32),
            qgp_fraction=data["qgp_fraction"].astype(np.float32),
            mixed_fraction=data["mixed_fraction"].astype(np.float32),
            hadron_fraction=data["hadron_fraction"].astype(np.float32),
            mean_cs2=data["mean_cs2"].astype(np.float32),
            radial_flow=data["radial_flow"].astype(np.float32),
            eccentricity=data["eccentricity"].astype(np.float32),
            n_part=int(data["n_part"]),
            n_coll=int(data["n_coll"]),
            multiplicity=int(data["multiplicity"]),
        )


def smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    x = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return x * x * (3.0 - 2.0 * x)


def soften_rgb(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    center = padded[1:-1, 1:-1]
    cardinal = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    diagonal = padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:]
    return (4.0 * center + 2.0 * cardinal + diagonal) / 16.0


def deterministic_unit(indices: np.ndarray, multiplier: int, offset: int = 0) -> np.ndarray:
    phase = indices.astype(np.float64) * (12.9898 + multiplier * 0.000_137) + offset * 78.233
    values = np.sin(phase) * 43_758.545_312_3
    return (values - np.floor(values)).astype(np.float32)


def stage_for(progress: float, qgp_fraction: float, mixed_fraction: float, hadron_fraction: float) -> str:
    if progress < 0.20:
        return "incoming nuclei"
    if progress < 0.32:
        return "pre-equilibrium"
    if qgp_fraction > 0.018:
        return "QGP hydrodynamics"
    if mixed_fraction > 0.012:
        return "mixed / freeze-out"
    if hadron_fraction > 0.020:
        return "hadron gas"
    return "cooling plasma"


def physics_status_for(
    progress: float,
    n_part: int,
    multiplicity: int,
    qgp_fraction: float,
    particle_count: int,
    emitted_v2: float,
    eccentricity: float,
) -> str:
    warnings = []
    if n_part == 0 and (multiplicity > 0 or particle_count > 0 or qgp_fraction > 0.001):
        warnings.append("no-overlap yield")
    if progress < 0.60 and particle_count > 0:
        warnings.append("early hadrons")
    if progress > 0.82 and qgp_fraction > 0.005:
        warnings.append("late QGP")
    if eccentricity < 0.02 and abs(emitted_v2) > 0.04:
        warnings.append("central v2")
    return "OK" if not warnings else "; ".join(warnings)


def empty_particle_sample() -> ParticleSample:
    return ParticleSample(
        x=np.empty(0, dtype=np.float32),
        y=np.empty(0, dtype=np.float32),
        size=np.empty(0, dtype=np.float32),
        species=np.empty(0, dtype=np.int16),
        emitted_v2=0.0,
        mean_pt=0.0,
        pion_count=0,
        kaon_count=0,
        proton_count=0,
    )


def sample_hadrons_from_surface(
    progress: float,
    temperature: np.ndarray,
    freezeout: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    multiplicity: int,
    eccentricity: float,
    radial_flow: float,
    fixed_surface: np.ndarray | None = None,
    emission_eccentricity: float | None = None,
    event_plane_angle: float = math.pi / 2.0,
) -> ParticleSample:
    grid_size = int(temperature.shape[0])
    hadron_progress = smoothstep(progress, 0.60, 0.96)
    particle_budget = min(360, round(0.30 * multiplicity))
    if particle_budget <= 0 or hadron_progress <= 0.0:
        return empty_particle_sample()

    shape_eccentricity = eccentricity if emission_eccentricity is None else emission_eccentricity
    indices = np.arange(particle_budget, dtype=np.float32)
    fractions = deterministic_unit(indices, 53)
    particle_species = np.where(fractions < 0.78, 0, np.where(fractions < 0.91, 1, 2)).astype(np.int16)
    velocity_scale = np.choose(particle_species, [1.00, 0.82, 0.62]).astype(np.float32)
    mass_gev = np.choose(particle_species, [0.13957, 0.49368, 0.93827]).astype(np.float32)

    axis = np.linspace(-1.25, 1.25, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    phi_grid = np.arctan2(yy, xx)
    in_plane_weight = np.clip(
        1.0 + 0.46 * shape_eccentricity * np.cos(2.0 * (phi_grid - event_plane_angle)),
        0.18,
        1.82,
    )
    stable_x = 0.58 * (1.0 + 0.36 * shape_eccentricity)
    stable_y = 0.52 * (1.0 - 0.12 * shape_eccentricity)
    stable_radius = np.sqrt((xx / stable_x) ** 2 + (yy / stable_y) ** 2)
    stable_surface = np.exp(-((stable_radius - 1.0) ** 2) / (2.0 * 0.10**2)).astype(np.float32)
    if fixed_surface is None:
        source_surface = stable_surface
    else:
        source_surface = fixed_surface.astype(np.float32)
    surface_weight = source_surface * in_plane_weight
    flat_weight = surface_weight.reshape(-1).astype(np.float64)
    if float(flat_weight.sum()) <= 1.0e-12:
        return empty_particle_sample()

    cdf = np.cumsum(flat_weight)
    sample_u = deterministic_unit(indices, 104_729, 17).astype(np.float64)
    flat_indices = np.searchsorted(cdf, sample_u * cdf[-1], side="right")
    flat_indices = np.clip(flat_indices, 0, flat_weight.size - 1)
    rows = flat_indices // grid_size
    cols = flat_indices % grid_size
    cell_dx = float(axis[1] - axis[0])
    jitter_x = (deterministic_unit(indices, 37_111, 5) - 0.5) * cell_dx * 0.72
    jitter_y = (deterministic_unit(indices, 91_733, 11) - 0.5) * cell_dx * 0.72
    source_x = axis[cols].astype(np.float32) + jitter_x
    source_y = axis[rows].astype(np.float32) + jitter_y
    radial_phi = np.arctan2(source_y, source_x)
    local_vx = vx[rows, cols]
    local_vy = vy[rows, cols]
    local_speed = np.sqrt(local_vx * local_vx + local_vy * local_vy)
    thermal_spread = (deterministic_unit(indices, 71_071, 3) - 0.5) * 0.18
    elliptic_deflection = 0.10 * shape_eccentricity * np.sin(2.0 * (radial_phi - event_plane_angle))
    momentum_phi = radial_phi + thermal_spread - elliptic_deflection

    emission_delay = deterministic_unit(indices, 29_003, 23)
    emission_threshold = 0.06 + 0.88 * emission_delay
    visible = hadron_progress >= emission_threshold
    in_plane_boost = 1.0 + 0.18 * shape_eccentricity * np.cos(2.0 * (momentum_phi - event_plane_angle))
    travel = velocity_scale * in_plane_boost
    travel *= 0.50 + 0.20 * shape_eccentricity
    world_x = source_x + np.cos(momentum_phi) * travel
    world_y = source_y + np.sin(momentum_phi) * travel

    sampled_temperature = temperature[rows, cols].astype(np.float32)
    thermal_pt = -np.log(np.clip(1.0 - deterministic_unit(indices, 19_019, 7), 1.0e-5, 1.0))
    thermal_pt *= np.maximum(sampled_temperature, TFO_GEV) * 1.10
    flow_pt = np.clip(local_speed + radial_flow, 0.0, 1.25) * (0.32 + 0.52 * mass_gev) * in_plane_boost
    pt = thermal_pt + flow_pt

    world_x = world_x[visible]
    world_y = world_y[visible]
    particle_species = particle_species[visible]
    particle_x = ((world_x + 1.25) / 2.5 * (grid_size - 1)).astype(np.float32)
    particle_y = ((world_y + 1.25) / 2.5 * (grid_size - 1)).astype(np.float32)
    particle_size = np.choose(particle_species, [4.0, 5.5, 7.0]).astype(np.float32)
    particle_size += (1.4 * deterministic_unit(np.arange(len(particle_species), dtype=np.float32), 13_013)).astype(np.float32)
    visible_phi = momentum_phi[visible]
    emitted_v2 = float(np.mean(np.cos(2.0 * (visible_phi - event_plane_angle)))) if visible_phi.size else 0.0
    if eccentricity < 0.02:
        emitted_v2 = 0.0
    return ParticleSample(
        x=particle_x,
        y=particle_y,
        size=particle_size,
        species=particle_species,
        emitted_v2=emitted_v2,
        mean_pt=float(np.mean(pt[visible])) if visible_phi.size else 0.0,
        pion_count=int(np.sum(particle_species == 0)),
        kaon_count=int(np.sum(particle_species == 1)),
        proton_count=int(np.sum(particle_species == 2)),
    )


def compute_field(progress: float, impact_parameter: float, grid_size: int = 180, include_particles: bool = True) -> FieldState:
    axis = np.linspace(-1.25, 1.25, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    eccentricity = min(0.85, impact_parameter / 12.0 * 1.05)
    overlap = max(0.0, 1.0 - impact_parameter / 12.0)
    centrality_percent = min(100.0, max(0.0, 100.0 * (impact_parameter / 12.0) ** 2))
    n_part = round(MAX_NUCLEONS * overlap**0.72)
    n_coll = round(1150 * overlap**1.45)
    spectators = max(0, MAX_NUCLEONS - n_part)
    collision = np.clip(progress / 0.30, 0.0, 1.0)
    center_a_x = -1.25 + 2.50 * collision
    center_b_x = 1.25 - 2.50 * collision
    center_a_y = -0.42 * impact_parameter / 12.0
    center_b_y = 0.42 * impact_parameter / 12.0
    squash = 0.18 + 0.24 * abs(0.5 - collision)
    nucleus_a = np.exp(-(((xx - center_a_x) / squash) ** 2 + ((yy - center_a_y) / 0.45) ** 2) * 2.2)
    nucleus_b = np.exp(-(((xx - center_b_x) / squash) ** 2 + ((yy - center_b_y) / 0.45) ** 2) * 2.2)
    nuclei = np.clip(nucleus_a + nucleus_b, 0.0, 1.0) * max(0.06, 1.0 - progress * 1.25)
    tau = max(0.0, (progress - 0.20) * 14.0)
    heat_up = smoothstep(progress, 0.22, 0.42)
    cooling = math.exp(-max(0.0, progress - 0.34) * 3.55)
    deposition_scale = overlap**0.42 if overlap > 0.0 else 0.0
    max_temp = 0.58 * heat_up * cooling * deposition_scale
    expansion = 0.34 + 0.43 * smoothstep(progress, 0.28, 0.92)
    sigma_x = expansion * (1.0 + 0.35 * eccentricity)
    sigma_y = expansion * (1.0 - 0.18 * eccentricity)
    angle = -0.22
    xr = xx * math.cos(angle) - yy * math.sin(angle)
    yr = xx * math.sin(angle) + yy * math.cos(angle)
    core = np.exp(-((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2) * 1.8)
    hot_spots = np.zeros_like(core)
    for index in range(18):
        a = index * 2.399963
        radius = 0.08 + (index % 7) * 0.045
        hx = math.cos(a) * radius * (1.2 + eccentricity)
        hy = math.sin(a) * radius * 0.8
        hot_spots += np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / 0.006))
    temperature = max_temp * np.clip(core + 0.14 * hot_spots, 0.0, 1.2)
    freezeout = np.exp(-((temperature - TFO_GEV) ** 2) / (2.0 * 0.012**2)) * smoothstep(progress, 0.48, 0.73)
    active = temperature > 0.025
    raw_qgp_fraction = float(np.mean((temperature >= TC_GEV) & active))
    raw_mixed_fraction = float(np.mean((temperature < TC_GEV) & (temperature >= TFO_GEV) & active))
    raw_hadron_fraction = float(np.mean((temperature < TFO_GEV) & (temperature >= 0.045) & active))
    qgp_fraction = raw_qgp_fraction if progress >= 0.30 else 0.0
    mixed_fraction = raw_mixed_fraction * smoothstep(progress, 0.42, 0.64)
    hadron_fraction = raw_hadron_fraction * smoothstep(progress, 0.60, 0.82)
    qgp_weight = 1.0 / (1.0 + np.exp(-(temperature - TC_GEV) / 0.018))
    crossover_dip = np.exp(-((temperature - TC_GEV) ** 2) / (2.0 * 0.030**2))
    cs2 = np.clip(0.18 + 0.12 * qgp_weight - 0.075 * crossover_dip, 0.055, 0.33)
    mean_cs2 = float(np.mean(cs2[active])) if np.any(active) else 0.0
    radius = np.sqrt(xx * xx + yy * yy) + 1.0e-6
    flow_scale = (0.35 + 0.65 * overlap**0.25) if overlap > 0.0 else 0.0
    radial_flow = 0.68 * smoothstep(progress, 0.31, 0.93) * flow_scale
    vx = radial_flow * xx / radius * core
    vy = radial_flow * yy / radius * core
    multiplicity = round(1.45 * n_part + 0.32 * n_coll)
    particles = (
        sample_hadrons_from_surface(
            progress=progress,
            temperature=temperature,
            freezeout=freezeout,
            vx=vx,
            vy=vy,
            multiplicity=multiplicity,
            eccentricity=eccentricity,
            radial_flow=radial_flow,
        )
        if include_particles
        else empty_particle_sample()
    )
    physics_status = physics_status_for(
        progress=progress,
        n_part=n_part,
        multiplicity=multiplicity,
        qgp_fraction=qgp_fraction,
        particle_count=len(particles.x),
        emitted_v2=particles.emitted_v2,
        eccentricity=eccentricity,
    )
    return FieldState(
        progress=progress,
        impact_parameter=impact_parameter,
        temperature=temperature.astype(np.float32),
        freezeout=freezeout.astype(np.float32),
        nuclei=nuclei.astype(np.float32),
        vx=vx.astype(np.float32),
        vy=vy.astype(np.float32),
        particle_x=particles.x,
        particle_y=particles.y,
        particle_size=particles.size,
        particle_species=particles.species,
        stage=stage_for(progress, qgp_fraction, mixed_fraction, hadron_fraction),
        tau=tau,
        radial_flow=radial_flow,
        eccentricity=eccentricity,
        multiplicity=multiplicity,
        n_part=n_part,
        n_coll=n_coll,
        spectators=spectators,
        mean_cs2=mean_cs2,
        emitted_v2=particles.emitted_v2,
        mean_pt=particles.mean_pt,
        pion_count=particles.pion_count,
        kaon_count=particles.kaon_count,
        proton_count=particles.proton_count,
        qgp_fraction=qgp_fraction,
        mixed_fraction=mixed_fraction,
        hadron_fraction=hadron_fraction,
        centrality_percent=centrality_percent,
        physics_status=physics_status,
    )


def interpolate_cache(cache: HydroFrameCache, values: np.ndarray, progress: float) -> np.ndarray | float:
    if progress <= float(cache.progress[0]):
        return values[0].copy() if values.ndim > 1 else float(values[0])
    if progress >= float(cache.progress[-1]):
        return values[-1].copy() if values.ndim > 1 else float(values[-1])
    upper = int(np.searchsorted(cache.progress, progress, side="right"))
    lower = max(0, upper - 1)
    denom = max(float(cache.progress[upper] - cache.progress[lower]), 1.0e-8)
    alpha = (progress - float(cache.progress[lower])) / denom
    blended = (1.0 - alpha) * values[lower] + alpha * values[upper]
    return blended.astype(np.float32) if values.ndim > 1 else float(blended)


def compute_hydro_cached_field(
    cache: HydroFrameCache,
    progress: float,
    include_particles: bool = True,
) -> FieldState:
    cache_start = float(cache.progress[0])
    transition_end = min(cache_start + 0.36, float(cache.progress[-1]))
    if progress < cache_start:
        return compute_field(progress, cache.impact_parameter, grid_size=cache.grid_size, include_particles=include_particles)
    transition_weight = smoothstep(progress, cache_start, transition_end)
    analytic_transition = (
        compute_field(progress, cache.impact_parameter, grid_size=cache.grid_size, include_particles=False)
        if transition_weight < 1.0
        else None
    )

    temperature = interpolate_cache(cache, cache.temperature, progress)
    freezeout = interpolate_cache(cache, cache.freezeout, progress)
    vx = interpolate_cache(cache, cache.vx, progress)
    vy = interpolate_cache(cache, cache.vy, progress)
    qgp_fraction = float(interpolate_cache(cache, cache.qgp_fraction, progress))
    mixed_fraction = float(interpolate_cache(cache, cache.mixed_fraction, progress))
    hadron_fraction = float(interpolate_cache(cache, cache.hadron_fraction, progress))
    mean_cs2 = float(interpolate_cache(cache, cache.mean_cs2, progress))
    radial_flow = float(interpolate_cache(cache, cache.radial_flow, progress))
    eccentricity = float(interpolate_cache(cache, cache.eccentricity, progress))
    tau = float(interpolate_cache(cache, cache.tau, progress))
    toy_nuclei = (
        analytic_transition.nuclei
        if analytic_transition is not None
        else compute_field(progress, cache.impact_parameter, grid_size=cache.grid_size, include_particles=False).nuclei
    )
    if analytic_transition is not None:
        w = transition_weight
        temperature = ((1.0 - w) * analytic_transition.temperature + w * temperature).astype(np.float32)
        freezeout = ((1.0 - w) * analytic_transition.freezeout + w * freezeout).astype(np.float32)
        vx = ((1.0 - w) * analytic_transition.vx + w * vx).astype(np.float32)
        vy = ((1.0 - w) * analytic_transition.vy + w * vy).astype(np.float32)
        qgp_fraction = (1.0 - w) * analytic_transition.qgp_fraction + w * qgp_fraction
        mixed_fraction = (1.0 - w) * analytic_transition.mixed_fraction + w * mixed_fraction
        hadron_fraction = (1.0 - w) * analytic_transition.hadron_fraction + w * hadron_fraction
        mean_cs2 = (1.0 - w) * analytic_transition.mean_cs2 + w * mean_cs2
        radial_flow = (1.0 - w) * analytic_transition.radial_flow + w * radial_flow
        eccentricity = (1.0 - w) * analytic_transition.eccentricity + w * eccentricity
        tau = (1.0 - w) * analytic_transition.tau + w * tau

    particles = (
        sample_hadrons_from_surface(
            progress=progress,
            temperature=temperature,
            freezeout=freezeout,
            vx=vx,
            vy=vy,
            multiplicity=cache.multiplicity,
            eccentricity=eccentricity,
            radial_flow=radial_flow,
            fixed_surface=np.max(cache.freezeout, axis=0),
            emission_eccentricity=float(np.clip(np.nanmean(cache.eccentricity[: max(1, len(cache.eccentricity) // 3)]), 0.0, 0.85)),
        )
        if include_particles
        else empty_particle_sample()
    )
    centrality_percent = min(100.0, max(0.0, 100.0 * (cache.impact_parameter / 12.0) ** 2))
    spectators = max(0, 2 * cache.mass_number - cache.n_part)
    physics_status = physics_status_for(
        progress=progress,
        n_part=cache.n_part,
        multiplicity=cache.multiplicity,
        qgp_fraction=qgp_fraction,
        particle_count=len(particles.x),
        emitted_v2=particles.emitted_v2,
        eccentricity=eccentricity,
    )
    return FieldState(
        progress=progress,
        impact_parameter=cache.impact_parameter,
        temperature=temperature.astype(np.float32),
        freezeout=freezeout.astype(np.float32),
        nuclei=toy_nuclei.astype(np.float32),
        vx=vx.astype(np.float32),
        vy=vy.astype(np.float32),
        particle_x=particles.x,
        particle_y=particles.y,
        particle_size=particles.size,
        particle_species=particles.species,
        stage=stage_for(progress, qgp_fraction, mixed_fraction, hadron_fraction),
        tau=tau,
        radial_flow=radial_flow,
        eccentricity=eccentricity,
        multiplicity=cache.multiplicity,
        n_part=cache.n_part,
        n_coll=cache.n_coll,
        spectators=spectators,
        mean_cs2=mean_cs2,
        emitted_v2=particles.emitted_v2,
        mean_pt=particles.mean_pt,
        pion_count=particles.pion_count,
        kaon_count=particles.kaon_count,
        proton_count=particles.proton_count,
        qgp_fraction=qgp_fraction,
        mixed_fraction=mixed_fraction,
        hadron_fraction=hadron_fraction,
        centrality_percent=centrality_percent,
        physics_status=physics_status,
    )


def energy_to_temperature(energy_density: np.ndarray, freezeout_energy: float) -> np.ndarray:
    safe_energy = np.maximum(energy_density, 0.0)
    temperature = TFO_GEV * np.power(np.maximum(safe_energy, 1.0e-12) / max(freezeout_energy, 1.0e-12), 0.25)
    temperature = np.where(safe_energy > 1.0e-7, temperature, 0.0)
    return temperature.astype(np.float32)


def generate_hydro_cache(
    output: Path,
    impact_parameter: float,
    grid_size: int,
    extent: float,
    steps: int,
    sample_every: int,
    seed: int,
    cpu: bool,
    target_max_energy: float,
    freezeout_energy: float,
) -> Path:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise SystemExit("PyTorch is required to build a hydro cache. Run .\\setup.ps1 first.") from error

    from hic.eos import EquationOfState
    from hic.glauber_ensemble import build_initial_source, eccentricity_n
    from hic.ideal_hydro_2d import (
        apply_bjorken_cooling,
        apply_shear_viscous_momentum_diffusion,
        conserved_from_primitive,
        primitive_from_conserved,
        rusanov_step,
    )
    from hic.nuclear_collision_2d import make_grid

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"Building PySide hydro cache on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    eos = EquationOfState(
        kind="crossover",
        speed_of_sound2=1.0 / 3.0,
        low_speed_of_sound2=0.15,
        high_speed_of_sound2=1.0 / 3.0,
        transition_energy=1.0,
        transition_width=0.25,
    )
    xx, yy, dx = make_grid(grid_size, extent, device)
    rng = np.random.default_rng(seed)
    source, n_part, n_coll = build_initial_source(
        xx=xx,
        yy=yy,
        impact_parameter=impact_parameter,
        mass_number=197,
        radius=6.4,
        skin_depth=0.55,
        nucleon_width=0.45,
        inelastic_cross_section=4.2,
        binary_fraction=0.16,
        norm=1.0,
        rng=rng,
        device=device,
    )
    source = torch.clamp(source, min=0.0)
    source_max = float(torch.max(source).detach().cpu()) if source.numel() else 0.0
    if source_max > 1.0e-12:
        source = source / source_max * target_max_energy
    else:
        source = torch.zeros_like(source)

    state = conserved_from_primitive(source, torch.zeros_like(source), torch.zeros_like(source), eos)
    dt = 0.24 * dx / max(1.0, eos.max_sound_speed())
    tau0 = 0.6
    progress_values = []
    tau_values = []
    temperature_frames = []
    freezeout_frames = []
    vx_frames = []
    vy_frames = []
    qgp_values = []
    mixed_values = []
    hadron_values = []
    cs2_values = []
    radial_values = []
    eccentricity_values = []

    print(f"Initial condition: N_part={n_part}, N_coll={n_coll}, dt={dt:.5f}")
    for step in range(steps + 1):
        if step % sample_every == 0 or step == steps:
            energy_density, _, velocity_x, velocity_y = primitive_from_conserved(state[0], state[1], state[2], eos, 1.0e-8)
            energy_np = energy_density.detach().cpu().numpy().astype(np.float32)
            vx_np = velocity_x.detach().cpu().numpy().astype(np.float32)
            vy_np = velocity_y.detach().cpu().numpy().astype(np.float32)
            temperature_np = energy_to_temperature(energy_np, freezeout_energy)
            freezeout_np = np.exp(-((temperature_np - TFO_GEV) ** 2) / (2.0 * 0.012**2)).astype(np.float32)
            hydro_fraction = step / max(steps, 1)
            progress = 0.30 + 0.66 * hydro_fraction**0.65
            freezeout_np *= smoothstep(progress, 0.48, 0.73)
            active = temperature_np > 0.025
            qgp = float(np.mean((temperature_np >= TC_GEV) & active))
            mixed = float(np.mean((temperature_np < TC_GEV) & (temperature_np >= TFO_GEV) & active))
            hadron = float(np.mean((temperature_np < TFO_GEV) & (temperature_np >= 0.045) & active))
            speed_np = np.sqrt(vx_np * vx_np + vy_np * vy_np)
            weight_sum = float(np.sum(energy_np)) + 1.0e-12
            radial = float(np.sum(speed_np * energy_np) / weight_sum)
            progress_values.append(progress)
            tau_values.append(tau0 + step * dt)
            temperature_frames.append(temperature_np)
            freezeout_frames.append(freezeout_np)
            vx_frames.append(vx_np)
            vy_frames.append(vy_np)
            qgp_values.append(qgp)
            mixed_values.append(mixed)
            hadron_values.append(hadron)
            local_cs2 = eos.sound_speed2(energy_density).detach().cpu().numpy().astype(np.float32)
            cs2_values.append(float(np.sum(local_cs2 * energy_np) / weight_sum))
            radial_values.append(radial)
            eccentricity_values.append(eccentricity_n(energy_density, xx, yy, 2))

        if step < steps:
            state = rusanov_step(state, dx, dt, eos, 1.0e-8)
            state = apply_shear_viscous_momentum_diffusion(state, dx, dt, 0.0)
            state = apply_bjorken_cooling(state, tau=tau0 + step * dt, dt=dt, eos=eos, energy_floor=1.0e-8)

    multiplicity = round(1.45 * n_part + 0.32 * n_coll)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        impact_parameter=np.asarray(impact_parameter, dtype=np.float32),
        mass_number=np.asarray(197, dtype=np.int16),
        eos_name=np.asarray(eos.kind),
        progress=np.asarray(progress_values, dtype=np.float32),
        tau=np.asarray(tau_values, dtype=np.float32),
        temperature=np.asarray(temperature_frames, dtype=np.float32),
        freezeout=np.asarray(freezeout_frames, dtype=np.float32),
        vx=np.asarray(vx_frames, dtype=np.float32),
        vy=np.asarray(vy_frames, dtype=np.float32),
        qgp_fraction=np.asarray(qgp_values, dtype=np.float32),
        mixed_fraction=np.asarray(mixed_values, dtype=np.float32),
        hadron_fraction=np.asarray(hadron_values, dtype=np.float32),
        mean_cs2=np.asarray(cs2_values, dtype=np.float32),
        radial_flow=np.asarray(radial_values, dtype=np.float32),
        eccentricity=np.asarray(eccentricity_values, dtype=np.float32),
        n_part=np.asarray(n_part, dtype=np.int16),
        n_coll=np.asarray(n_coll, dtype=np.int32),
        multiplicity=np.asarray(multiplicity, dtype=np.int32),
    )
    print(f"Saved hydro cache: {output}")
    print(f"frames={len(progress_values)}, grid={grid_size}x{grid_size}, multiplicity={multiplicity}")
    return output


class PySideHICApp(QtWidgets.QMainWindow):
    def __init__(self, hydro_cache: HydroFrameCache | None = None) -> None:
        super().__init__()
        self.setFont(QtGui.QFont("Segoe UI", 10))
        self.hydro_cache = hydro_cache
        self.setWindowTitle("Heavy-Ion Collision PySide Simulator")
        self.resize(1280, 820)
        self.progress = 0.0
        self.running = True
        self.impact_parameter = hydro_cache.impact_parameter if hydro_cache is not None else 5.7
        self.speed = 0.7
        self.phase_cache_key: object | None = None
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(24)
        self.update_view()

    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setBackground("#020504")
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.freeze_scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(246, 196, 83, 170), pen=None)
        self.plot.addItem(self.freeze_scatter)
        self.pion_scatter = pg.ScatterPlotItem(brush=pg.mkBrush(158, 231, 195, 215), pen=None)
        self.kaon_scatter = pg.ScatterPlotItem(brush=pg.mkBrush(246, 196, 83, 225), pen=None)
        self.proton_scatter = pg.ScatterPlotItem(brush=pg.mkBrush(185, 167, 255, 230), pen=None)
        self.plot.addItem(self.pion_scatter)
        self.plot.addItem(self.kaon_scatter)
        self.plot.addItem(self.proton_scatter)
        self.flow_slow = pg.PlotDataItem(pen=pg.mkPen("#69a7ff", width=1))
        self.flow_medium = pg.PlotDataItem(pen=pg.mkPen("#8ee3b5", width=1))
        self.flow_fast = pg.PlotDataItem(pen=pg.mkPen("#f6c453", width=1))
        self.plot.addItem(self.flow_slow)
        self.plot.addItem(self.flow_medium)
        self.plot.addItem(self.flow_fast)
        layout.addWidget(self.plot, 1)

        panel = QtWidgets.QFrame()
        panel.setFixedWidth(360)
        panel.setStyleSheet("QFrame { background:#101817; color:#eef7f4; } QLabel { color:#eef7f4; font-family:'Segoe UI'; }")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("PySide Field Simulator")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        panel_layout.addWidget(title)
        self.stage_label = QtWidgets.QLabel("")
        self.stage_label.setStyleSheet("font-size: 16px; font-weight: 700; color:#f6c453;")
        panel_layout.addWidget(self.stage_label)

        buttons = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_running)
        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        buttons.addWidget(self.play_button)
        buttons.addWidget(reset_button)
        panel_layout.addLayout(buttons)

        self.time_slider = self._slider(0, 1000, 0, self.set_progress)
        self.impact_slider = self._slider(0, 120, round(self.impact_parameter * 10), self.set_impact)
        self.speed_slider = self._slider(1, 20, round(self.speed * 10), self.set_speed)
        panel_layout.addWidget(QtWidgets.QLabel("time"))
        panel_layout.addWidget(self.time_slider)
        panel_layout.addWidget(QtWidgets.QLabel("impact parameter b"))
        panel_layout.addWidget(self.impact_slider)
        panel_layout.addWidget(QtWidgets.QLabel("speed"))
        panel_layout.addWidget(self.speed_slider)

        self.readouts: dict[str, QtWidgets.QLabel] = {}
        for key in (
            "mode",
            "equation of state",
            "tau",
            "temperature",
            "radial flow",
            "mean cs2",
            "centrality",
            "eccentricity",
            "Npart / Ncoll",
            "spectators",
            "multiplicity",
            "pi / K / p",
            "hadron v2",
            "mean pT",
            "QGP / mixed / hadron",
            "physics checks",
        ):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(key))
            value = QtWidgets.QLabel("0")
            value.setAlignment(QtCore.Qt.AlignRight)
            value.setStyleSheet("font-size: 15px; font-weight: 700;")
            row.addWidget(value)
            panel_layout.addLayout(row)
            self.readouts[key] = value
        self.phase_plot = pg.PlotWidget()
        self.phase_plot.setFixedHeight(150)
        self.phase_plot.setBackground("#101817")
        self.phase_plot.setMouseEnabled(x=False, y=False)
        self.phase_plot.hideAxis("bottom")
        self.phase_plot.hideAxis("left")
        self.phase_plot.setMenuEnabled(False)
        self.phase_plot.setYRange(0, 18)
        self.qgp_curve = self.phase_plot.plot(pen=pg.mkPen("#ff5a3d", width=2))
        self.mixed_curve = self.phase_plot.plot(pen=pg.mkPen("#f6c453", width=2))
        self.hadron_curve = self.phase_plot.plot(pen=pg.mkPen("#9ee7c3", width=2))
        self.phase_marker = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#eef7f4", width=1))
        self.phase_plot.addItem(self.phase_marker)
        panel_layout.addWidget(self.phase_plot)
        panel_layout.addStretch(1)
        layout.addWidget(panel)

    def _slider(self, minimum: int, maximum: int, value: int, callback) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        return slider

    def toggle_running(self) -> None:
        self.running = not self.running
        self.play_button.setText("Pause" if self.running else "Play")

    def reset(self) -> None:
        self.progress = 0.0
        self.running = True
        self.play_button.setText("Pause")
        self.time_slider.setValue(0)

    def set_progress(self, value: int) -> None:
        self.progress = value / 1000.0
        self.running = False
        self.play_button.setText("Play")
        self.update_view()

    def set_impact(self, value: int) -> None:
        self.impact_parameter = value / 10.0
        self.phase_cache_key = None
        self.update_view()

    def set_speed(self, value: int) -> None:
        self.speed = value / 10.0

    def tick(self) -> None:
        if self.running:
            self.progress = (self.progress + 0.0024 * self.speed) % 1.0
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(round(self.progress * 1000))
            self.time_slider.blockSignals(False)
            self.update_view()

    def should_use_hydro_cache(self) -> bool:
        if self.hydro_cache is None:
            return False
        return abs(self.impact_parameter - self.hydro_cache.impact_parameter) <= 0.05

    def current_state(self) -> tuple[FieldState, bool]:
        use_hydro_cache = self.should_use_hydro_cache()
        if use_hydro_cache and self.hydro_cache is not None:
            return compute_hydro_cached_field(self.hydro_cache, self.progress), True
        return compute_field(self.progress, self.impact_parameter), False

    def update_view(self) -> None:
        state, use_hydro_cache = self.current_state()
        self.update_phase_plot(state, use_hydro_cache)
        tc = TC_GEV
        tfo = TFO_GEV
        tmax = max(float(state.temperature.max()), 1.0e-6)
        formation = smoothstep(state.progress, 0.24, 0.52)
        thermal_opacity = np.clip((state.temperature - 0.055) / max(tmax - 0.055, 1.0e-6), 0, 1) ** 1.30
        thermal_opacity *= formation
        qgp = np.clip((state.temperature - tc) / 0.18, 0, 1) * thermal_opacity
        decoupling = smoothstep(state.progress, 0.70, 0.94)
        mixed = np.clip(1.0 - np.abs(state.temperature - 0.140) / 0.040, 0, 1) * thermal_opacity
        mixed *= 1.0 - decoupling
        hadron_mist = np.clip((tfo - state.temperature) / 0.09, 0, 1) * (state.temperature > 0.055) * thermal_opacity
        hadron = np.maximum(0.62 * state.freezeout, 0.28 * hadron_mist)
        hadron *= 1.0 - decoupling
        freeze_visual = state.freezeout * (1.0 - smoothstep(state.progress, 0.68, 0.95))
        display = np.dstack(
            [
                np.clip(qgp * 1.00 + mixed * 0.95 + freeze_visual * 0.45, 0, 1),
                np.clip(state.nuclei * 0.65 + mixed * 0.72 + hadron * 0.78 + freeze_visual * 0.42, 0, 1),
                np.clip(state.nuclei * 1.00 + hadron * 0.48 + freeze_visual * 0.12, 0, 1),
            ]
        )
        display = soften_rgb(display)
        self.image.setImage((display * 255).astype(np.uint8), levels=(0, 255))
        freeze_mask = freeze_visual > 0.55
        padded_mask = np.pad(freeze_mask, 1, mode="constant", constant_values=False)
        freeze_interior = (
            padded_mask[1:-1, 1:-1]
            & padded_mask[:-2, 1:-1]
            & padded_mask[2:, 1:-1]
            & padded_mask[1:-1, :-2]
            & padded_mask[1:-1, 2:]
        )
        ys, xs = np.where(freeze_mask & ~freeze_interior)
        if len(xs) > 320:
            step = max(1, len(xs) // 320)
            xs = xs[::step]
            ys = ys[::step]
        self.freeze_scatter.setData(xs, ys)
        pion_mask = state.particle_species == 0
        kaon_mask = state.particle_species == 1
        proton_mask = state.particle_species == 2
        self.pion_scatter.setData(state.particle_x[pion_mask], state.particle_y[pion_mask], size=state.particle_size[pion_mask])
        self.kaon_scatter.setData(state.particle_x[kaon_mask], state.particle_y[kaon_mask], size=state.particle_size[kaon_mask])
        self.proton_scatter.setData(state.particle_x[proton_mask], state.particle_y[proton_mask], size=state.particle_size[proton_mask])
        stride = 18
        flow_lines = {
            "slow": ([], []),
            "medium": ([], []),
            "fast": ([], []),
        }
        flow_line_scale = smoothstep(state.progress, 0.34, 0.54) * (1.0 - smoothstep(state.progress, 0.78, 0.94))
        for y in range(8, state.vx.shape[0], stride):
            for x in range(8, state.vx.shape[1], stride):
                local_speed = float(math.hypot(state.vx[y, x], state.vy[y, x]))
                vx = state.vx[y, x] * 14 * flow_line_scale
                vy = state.vy[y, x] * 14 * flow_line_scale
                if abs(vx) + abs(vy) > 0.08:
                    speed_band = "slow" if local_speed < 0.16 else "medium" if local_speed < 0.32 else "fast"
                    lines_x, lines_y = flow_lines[speed_band]
                    lines_x.extend([x, x + vx, np.nan])
                    lines_y.extend([y, y + vy, np.nan])
        self.flow_slow.setData(*flow_lines["slow"])
        self.flow_medium.setData(*flow_lines["medium"])
        self.flow_fast.setData(*flow_lines["fast"])
        self.stage_label.setText(state.stage)
        if use_hydro_cache:
            mode_text = f"hydro cache b={self.hydro_cache.impact_parameter:.1f}" if self.hydro_cache is not None else "hydro cache"
        elif self.hydro_cache is not None:
            mode_text = "analytic b scan"
        else:
            mode_text = "analytic"
        self.readouts["mode"].setText(mode_text)
        eos_text = self.hydro_cache.eos_name if use_hydro_cache and self.hydro_cache is not None else "crossover proxy"
        self.readouts["equation of state"].setText(eos_text)
        self.readouts["tau"].setText(f"{state.tau:.1f} fm/c")
        self.readouts["temperature"].setText(f"{float(state.temperature.max()):.2f}")
        self.readouts["radial flow"].setText(f"{state.radial_flow:.2f}")
        self.readouts["mean cs2"].setText(f"{state.mean_cs2:.3f}")
        self.readouts["centrality"].setText(f"{state.centrality_percent:.0f}%")
        self.readouts["eccentricity"].setText(f"{state.eccentricity:.2f}")
        self.readouts["Npart / Ncoll"].setText(f"{state.n_part} / {state.n_coll}")
        self.readouts["spectators"].setText(str(state.spectators))
        self.readouts["multiplicity"].setText(str(state.multiplicity))
        self.readouts["pi / K / p"].setText(f"{state.pion_count} / {state.kaon_count} / {state.proton_count}")
        self.readouts["hadron v2"].setText(f"{state.emitted_v2:.3f}")
        self.readouts["mean pT"].setText(f"{state.mean_pt:.3f}")
        self.readouts["QGP / mixed / hadron"].setText(
            f"{100 * state.qgp_fraction:.1f} / {100 * state.mixed_fraction:.1f} / {100 * state.hadron_fraction:.1f}%"
        )
        self.readouts["physics checks"].setText(state.physics_status)

    def update_phase_plot(self, state: FieldState, use_hydro_cache: bool) -> None:
        cache_key: object = (
            ("hydro", str(self.hydro_cache.path))
            if use_hydro_cache and self.hydro_cache is not None
            else ("analytic", round(state.impact_parameter, 1))
        )
        if self.phase_cache_key != cache_key:
            if use_hydro_cache and self.hydro_cache is not None:
                xs = self.hydro_cache.progress
                qgp = (100.0 * self.hydro_cache.qgp_fraction).tolist()
                mixed = (100.0 * self.hydro_cache.mixed_fraction).tolist()
                hadron = (100.0 * self.hydro_cache.hadron_fraction).tolist()
            else:
                xs = np.linspace(0.0, 1.0, 48, dtype=np.float32)
                qgp = []
                mixed = []
                hadron = []
                for progress in xs:
                    sample = compute_field(float(progress), state.impact_parameter, grid_size=84, include_particles=False)
                    qgp.append(100.0 * sample.qgp_fraction)
                    mixed.append(100.0 * sample.mixed_fraction)
                    hadron.append(100.0 * sample.hadron_fraction)
            self.qgp_curve.setData(xs, qgp)
            self.mixed_curve.setData(xs, mixed)
            self.hadron_curve.setData(xs, hadron)
            ymax = max(4.0, max(qgp + mixed + hadron) * 1.15)
            self.phase_plot.setYRange(0, ymax)
            self.phase_cache_key = cache_key
        self.phase_marker.setPos(state.progress)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PySide/pyqtgraph heavy-ion collision field simulator.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a non-GUI field computation test.")
    parser.add_argument("--physics-audit", action="store_true", help="Check centrality, phase ordering, and hadron yields.")
    parser.add_argument("--visual-audit", action="store_true", help="Check expansion continuity, centroid drift, and hadron stability.")
    parser.add_argument("--build-hydro-cache", action="store_true", help="Precompute a Torch hydro frame cache and exit.")
    parser.add_argument("--hydro-cache", type=Path, default=None, help="Load a precomputed hydro cache .npz for GUI playback.")
    parser.add_argument("--hydro-impact-parameter", type=float, default=5.7, help="Impact parameter used when building a hydro cache.")
    parser.add_argument("--hydro-grid-size", type=int, default=96, help="Hydro cache transverse grid size.")
    parser.add_argument("--hydro-extent", type=float, default=8.0, help="Hydro cache transverse half-width in fm-like units.")
    parser.add_argument("--hydro-steps", type=int, default=240, help="Hydro time steps for cache generation.")
    parser.add_argument("--hydro-sample-every", type=int, default=4, help="Save one hydro frame every N steps.")
    parser.add_argument("--hydro-seed", type=int, default=7, help="Monte Carlo Glauber seed for hydro cache generation.")
    parser.add_argument("--hydro-target-max-energy", type=float, default=28.0, help="Initial max energy density after normalization.")
    parser.add_argument("--hydro-freezeout-energy", type=float, default=0.35, help="Energy density mapped to T_fo in the cache.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for hydro cache generation.")
    return parser.parse_args(argv)


def run_physics_audit() -> bool:
    failures: list[str] = []
    phase_points = [0.22, 0.34, 0.50, 0.62, 0.72, 0.84, 0.96]
    phase_states = [compute_field(progress, 5.7) for progress in phase_points]
    if phase_states[1].qgp_fraction <= 0.0:
        failures.append("QGP never appears after overlap")
    if len(phase_states[1].particle_x) != 0 or len(phase_states[2].particle_x) != 0:
        failures.append("hadrons emitted before freeze-out")
    if phase_states[-2].qgp_fraction > 0.005:
        failures.append("QGP survives into hadron-gas stage")
    if phase_states[-2].pion_count <= phase_states[-2].kaon_count + phase_states[-2].proton_count:
        failures.append("pions are not the dominant final hadron species")

    centrality_states = [compute_field(0.84, impact) for impact in [0.0, 2.0, 5.7, 9.0, 12.0]]
    for earlier, later in zip(centrality_states, centrality_states[1:]):
        if earlier.n_part < later.n_part:
            failures.append("Npart increases toward peripheral collisions")
        if earlier.multiplicity < later.multiplicity:
            failures.append("multiplicity increases toward peripheral collisions")
        if len(earlier.particle_x) < len(later.particle_x):
            failures.append("visible hadron count increases toward peripheral collisions")
    no_overlap = centrality_states[-1]
    if no_overlap.n_part != 0 or no_overlap.multiplicity != 0 or len(no_overlap.particle_x) != 0:
        failures.append("no-overlap event still creates matter")
    if abs(centrality_states[0].emitted_v2) > 0.04:
        failures.append("central event has large average v2")
    if centrality_states[2].emitted_v2 < -0.005:
        failures.append("mid-central v2 points away from the impact-parameter axis")
    if centrality_states[3].emitted_v2 + 0.03 < centrality_states[2].emitted_v2:
        failures.append("elliptic flow falls sharply toward peripheral collisions")

    species_state = compute_field(0.84, 5.7)
    total_species = max(1, species_state.pion_count + species_state.kaon_count + species_state.proton_count)
    pion_fraction = species_state.pion_count / total_species
    kaon_fraction = species_state.kaon_count / total_species
    proton_fraction = species_state.proton_count / total_species
    if not 0.65 <= pion_fraction <= 0.88:
        failures.append(f"pion fraction outside educational target range: {pion_fraction:.3f}")
    if not 0.05 <= kaon_fraction <= 0.20:
        failures.append(f"kaon fraction outside educational target range: {kaon_fraction:.3f}")
    if not 0.04 <= proton_fraction <= 0.18:
        failures.append(f"proton fraction outside educational target range: {proton_fraction:.3f}")

    print("Physics audit")
    for progress, state in zip(phase_points, phase_states):
        print(
            f"  p={progress:.2f} {state.stage:20s} "
            f"Tmax={state.temperature.max():.3f} "
            f"q/m/h={100 * state.qgp_fraction:.2f}/{100 * state.mixed_fraction:.2f}/{100 * state.hadron_fraction:.2f} "
            f"hadrons={len(state.particle_x):3d} status={state.physics_status}"
        )
    for state in centrality_states:
        print(
            f"  b={state.impact_parameter:4.1f} c={state.centrality_percent:5.1f}% "
            f"Npart={state.n_part:3d} Ncoll={state.n_coll:4d} mult={state.multiplicity:3d} "
            f"hadrons={len(state.particle_x):3d} v2={state.emitted_v2:.3f}"
        )
    if failures:
        print("Audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        return False
    print("Audit passed")
    return True


def run_hydro_cache_audit(cache: HydroFrameCache) -> bool:
    failures: list[str] = []
    phase_points = [0.30, 0.40, 0.50, 0.62, 0.72, 0.84, 0.96]
    states = [compute_hydro_cached_field(cache, progress) for progress in phase_points]
    if cache.n_part > 0 and not any(state.stage == "QGP hydrodynamics" for state in states[:3]):
        failures.append("QGP never appears in hydro cache")
    if any(len(state.particle_x) != 0 for progress, state in zip(phase_points, states) if progress < 0.60):
        failures.append("hadrons emitted before hydro freeze-out window")
    if states[-1].qgp_fraction > 0.005:
        failures.append("QGP survives at final hydro cache frame")
    if states[-1].mixed_fraction > 0.012 or states[-1].stage != "hadron gas":
        failures.append("hydro cache ends before the hadron-gas stage")
    for state in states:
        if state.physics_status != "OK":
            failures.append(f"physics check at p={state.progress:.2f}: {state.physics_status}")
    species_state = states[-2]
    total_species = max(1, species_state.pion_count + species_state.kaon_count + species_state.proton_count)
    if species_state.pion_count / total_species < 0.65:
        failures.append("pions are not dominant in cached hydro particlization")
    if species_state.emitted_v2 < -0.01:
        failures.append("cached hadron v2 points away from the impact-parameter axis")

    print(f"Hydro cache audit: {cache.path}")
    for state in states:
        print(
            f"  p={state.progress:.2f} {state.stage:20s} "
            f"Tmax={state.temperature.max():.3f} "
            f"q/m/h={100 * state.qgp_fraction:.2f}/{100 * state.mixed_fraction:.2f}/{100 * state.hadron_fraction:.2f} "
            f"hadrons={len(state.particle_x):3d} status={state.physics_status}"
        )
    print(
        f"  cache: b={cache.impact_parameter:.1f}, Npart={cache.n_part}, "
        f"Ncoll={cache.n_coll}, multiplicity={cache.multiplicity}, frames={len(cache.progress)}"
    )
    if failures:
        print("Hydro cache audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        return False
    print("Hydro cache audit passed")
    return True


def thermal_centroid(state: FieldState, threshold: float = 0.055) -> tuple[float, float]:
    weights = np.maximum(state.temperature - threshold, 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 1.0e-12:
        return math.nan, math.nan
    yy, xx = np.indices(state.temperature.shape)
    return float(np.sum(xx * weights) / total_weight), float(np.sum(yy * weights) / total_weight)


def particle_signature(state: FieldState) -> set[tuple[float, float, int]]:
    return {
        (round(float(x), 2), round(float(y), 2), int(species))
        for x, y, species in zip(state.particle_x, state.particle_y, state.particle_species)
    }


def run_visual_audit(cache: HydroFrameCache | None = None) -> bool:
    failures: list[str] = []
    grid_size = cache.grid_size if cache is not None else 96

    def state_at(progress: float, include_particles: bool = False) -> FieldState:
        if cache is not None:
            return compute_hydro_cached_field(cache, progress, include_particles=include_particles)
        return compute_field(progress, 5.7, grid_size=grid_size, include_particles=include_particles)

    expansion_points = [0.28, 0.30, 0.32, 0.34, 0.36, 0.40, 0.44, 0.48, 0.52, 0.58]
    expansion_states = [state_at(progress) for progress in expansion_points]
    areas = [int(np.count_nonzero(state.temperature > 0.08)) for state in expansion_states]
    centroids = [thermal_centroid(state) for state in expansion_states]

    for index in range(1, len(expansion_states)):
        previous_area = areas[index - 1]
        current_area = areas[index]
        if previous_area >= 100 and current_area / previous_area > 2.60:
            failures.append(
                f"thermal area jumps {current_area / previous_area:.2f}x "
                f"between p={expansion_points[index - 1]:.2f} and p={expansion_points[index]:.2f}"
            )
        previous_center = centroids[index - 1]
        current_center = centroids[index]
        if previous_area >= 100 and all(math.isfinite(value) for value in (*previous_center, *current_center)):
            drift = math.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1])
            if drift / grid_size > 0.018:
                failures.append(
                    f"thermal centroid drifts {drift:.2f} cells "
                    f"between p={expansion_points[index - 1]:.2f} and p={expansion_points[index]:.2f}"
                )

    hadron_points = [0.84, 0.90, 0.96]
    hadron_states = [state_at(progress, include_particles=True) for progress in hadron_points]
    signatures = [particle_signature(state) for state in hadron_states]
    for index in range(1, len(hadron_states)):
        previous = signatures[index - 1]
        current = signatures[index]
        if len(current) < len(previous):
            failures.append(
                f"visible hadron count decreases between p={hadron_points[index - 1]:.2f} "
                f"and p={hadron_points[index]:.2f}"
            )
        retention = len(previous & current) / max(1, len(previous))
        if previous and retention < 0.95:
            failures.append(
                f"only {100.0 * retention:.1f}% of existing hadrons retain position "
                f"between p={hadron_points[index - 1]:.2f} and p={hadron_points[index]:.2f}"
            )

    mode = f"hydro cache ({cache.eos_name})" if cache is not None else "analytic fallback"
    print(f"Visual continuity audit: {mode}")
    for progress, area, center in zip(expansion_points, areas, centroids):
        print(f"  p={progress:.2f} active_area={area:5d} centroid=({center[0]:6.2f}, {center[1]:6.2f})")
    print("  hadrons: " + ", ".join(f"p={p:.2f}:{len(state.particle_x)}" for p, state in zip(hadron_points, hadron_states)))
    if failures:
        print("Visual audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        return False
    print("Visual audit passed")
    return True


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.build_hydro_cache:
        output = args.hydro_cache or DEFAULT_HYDRO_CACHE
        generate_hydro_cache(
            output=output,
            impact_parameter=args.hydro_impact_parameter,
            grid_size=args.hydro_grid_size,
            extent=args.hydro_extent,
            steps=args.hydro_steps,
            sample_every=args.hydro_sample_every,
            seed=args.hydro_seed,
            cpu=args.cpu,
            target_max_energy=args.hydro_target_max_energy,
            freezeout_energy=args.hydro_freezeout_energy,
        )
        return
    hydro_cache = HydroFrameCache.load(args.hydro_cache) if args.hydro_cache is not None else None
    if args.smoke_test:
        state = compute_hydro_cached_field(hydro_cache, 0.84) if hydro_cache is not None else compute_field(0.84, 5.7)
        print(
            "smoke ok",
            state.stage,
            f"Tmax={state.temperature.max():.3f}",
            f"flow={state.radial_flow:.3f}",
            f"hadrons={len(state.particle_x)}",
            f"v2={state.emitted_v2:.3f}",
        )
        return
    if args.physics_audit:
        if hydro_cache is not None:
            raise SystemExit(0 if run_hydro_cache_audit(hydro_cache) else 1)
        raise SystemExit(0 if run_physics_audit() else 1)
    if args.visual_audit:
        raise SystemExit(0 if run_visual_audit(hydro_cache) else 1)
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    window = PySideHICApp(hydro_cache=hydro_cache)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
