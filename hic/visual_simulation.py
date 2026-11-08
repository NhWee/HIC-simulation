import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency in this Python environment: torch\n"
        "Set up the project environment first:\n"
        r"  .\setup.ps1"
    ) from error


def sample_woods_saxon_nucleons(
    mass_number: int,
    radius: float,
    skin_depth: float,
    rng: np.random.Generator,
) -> np.ndarray:
    max_radius = radius + 8.0 * skin_depth
    positions: list[list[float]] = []
    while len(positions) < mass_number:
        batch = max(1024, (mass_number - len(positions)) * 10)
        xyz = rng.uniform(-max_radius, max_radius, size=(batch, 3))
        radial_distance = np.linalg.norm(xyz, axis=1)
        accept_density = 1.0 / (1.0 + np.exp((radial_distance - radius) / max(skin_depth, 1.0e-6)))
        accept = (radial_distance <= max_radius) & (rng.random(batch) < accept_density)
        positions.extend(xyz[accept, :2].tolist())

    transverse = np.asarray(positions[:mass_number], dtype=np.float32)
    transverse -= np.mean(transverse, axis=0, keepdims=True)
    return transverse


def tag_glauber_event(
    nucleus_a: np.ndarray,
    nucleus_b: np.ndarray,
    impact_parameter: float,
    inelastic_cross_section: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifted_a = nucleus_a + np.array([0.0, -0.5 * impact_parameter], dtype=np.float32)
    shifted_b = nucleus_b + np.array([0.0, 0.5 * impact_parameter], dtype=np.float32)
    dx = shifted_a[:, 0, None] - shifted_b[None, :, 0]
    dy = shifted_a[:, 1, None] - shifted_b[None, :, 1]
    max_distance2 = max(inelastic_cross_section, 1.0e-6) / math.pi
    pairs = np.argwhere(dx * dx + dy * dy <= max_distance2)

    participant_a = np.zeros(nucleus_a.shape[0], dtype=bool)
    participant_b = np.zeros(nucleus_b.shape[0], dtype=bool)
    if pairs.size > 0:
        participant_a[np.unique(pairs[:, 0])] = True
        participant_b[np.unique(pairs[:, 1])] = True

    collision_centers = 0.5 * (shifted_a[pairs[:, 0]] + shifted_b[pairs[:, 1]]) if pairs.size > 0 else np.empty((0, 2), dtype=np.float32)
    participants = np.vstack([shifted_a[participant_a], shifted_b[participant_b]]).astype(np.float32)
    return participants, collision_centers.astype(np.float32), pairs


def participant_plane(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    qx = float(np.mean(centered[:, 0] ** 2 - centered[:, 1] ** 2))
    qy = float(np.mean(2.0 * centered[:, 0] * centered[:, 1]))
    return 0.5 * math.atan2(qy, qx)


def torch_grid(grid_size: int, extent: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    axis = torch.linspace(-extent, extent, grid_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = float(axis[1] - axis[0])
    return xx, yy, dx


def deposit_points_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    points: np.ndarray,
    width: float,
    normalization: float,
    device: torch.device,
    chunk_size: int = 96,
) -> torch.Tensor:
    if points.size == 0:
        return torch.zeros_like(xx)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    density = torch.zeros_like(xx)
    sigma2 = max(width * width, 1.0e-6)
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start : start + chunk_size]
        dx = xx.unsqueeze(0) - chunk[:, 0].view(-1, 1, 1)
        dy = yy.unsqueeze(0) - chunk[:, 1].view(-1, 1, 1)
        density += torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma2)).sum(dim=0)
    total = torch.sum(density)
    if float(total.detach().cpu()) > 0.0:
        density = density * (normalization / (total + 1.0e-8))
    return density


def gaussian_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    sigma_x: float,
    sigma_y: float,
) -> torch.Tensor:
    return torch.exp(-(((xx - cx) ** 2) / (2.0 * sigma_x * sigma_x) + ((yy - cy) ** 2) / (2.0 * sigma_y * sigma_y)))


def eos_speed_of_sound2(temperature: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    qgp = torch.sigmoid((temperature - args.critical_temperature) / 0.018)
    hadron = torch.sigmoid((args.freezeout_temperature - temperature) / 0.018)
    crossover_dip = torch.exp(-((temperature - args.critical_temperature) ** 2) / (2.0 * 0.030 * 0.030))
    cs2 = 0.18 + 0.12 * qgp - 0.075 * crossover_dip - 0.035 * hadron
    return torch.clamp(cs2, min=0.055, max=0.33)


def laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        torch.roll(field, 1, 0)
        + torch.roll(field, -1, 0)
        + torch.roll(field, 1, 1)
        + torch.roll(field, -1, 1)
        - 4.0 * field
    ) / (dx * dx)


def gradient_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, -1, 1) - torch.roll(field, 1, 1)) / (2.0 * dx)


def gradient_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, -1, 0) - torch.roll(field, 1, 0)) / (2.0 * dx)


def divergence_x_y(fx: torch.Tensor, fy: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(fx, dx) + gradient_y(fy, dx)


def downsample(field: torch.Tensor, output_size: int) -> np.ndarray:
    if field.shape[0] == output_size:
        return field.detach().cpu().numpy()
    tensor = field.detach().float().unsqueeze(0).unsqueeze(0)
    pooled = torch.nn.functional.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return pooled.squeeze().cpu().numpy()


def stage_for(tau: float, qgp_fraction: float, mixed_fraction: float, hadron_fraction: float, particles: int, args: argparse.Namespace) -> str:
    if tau < 0.0:
        return "incoming nuclei"
    if tau < args.tau0:
        return "pre-equilibrium"
    if qgp_fraction > 0.018:
        return "QGP hydrodynamics"
    if mixed_fraction > 0.012:
        return "mixed phase"
    if particles > 0 or hadron_fraction > 0.020:
        return "hadron gas"
    return "cooling plasma"


def sample_particles_from_freezeout(
    weights: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    rng: np.random.Generator,
    count: int,
    eps2: float,
    plane: float,
) -> list[dict[str, float]]:
    flat = np.clip(weights.ravel(), 0.0, None)
    total = float(flat.sum())
    if total <= 0.0:
        return []
    probabilities = flat / total
    count = min(count, max(1, int(np.count_nonzero(flat) * 0.65)))
    indices = rng.choice(flat.size, size=count, replace=True, p=probabilities)
    ys, xs = np.unravel_index(indices, weights.shape)
    particles = []
    for x_idx, y_idx in zip(xs, ys):
        x = float(xx[y_idx, x_idx])
        y = float(yy[y_idx, x_idx])
        radial = math.atan2(y, x)
        local_flow = math.atan2(float(vy[y_idx, x_idx]), float(vx[y_idx, x_idx])) if abs(vx[y_idx, x_idx]) + abs(vy[y_idx, x_idx]) > 1.0e-5 else radial
        flow_angle = 0.55 * local_flow + 0.45 * radial
        elliptic_bias = 0.42 * min(eps2, 0.65) * math.sin(2.0 * (plane - flow_angle))
        angle = flow_angle + elliptic_bias + rng.normal(0.0, 0.24)
        speed = float(rng.uniform(0.018, 0.060))
        particles.append(
            {
                "x": x,
                "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": float(rng.uniform(0.65, 1.0)),
                "size": float(rng.uniform(1.1, 2.7)),
            }
        )
    return particles


def emitted_particle_observables(particles: list[dict[str, float]], event_plane: float) -> tuple[float, float]:
    if not particles:
        return 0.0, 0.0
    speeds = []
    harmonics = []
    for particle in particles:
        px = particle["vx"]
        py = particle["vy"]
        angle = math.atan2(py, px)
        speeds.append(math.hypot(px, py))
        harmonics.append(math.cos(2.0 * (angle - event_plane)))
    return float(np.mean(speeds)), float(np.mean(harmonics))


def simulate_frames(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(0)

    xx, yy, dx = torch_grid(args.grid_size, args.extent, device)
    x_cpu = np.linspace(-args.extent, args.extent, args.output_grid_size, dtype=np.float32)
    yy_cpu, xx_cpu = np.meshgrid(x_cpu, x_cpu)

    nucleons_a = sample_woods_saxon_nucleons(args.mass_number, args.radius, args.skin_depth, rng)
    nucleons_b = sample_woods_saxon_nucleons(args.mass_number, args.radius, args.skin_depth, rng)
    participants, collision_centers, collision_pairs = tag_glauber_event(
        nucleons_a,
        nucleons_b,
        args.impact_parameter,
        args.inelastic_cross_section,
    )
    n_part = int(participants.shape[0])
    n_coll = int(collision_pairs.shape[0])
    spectators = max(0, 2 * args.mass_number - n_part)
    centrality_proxy = min(100.0, 100.0 * args.impact_parameter * args.impact_parameter / max((2.0 * args.radius) ** 2, 1.0e-8))
    plane = participant_plane(participants)

    participant_source = deposit_points_torch(xx, yy, participants, args.nucleon_width, max(n_part, 1), device)
    binary_source = deposit_points_torch(xx, yy, collision_centers, args.nucleon_width * 0.72, max(n_coll, 1), device)
    glauber_source = ((1.0 - args.binary_fraction) * 0.5 * participant_source) + (args.binary_fraction * binary_source)

    if args.fluctuations > 0.0:
        shape = max(1.0 / (args.fluctuations * args.fluctuations), 1.0)
        noise_np = rng.gamma(shape=shape, scale=1.0 / shape, size=(args.grid_size, args.grid_size)).astype(np.float32)
        noise = torch.as_tensor(noise_np, device=device)
        glauber_source = glauber_source * noise / (torch.mean(noise) + 1.0e-8)

    e = torch.zeros_like(xx)
    vx = torch.zeros_like(xx)
    vy = torch.zeros_like(xx)
    freezeout_seen = torch.zeros_like(xx, dtype=torch.bool)
    freezeout_recent = torch.zeros_like(xx)
    particles: list[dict[str, float]] = []
    frames = []
    start = args.extent * 0.82
    peak_eps2 = 0.0
    global_max_energy = 1.0e-8
    global_max_temperature = 1.0e-8
    initial_source_total = float(torch.sum(glauber_source).detach().cpu())

    for step in range(args.steps):
        t_lab = (step - args.crossing_step) * args.dt
        tau = max(0.0, t_lab)
        progress = step / max(args.steps - 1, 1)
        crossing_progress = min(max(step / max(args.crossing_step * 2, 1), 0.0), 1.0)
        center_a_x = torch.tensor(-start + 2.0 * start * crossing_progress, device=device)
        center_b_x = torch.tensor(start - 2.0 * start * crossing_progress, device=device)
        overlap = math.exp(-((float(center_a_x - center_b_x)) ** 2) / (2.0 * args.collision_width * args.collision_width))

        source_pulse = math.exp(-((t_lab - 0.12) ** 2) / (2.0 * args.source_duration * args.source_duration))
        e = e + args.normalization * source_pulse * overlap * glauber_source * args.dt
        temperature_for_eos = args.temperature_scale * torch.pow(torch.clamp(e, min=0.0) / max(args.energy_scale, 1.0e-8), 0.25)

        if tau > args.tau0:
            cs2 = eos_speed_of_sound2(temperature_for_eos, args)
            pressure = cs2 * e
            enthalpy = e + pressure + args.energy_floor
            ax = -gradient_x(pressure, dx) / enthalpy
            ay = -gradient_y(pressure, dx) / enthalpy
            vx = torch.clamp((vx + ax * args.dt) * (1.0 - args.viscosity * args.dt), -args.max_flow, args.max_flow)
            vy = torch.clamp((vy + ay * args.dt) * (1.0 - args.viscosity * args.dt), -args.max_flow, args.max_flow)
            flux_div = divergence_x_y(e * vx, e * vy, dx)
            longitudinal = (1.0 + float(torch.mean(cs2).detach().cpu())) / max(tau, args.tau0)
            e = e - args.dt * flux_div
            e = e + args.diffusion * laplacian(e, dx) * args.dt
            e = e * math.exp(-longitudinal * args.dt)
            e = torch.clamp(e, min=0.0)

        temperature = args.temperature_scale * torch.pow(torch.clamp(e, min=0.0) / max(args.energy_scale, 1.0e-8), 0.25)
        cs2_now = eos_speed_of_sound2(temperature, args)
        active = e > args.energy_floor
        newly_freezing = (temperature < args.freezeout_temperature) & (temperature > args.hadron_temperature) & active & (~freezeout_seen) & (tau > args.tau0)
        freezeout_seen |= newly_freezing
        freezeout_recent = torch.where(newly_freezing, torch.clamp(e, min=0.0), freezeout_recent * 0.86)

        positive = torch.clamp(e, min=0.0)
        total = torch.sum(positive) + 1.0e-8
        r2 = xx * xx + yy * yy
        eps_real = torch.sum(positive * (yy * yy - xx * xx)) / (torch.sum(positive * r2) + 1.0e-8)
        eps_imag = torch.sum(positive * (2.0 * xx * yy)) / (torch.sum(positive * r2) + 1.0e-8)
        eps2 = float(torch.sqrt(eps_real * eps_real + eps_imag * eps_imag).detach().cpu())
        peak_eps2 = max(peak_eps2, eps2)
        max_energy = float(torch.max(positive).detach().cpu())
        max_temperature = float(torch.max(temperature).detach().cpu())
        freeze_fraction = float(torch.mean(newly_freezing.float()).detach().cpu())
        cumulative_freeze_fraction = float(torch.mean(freezeout_seen.float()).detach().cpu())
        mean_cs2 = float(torch.mean(cs2_now[active]).detach().cpu()) if bool(torch.any(active).detach().cpu()) else 0.0
        flow_speed = torch.sqrt(vx * vx + vy * vy)
        mean_radial_flow = float((torch.sum(flow_speed * positive) / total).detach().cpu())
        entropy_proxy = float(torch.sum(torch.pow(torch.clamp(temperature, min=0.0), 3.0)).detach().cpu())
        global_max_energy = max(global_max_energy, max_energy)
        global_max_temperature = max(global_max_temperature, max_temperature)

        if tau > args.tau0 and len(particles) < args.particles and (step % args.particle_every == 0):
            recent_cpu = downsample(freezeout_recent, args.output_grid_size)
            if float(np.sum(recent_cpu)) > 0.0:
                vx_cpu = downsample(vx, args.output_grid_size)
                vy_cpu = downsample(vy, args.output_grid_size)
                new_count = min(args.particles - len(particles), args.particles_per_sample)
                particles.extend(sample_particles_from_freezeout(recent_cpu, xx_cpu, yy_cpu, vx_cpu, vy_cpu, rng, new_count, peak_eps2, plane))

        for particle in particles:
            particle["x"] += particle["vx"] * args.extent * args.dt * args.particle_motion_scale
            particle["y"] += particle["vy"] * args.extent * args.dt * args.particle_motion_scale
            particle["life"] *= 0.9985

        if step % args.frame_every == 0 or step == args.steps - 1:
            temp_cpu = downsample(temperature, args.output_grid_size)
            energy_cpu = downsample(positive, args.output_grid_size)
            freeze_cpu = downsample(freezeout_recent, args.output_grid_size)
            vx_cpu = downsample(vx, args.output_grid_size)
            vy_cpu = downsample(vy, args.output_grid_size)
            active_cpu = energy_cpu > args.energy_floor
            qgp_fraction = float(np.mean((temp_cpu >= args.critical_temperature) & active_cpu))
            mixed_fraction = float(np.mean((temp_cpu < args.critical_temperature) & (temp_cpu >= args.freezeout_temperature) & active_cpu))
            hadron_fraction = float(np.mean((temp_cpu < args.freezeout_temperature) & (temp_cpu >= args.hadron_temperature) & active_cpu))
            temp_norm = np.clip(temp_cpu / max(args.temperature_scale, 1.0e-8), 0.0, 1.8)
            energy_norm = np.clip(np.sqrt(energy_cpu / max(global_max_energy, 1.0e-8)), 0.0, 1.0)
            freeze_norm = np.clip(freeze_cpu / max(float(np.max(freeze_cpu)), 1.0e-8), 0.0, 1.0)
            stride = max(args.output_grid_size // args.flow_vectors, 1)
            flow_vectors = []
            for iy in range(stride // 2, args.output_grid_size, stride):
                for ix in range(stride // 2, args.output_grid_size, stride):
                    speed = math.hypot(float(vx_cpu[iy, ix]), float(vy_cpu[iy, ix]))
                    if speed > 0.012 and temp_cpu[iy, ix] > args.hadron_temperature:
                        flow_vectors.append(
                            [
                                round(float(xx_cpu[iy, ix]), 3),
                                round(float(yy_cpu[iy, ix]), 3),
                                round(float(vx_cpu[iy, ix]), 4),
                                round(float(vy_cpu[iy, ix]), 4),
                            ]
                        )

            rms = float(torch.sqrt(torch.sum(positive * r2) / total).detach().cpu())
            visible_particles = [
                particle
                for particle in particles
                if particle["life"] > 0.08 and -args.extent * 1.25 <= particle["x"] <= args.extent * 1.25 and -args.extent * 1.25 <= particle["y"] <= args.extent * 1.25
            ]
            mean_pt_proxy, emitted_v2 = emitted_particle_observables(visible_particles, plane)
            stage = stage_for(t_lab, qgp_fraction, mixed_fraction, hadron_fraction, len(visible_particles), args)
            frames.append(
                {
                    "time": round(float(t_lab), 4),
                    "tau": round(float(tau), 4),
                    "stage": stage,
                    "nuclei": [
                        {"x": float(center_a_x.detach().cpu()), "y": -0.5 * args.impact_parameter, "r": args.radius, "opacity": max(0.10, 1.0 - progress * 0.9)},
                        {"x": float(center_b_x.detach().cpu()), "y": 0.5 * args.impact_parameter, "r": args.radius, "opacity": max(0.10, 1.0 - progress * 0.9)},
                    ],
                    "temperature": temp_norm.round(4).tolist(),
                    "energy": energy_norm.round(4).tolist(),
                    "freezeout": freeze_norm.round(4).tolist(),
                    "flow": flow_vectors[: args.max_flow_vectors],
                    "particles": visible_particles[:],
                    "maxEnergy": max_energy,
                    "maxTemperature": max_temperature,
                    "eps2": eps2,
                    "rms": rms,
                    "nPart": n_part,
                    "nColl": n_coll,
                    "spectators": spectators,
                    "centralityProxy": centrality_proxy,
                    "eventPlane": plane,
                    "multiplicityProxy": int(round(float(total.detach().cpu()) * args.multiplicity_scale)),
                    "freezeFraction": freeze_fraction,
                    "cumulativeFreezeFraction": cumulative_freeze_fraction,
                    "qgpFraction": qgp_fraction,
                    "mixedFraction": mixed_fraction,
                    "hadronFraction": hadron_fraction,
                    "meanCs2": mean_cs2,
                    "meanRadialFlow": mean_radial_flow,
                    "entropyProxy": entropy_proxy,
                    "meanPtProxy": mean_pt_proxy,
                    "emittedV2": emitted_v2,
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "extent": args.extent,
        "gridSize": args.output_grid_size,
        "frameMs": args.frame_duration_ms,
        "globalMaxEnergy": global_max_energy,
        "globalMaxTemperature": global_max_temperature,
        "frames": frames,
        "settings": {
            "device": str(device),
            "impactParameter": args.impact_parameter,
            "massNumber": args.mass_number,
            "nPart": n_part,
            "nColl": n_coll,
            "spectators": spectators,
            "centralityProxy": centrality_proxy,
            "binaryFraction": args.binary_fraction,
            "tau0": args.tau0,
            "criticalTemperature": args.critical_temperature,
            "freezeoutTemperature": args.freezeout_temperature,
            "hadronTemperature": args.hadron_temperature,
            "temperatureScale": args.temperature_scale,
            "initialSourceTotal": initial_source_total,
        },
    }


def build_html(payload: dict[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heavy-Ion Collision GPU Visual Simulation</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #050706;
  --panel: #101615;
  --line: #2a3835;
  --text: #eef7f4;
  --muted: #9fb1ad;
  --cyan: #69d2e7;
  --qgp: #f04736;
  --mixed: #f6c453;
  --hadron: #a7f3d0;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  background: #050706;
  color: var(--text);
  font-family: Inter, Segoe UI, system-ui, sans-serif;
}}
.app {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 348px;
  gap: 14px;
  min-height: 100vh;
  padding: 14px;
}}
.stage {{
  position: relative;
  min-height: calc(100vh - 28px);
  overflow: hidden;
  border: 1px solid var(--line);
  background: #010202;
}}
canvas {{
  width: 100%;
  height: 100%;
  display: block;
}}
.hud {{
  position: absolute;
  left: 16px;
  top: 14px;
  right: 16px;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  pointer-events: none;
}}
h1 {{
  margin: 0;
  font-size: clamp(22px, 3vw, 42px);
  letter-spacing: 0;
}}
.subtitle {{
  margin-top: 6px;
  color: var(--muted);
  font-size: 14px;
}}
.stage-badge {{
  align-self: flex-start;
  min-width: 168px;
  padding: 9px 11px;
  border: 1px solid var(--line);
  background: rgba(6, 10, 9, 0.78);
  text-align: center;
  font-size: 13px;
}}
.panel {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.box {{
  border: 1px solid var(--line);
  background: var(--panel);
  padding: 11px;
}}
.controls {{
  display: grid;
  grid-template-columns: 42px 42px minmax(0, 1fr);
  gap: 8px;
  align-items: center;
}}
button {{
  width: 42px;
  height: 36px;
  border: 1px solid var(--line);
  background: #16201e;
  color: var(--text);
  cursor: pointer;
  font-size: 17px;
}}
button:hover {{ border-color: var(--cyan); }}
input[type="range"] {{ width: 100%; }}
.row {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  min-height: 30px;
  align-items: center;
}}
.label {{
  color: var(--muted);
  font-size: 12px;
}}
.value {{
  font-variant-numeric: tabular-nums;
  font-size: 16px;
  text-align: right;
}}
.timeline {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
  gap: 4px;
}}
.step {{
  min-height: 42px;
  border: 1px solid var(--line);
  padding: 6px;
  color: var(--muted);
  font-size: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}}
.step.active {{
  color: var(--text);
  border-color: var(--mixed);
  background: #222016;
}}
.meter {{
  height: 9px;
  background: #050706;
  border: 1px solid var(--line);
  overflow: hidden;
}}
.meter > span {{
  display: block;
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--cyan), var(--mixed), var(--qgp));
}}
.legend {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}}
.swatch {{
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 6px;
  vertical-align: -1px;
}}
.chart {{
  width: 100%;
  height: 150px;
}}
.micro {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}
.micro .box {{
  padding: 9px;
}}
.micro .value {{
  font-size: 15px;
}}
@media (max-width: 920px) {{
  .app {{ grid-template-columns: 1fr; }}
  .stage {{ min-height: 66vh; }}
  .hud {{ flex-direction: column; }}
  .stage-badge {{ align-self: flex-end; }}
}}
</style>
</head>
<body>
<main class="app">
  <section class="stage">
    <canvas id="scene"></canvas>
    <div class="hud">
      <div>
        <h1>Heavy-Ion Collision</h1>
        <div class="subtitle">GPU Glauber initial state, 2D hydro-like expansion, freeze-out hypersurface</div>
      </div>
      <div class="stage-badge" id="stageName">incoming nuclei</div>
    </div>
  </section>
  <aside class="panel">
    <div class="box controls">
      <button id="play" title="Play or pause">||</button>
      <button id="restart" title="Restart">R</button>
      <input id="scrub" type="range" min="0" max="1" step="1" value="0" title="Timeline">
    </div>
    <div class="box timeline">
      <div class="step" data-stage="incoming nuclei">incoming</div>
      <div class="step" data-stage="pre-equilibrium">pre-eq</div>
      <div class="step" data-stage="QGP hydrodynamics">QGP hydro</div>
      <div class="step" data-stage="mixed phase">mixed</div>
      <div class="step" data-stage="hadron gas">hadrons</div>
    </div>
    <div class="box">
      <div class="row"><span class="label">proper time tau</span><span class="value" id="tau">0.00</span></div>
      <div class="row"><span class="label">max energy density</span><span class="value" id="energy">0.00</span></div>
      <div class="row"><span class="label">max temperature</span><span class="value" id="temperature">0.00</span></div>
      <div class="row"><span class="label">spatial eps2</span><span class="value" id="eps2">0.000</span></div>
      <div class="row"><span class="label">rms radius</span><span class="value" id="rms">0.00</span></div>
      <div class="row"><span class="label">Npart / Ncoll</span><span class="value" id="geometry">0 / 0</span></div>
      <div class="row"><span class="label">spectators / cent.</span><span class="value" id="centrality">0 / 0%</span></div>
      <div class="row"><span class="label">multiplicity proxy</span><span class="value" id="multiplicity">0</span></div>
      <div class="row"><span class="label">freeze-out area</span><span class="value" id="freezeFraction">0.000</span></div>
      <div class="row"><span class="label">frozen fraction</span><span class="value" id="frozenFraction">0.000</span></div>
      <div class="row"><span class="label">QGP / mixed / hadron</span><span class="value" id="phaseFractions">0 / 0 / 0</span></div>
    </div>
    <div class="micro">
      <div class="box">
        <div class="label">mean radial flow</div>
        <div class="value" id="radialFlow">0.000</div>
      </div>
      <div class="box">
        <div class="label">mean cs2</div>
        <div class="value" id="meanCs2">0.000</div>
      </div>
      <div class="box">
        <div class="label">hadron v2</div>
        <div class="value" id="hadronV2">0.000</div>
      </div>
      <div class="box">
        <div class="label">mean pT proxy</div>
        <div class="value" id="meanPt">0.000</div>
      </div>
    </div>
    <div class="box">
      <div class="label">temperature scale</div>
      <div class="meter"><span id="intensity"></span></div>
    </div>
    <div class="box">
      <canvas class="chart" id="chart"></canvas>
    </div>
    <div class="box legend">
      <div><span class="swatch" style="background:#69d2e7"></span>Lorentz nuclei</div>
      <div><span class="swatch" style="background:#f04736"></span>QGP T field</div>
      <div><span class="swatch" style="background:#f6c453"></span>freeze-out</div>
      <div><span class="swatch" style="background:#a7f3d0"></span>hadrons</div>
      <div><span class="swatch" style="background:#eef7f4"></span>event plane</div>
      <div><span class="swatch" style="background:#9fb1ad"></span>flow vectors</div>
    </div>
  </aside>
</main>
<script>
const data = {data};
const frames = data.frames;
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const chart = document.getElementById("chart");
const chartCtx = chart.getContext("2d");
const play = document.getElementById("play");
const restart = document.getElementById("restart");
const scrub = document.getElementById("scrub");
const stageName = document.getElementById("stageName");
const tau = document.getElementById("tau");
const energy = document.getElementById("energy");
const temperature = document.getElementById("temperature");
const eps2 = document.getElementById("eps2");
const rms = document.getElementById("rms");
const geometry = document.getElementById("geometry");
const centrality = document.getElementById("centrality");
const multiplicity = document.getElementById("multiplicity");
const freezeFraction = document.getElementById("freezeFraction");
const frozenFraction = document.getElementById("frozenFraction");
const phaseFractions = document.getElementById("phaseFractions");
const radialFlow = document.getElementById("radialFlow");
const meanCs2 = document.getElementById("meanCs2");
const hadronV2 = document.getElementById("hadronV2");
const meanPt = document.getElementById("meanPt");
const intensity = document.getElementById("intensity");
const stageSteps = Array.from(document.querySelectorAll(".step"));
let index = 0;
let running = true;
let last = 0;
scrub.max = String(frames.length - 1);

function resizeCanvas(target) {{
  const box = target.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  target.width = Math.max(1, Math.floor(box.width * ratio));
  target.height = Math.max(1, Math.floor(box.height * ratio));
}}

function worldToScreen(x, y) {{
  const extent = data.extent * 1.18;
  const size = Math.min(canvas.width, canvas.height);
  const ox = (canvas.width - size) * 0.5;
  const oy = (canvas.height - size) * 0.5;
  return [ox + (x + extent) / (2 * extent) * size, oy + (extent - y) / (2 * extent) * size];
}}

function colorForCell(t, e, f) {{
  const tc = data.settings.criticalTemperature / data.settings.temperatureScale;
  const tfo = data.settings.freezeoutTemperature / data.settings.temperatureScale;
  const th = data.settings.hadronTemperature / data.settings.temperatureScale;
  if (f > 0.08) return `rgba(246, 196, 83, ${{0.14 + 0.72 * f}})`;
  if (t >= tc) {{
    const hot = Math.min(1, (t - tc) / 0.55);
    return `rgba(${{Math.floor(205 + 50 * hot)}}, ${{Math.floor(42 + 38 * hot)}}, 42, ${{0.24 + 0.68 * Math.max(e, hot)}})`;
  }}
  if (t >= tfo) {{
    const mix = Math.min(1, (t - tfo) / Math.max(tc - tfo, 0.001));
    return `rgba(${{Math.floor(205 + 45 * mix)}}, ${{Math.floor(136 + 56 * mix)}}, 62, ${{0.18 + 0.58 * Math.max(e, mix)}})`;
  }}
  if (t >= th) {{
    const cool = Math.min(1, (t - th) / Math.max(tfo - th, 0.001));
    return `rgba(92, ${{Math.floor(154 + 80 * cool)}}, ${{Math.floor(128 + 72 * cool)}}, ${{0.12 + 0.42 * cool}})`;
  }}
  return `rgba(44, 63, 72, ${{0.06 + 0.20 * e}})`;
}}

function drawFields(frame) {{
  const n = data.gridSize;
  const extent = data.extent;
  const cell = Math.min(canvas.width, canvas.height) / (n * 2.36);
  for (let y = 0; y < n; y++) {{
    for (let x = 0; x < n; x++) {{
      const t = frame.temperature[y][x];
      const e = frame.energy[y][x];
      const f = frame.freezeout[y][x];
      if (t < 0.025 && e < 0.025 && f < 0.08) continue;
      const wx = -extent + (2 * extent * x) / (n - 1);
      const wy = -extent + (2 * extent * y) / (n - 1);
      const [sx, sy] = worldToScreen(wx, wy);
      ctx.fillStyle = colorForCell(t, e, f);
      ctx.fillRect(sx - cell * 0.65, sy - cell * 0.65, cell * 1.35, cell * 1.35);
    }}
  }}
}}

function drawIsotherms(frame) {{
  const n = data.gridSize;
  const extent = data.extent;
  const cell = Math.min(canvas.width, canvas.height) / (n * 2.36);
  const thresholds = [
    [data.settings.criticalTemperature / data.settings.temperatureScale, "rgba(255, 246, 214, 0.72)"],
    [data.settings.freezeoutTemperature / data.settings.temperatureScale, "rgba(246, 196, 83, 0.68)"]
  ];
  ctx.save();
  ctx.lineWidth = 1.0 * (window.devicePixelRatio || 1);
  for (const [threshold, color] of thresholds) {{
    ctx.strokeStyle = color;
    for (let y = 1; y < n - 1; y++) {{
      for (let x = 1; x < n - 1; x++) {{
        const here = frame.temperature[y][x] >= threshold;
        if (here === (frame.temperature[y][x + 1] >= threshold) && here === (frame.temperature[y + 1][x] >= threshold)) continue;
        const wx = -extent + (2 * extent * x) / (n - 1);
        const wy = -extent + (2 * extent * y) / (n - 1);
        const [sx, sy] = worldToScreen(wx, wy);
        ctx.strokeRect(sx - cell * 0.55, sy - cell * 0.55, cell * 1.1, cell * 1.1);
      }}
    }}
  }}
  ctx.restore();
}}

function drawNucleus(nucleus) {{
  const [x, y] = worldToScreen(nucleus.x, nucleus.y);
  const [r2] = worldToScreen(nucleus.x + nucleus.r, nucleus.y);
  const radius = Math.abs(r2 - x);
  const gradient = ctx.createRadialGradient(x, y, radius * 0.10, x, y, radius);
  gradient.addColorStop(0, `rgba(105, 210, 231, ${{0.70 * nucleus.opacity}})`);
  gradient.addColorStop(0.70, `rgba(65, 138, 178, ${{0.35 * nucleus.opacity}})`);
  gradient.addColorStop(1, "rgba(65, 138, 178, 0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.ellipse(x, y, radius * 0.34, radius, 0, 0, Math.PI * 2);
  ctx.fill();
}}

function drawFlow(frame) {{
  ctx.save();
  ctx.strokeStyle = "rgba(159, 177, 173, 0.60)";
  ctx.lineWidth = 1.1 * (window.devicePixelRatio || 1);
  for (const item of frame.flow) {{
    const [x, y, vx, vy] = item;
    const [sx, sy] = worldToScreen(x, y);
    const [tx, ty] = worldToScreen(x + vx * 3.0, y + vy * 3.0);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
  }}
  ctx.restore();
}}

function drawParticles(frame) {{
  for (const particle of frame.particles) {{
    const [x, y] = worldToScreen(particle.x, particle.y);
    ctx.fillStyle = `rgba(167, 243, 208, ${{Math.min(0.90, particle.life)}})`;
    ctx.beginPath();
    ctx.arc(x, y, particle.size * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fill();
  }}
}}

function drawEventPlane(frame) {{
  if (!Number.isFinite(frame.eventPlane)) return;
  const length = data.extent * 0.95;
  const dx = Math.cos(frame.eventPlane) * length;
  const dy = Math.sin(frame.eventPlane) * length;
  const [x1, y1] = worldToScreen(-dx, -dy);
  const [x2, y2] = worldToScreen(dx, dy);
  ctx.save();
  ctx.strokeStyle = "rgba(238, 247, 244, 0.52)";
  ctx.setLineDash([8 * (window.devicePixelRatio || 1), 7 * (window.devicePixelRatio || 1)]);
  ctx.lineWidth = 1.3 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}}

function drawChart() {{
  resizeCanvas(chart);
  chartCtx.clearRect(0, 0, chart.width, chart.height);
  chartCtx.strokeStyle = "#2a3835";
  chartCtx.strokeRect(0.5, 0.5, chart.width - 1, chart.height - 1);
  const maxTemp = Math.max(...frames.map(f => f.maxTemperature), 0.01);
  const maxEps = Math.max(...frames.map(f => f.eps2), 0.01);
  const maxFlow = Math.max(...frames.map(f => f.meanRadialFlow), 0.01);
  const maxV2 = Math.max(...frames.map(f => Math.abs(f.emittedV2)), 0.01);
  function line(values, maxValue, color) {{
    chartCtx.beginPath();
    values.forEach((value, i) => {{
      const x = (i / (values.length - 1)) * chart.width;
      const y = chart.height - 12 - (value / maxValue) * (chart.height - 24);
      if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
    }});
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
    chartCtx.stroke();
  }}
  line(frames.map(f => f.maxTemperature), maxTemp, "#f04736");
  line(frames.map(f => f.eps2), maxEps, "#69d2e7");
  line(frames.map(f => f.meanRadialFlow), maxFlow, "#a7f3d0");
  line(frames.map(f => Math.abs(f.emittedV2)), maxV2, "#f6c453");
  chartCtx.fillStyle = "#f6c453";
  chartCtx.fillRect((index / (frames.length - 1)) * chart.width - 2, 0, 4, chart.height);
  chartCtx.fillStyle = "#9fb1ad";
  chartCtx.font = `${{11 * (window.devicePixelRatio || 1)}}px Segoe UI`;
  chartCtx.fillText("T / eps2 / flow / |v2|", 8 * (window.devicePixelRatio || 1), 16 * (window.devicePixelRatio || 1));
}}

function render() {{
  resizeCanvas(canvas);
  const frame = frames[index];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#010202";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "screen";
  frame.nuclei.forEach(drawNucleus);
  drawFields(frame);
  drawIsotherms(frame);
  drawFlow(frame);
  drawParticles(frame);
  ctx.globalCompositeOperation = "source-over";
  drawEventPlane(frame);
  stageName.textContent = frame.stage;
  tau.textContent = frame.tau.toFixed(2);
  energy.textContent = frame.maxEnergy.toFixed(3);
  temperature.textContent = frame.maxTemperature.toFixed(3);
  eps2.textContent = frame.eps2.toFixed(3);
  rms.textContent = frame.rms.toFixed(2);
  geometry.textContent = `${{frame.nPart}} / ${{frame.nColl}}`;
  centrality.textContent = `${{frame.spectators}} / ${{frame.centralityProxy.toFixed(1)}}%`;
  multiplicity.textContent = String(frame.multiplicityProxy);
  freezeFraction.textContent = frame.freezeFraction.toFixed(4);
  frozenFraction.textContent = frame.cumulativeFreezeFraction.toFixed(3);
  phaseFractions.textContent = `${{(100 * frame.qgpFraction).toFixed(1)}} / ${{(100 * frame.mixedFraction).toFixed(1)}} / ${{(100 * frame.hadronFraction).toFixed(1)}}%`;
  radialFlow.textContent = frame.meanRadialFlow.toFixed(3);
  meanCs2.textContent = frame.meanCs2.toFixed(3);
  hadronV2.textContent = frame.emittedV2.toFixed(3);
  meanPt.textContent = frame.meanPtProxy.toFixed(3);
  intensity.style.width = `${{Math.min(100, (frame.maxTemperature / data.globalMaxTemperature) * 100)}}%`;
  stageSteps.forEach(step => step.classList.toggle("active", step.dataset.stage === frame.stage));
  scrub.value = String(index);
  drawChart();
}}

function tick(stamp) {{
  if (running && stamp - last > data.frameMs) {{
    index = (index + 1) % frames.length;
    last = stamp;
    render();
  }}
  requestAnimationFrame(tick);
}}

play.addEventListener("click", () => {{
  running = !running;
  play.textContent = running ? "||" : ">";
}});
restart.addEventListener("click", () => {{
  index = 0;
  running = true;
  play.textContent = "||";
  render();
}});
scrub.addEventListener("input", () => {{
  index = Number(scrub.value);
  running = false;
  play.textContent = ">";
  render();
}});
window.addEventListener("resize", render);
render();
requestAnimationFrame(tick);
</script>
</body>
</html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GPU-accelerated staged heavy-ion collision visual simulation.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Simulation device.")
    parser.add_argument("--grid-size", type=int, default=224, help="GPU hydrodynamic grid resolution.")
    parser.add_argument("--output-grid-size", type=int, default=112, help="Stored HTML frame grid resolution.")
    parser.add_argument("--steps", type=int, default=960, help="Number of simulation steps.")
    parser.add_argument("--frame-every", type=int, default=8, help="Save one browser frame every N simulation steps.")
    parser.add_argument("--extent", type=float, default=12.0, help="Half-width of transverse plane.")
    parser.add_argument("--dt", type=float, default=0.018, help="Simulation time step.")
    parser.add_argument("--crossing-step", type=int, default=95, help="Step where the nuclei overlap most strongly.")
    parser.add_argument("--impact-parameter", type=float, default=5.7, help="Transverse offset between nuclei.")
    parser.add_argument("--mass-number", type=int, default=208, help="Nuclear mass number.")
    parser.add_argument("--radius", type=float, default=6.55, help="Woods-Saxon nuclear radius.")
    parser.add_argument("--skin-depth", type=float, default=0.55, help="Woods-Saxon surface diffuseness.")
    parser.add_argument("--inelastic-cross-section", type=float, default=4.2, help="Nucleon-nucleon inelastic cross section.")
    parser.add_argument("--nucleon-width", type=float, default=0.42, help="Gaussian nucleon deposition width.")
    parser.add_argument("--binary-fraction", type=float, default=0.16, help="Binary-collision source fraction.")
    parser.add_argument("--fluctuations", type=float, default=0.38, help="Gamma-distributed source hot-spot fluctuations.")
    parser.add_argument("--normalization", type=float, default=6.8, help="Initial energy-density normalization.")
    parser.add_argument("--source-duration", type=float, default=0.18, help="Finite source production duration.")
    parser.add_argument("--collision-width", type=float, default=2.2, help="Beam-overlap width.")
    parser.add_argument("--tau0", type=float, default=0.55, help="Hydrodynamic starting proper time.")
    parser.add_argument("--speed-of-sound2", type=float, default=0.22, help="Equation-of-state pressure slope.")
    parser.add_argument("--viscosity", type=float, default=0.035, help="Flow damping proxy.")
    parser.add_argument("--diffusion", type=float, default=0.030, help="Numerical/viscous smoothing proxy.")
    parser.add_argument("--max-flow", type=float, default=0.72, help="Maximum transverse flow velocity.")
    parser.add_argument("--energy-floor", type=float, default=1.0e-5, help="Small floor for stable divisions.")
    parser.add_argument("--temperature-scale", type=float, default=0.62, help="Temperature scale in GeV-like units.")
    parser.add_argument("--energy-scale", type=float, default=1.0, help="Energy scale for temperature conversion.")
    parser.add_argument("--critical-temperature", type=float, default=0.165, help="QGP crossover temperature threshold.")
    parser.add_argument("--freezeout-temperature", type=float, default=0.118, help="Kinetic freeze-out temperature threshold.")
    parser.add_argument("--hadron-temperature", type=float, default=0.045, help="Lower visible hadron-gas temperature threshold.")
    parser.add_argument("--particles", type=int, default=1400, help="Maximum emitted visual hadrons.")
    parser.add_argument("--particle-every", type=int, default=8, help="Sample particles every N steps after freeze-out begins.")
    parser.add_argument("--particles-per-sample", type=int, default=60, help="Maximum new particles per sampling step.")
    parser.add_argument("--particle-motion-scale", type=float, default=5.0, help="Visual hadron propagation scale.")
    parser.add_argument("--flow-vectors", type=int, default=14, help="Flow-vector grid density.")
    parser.add_argument("--max-flow-vectors", type=int, default=190, help="Maximum flow vectors saved per frame.")
    parser.add_argument("--multiplicity-scale", type=float, default=32.0, help="Scale for multiplicity proxy.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--frame-duration-ms", type=int, default=46, help="Browser animation frame duration.")
    parser.add_argument("--output", type=Path, default=Path("outputs/visual_simulation.html"), help="Output HTML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_grid_size > args.grid_size:
        raise SystemExit("--output-grid-size must be <= --grid-size")
    payload = simulate_frames(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload), encoding="utf-8")
    device = payload["settings"]["device"]
    print(f"Saved GPU staged visual simulation: {args.output}")
    print(f"Device: {device}, frames: {len(payload['frames'])}, Npart={payload['settings']['nPart']}, Ncoll={payload['settings']['nColl']}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        if "CUDA out of memory" in str(error):
            raise SystemExit(
                "CUDA ran out of memory. Try lowering --grid-size or --steps, "
                "or run with --device cpu."
            ) from error
        raise
