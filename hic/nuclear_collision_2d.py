# 2D heavy-ion collision initial-condition and evolution toy model using PyTorch GPU acceleration.
# Initial condition:
#   T_A(x,y,t) and T_B(x,y,t) are moving Woods-Saxon-like thickness profiles.
#   Smooth mode uses the optical Glauber densities
#     wounded = T_A (1 - exp(-sigma T_B)) + T_B (1 - exp(-sigma T_A)),
#     binary  = sigma T_A T_B,
#   both expressed per grid cell like the Monte Carlo mode.
#   Optional Monte Carlo Glauber mode samples fluctuating nucleon positions,
#   tags participants, and deposits binary collision centers.
#   e0(x,y,t) = norm * [(1-alpha) * wounded + alpha * binary]
# Evolution:
#   de/dt = diffusion * laplacian(e) - radial_expansion * div(e * r_hat) - damping * e
# This is a qualitative GPU prototype, not a validated relativistic hydro code.
# You can handle the parameters:
# grid_size, steps, frame_every, mass_number, radius, skin_depth,
# impact_parameter, beam_velocity, collision_width, binary_fraction,
# nucleon_width, inelastic_cross_section, seed, diffusion, radial_expansion,
# damping, cpu, output
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import torch
except ModuleNotFoundError as error:
    if error.name not in {"plotly", "torch"}:
        raise
    raise SystemExit(
        f"Missing dependency in this Python environment: {error.name}\n"
        "Set up the project environment first:\n"
        r"  .\setup.ps1"
        "\nThen use the project launcher:\n"
        r"  .\run_nuclear_collision_2d.ps1"
    ) from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_grid(grid_size: int, extent: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    axis = torch.linspace(-extent, extent, grid_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = float(axis[1] - axis[0])
    return xx, yy, dx


def woods_saxon_thickness(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    radius: float,
    skin_depth: float,
    mass_number: int,
) -> torch.Tensor:
    r = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    density = 1.0 / (1.0 + torch.exp((r - radius) / max(skin_depth, 1.0e-6)))
    norm = mass_number / (torch.sum(density) + 1.0e-8)
    return density * norm


def sample_woods_saxon_nucleons(
    mass_number: int,
    radius: float,
    skin_depth: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_radius = radius + 8.0 * skin_depth
    positions = []

    while len(positions) < mass_number:
        batch = max(512, (mass_number - len(positions)) * 8)
        xyz = rng.uniform(-max_radius, max_radius, size=(batch, 3))
        radial_distance = np.linalg.norm(xyz, axis=1)
        accept_density = 1.0 / (1.0 + np.exp((radial_distance - radius) / max(skin_depth, 1.0e-6)))
        accept = (radial_distance <= max_radius) & (rng.random(batch) < accept_density)
        positions.extend(xyz[accept, :2].tolist())

    transverse = np.asarray(positions[:mass_number], dtype=np.float32)
    transverse -= np.mean(transverse, axis=0, keepdims=True)
    return transverse[:, 0], transverse[:, 1]


def tag_glauber_collisions(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    impact_parameter: float,
    inelastic_cross_section: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_distance2 = max(inelastic_cross_section, 1.0e-6) / np.pi
    dx = ax[:, None] - bx[None, :]
    dy = (ay[:, None] - 0.5 * impact_parameter) - (by[None, :] + 0.5 * impact_parameter)
    collision_pairs = np.argwhere(dx * dx + dy * dy <= max_distance2)

    participant_a = np.zeros(ax.shape[0], dtype=bool)
    participant_b = np.zeros(bx.shape[0], dtype=bool)
    if collision_pairs.size > 0:
        participant_a[np.unique(collision_pairs[:, 0])] = True
        participant_b[np.unique(collision_pairs[:, 1])] = True

    coll_x = 0.5 * (ax[collision_pairs[:, 0]] + bx[collision_pairs[:, 1]]) if collision_pairs.size > 0 else np.empty(0)
    coll_y = (
        0.5
        * (
            ay[collision_pairs[:, 0]]
            - 0.5 * impact_parameter
            + by[collision_pairs[:, 1]]
            + 0.5 * impact_parameter
        )
        if collision_pairs.size > 0
        else np.empty(0)
    )
    return participant_a, participant_b, coll_x.astype(np.float32), coll_y.astype(np.float32)


def gaussian_point_thickness(
    xx: torch.Tensor,
    yy: torch.Tensor,
    point_x: torch.Tensor,
    point_y: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    width: float,
    normalization: float,
) -> torch.Tensor:
    if point_x.numel() == 0:
        return torch.zeros_like(xx)

    sigma2 = max(width * width, 1.0e-6)
    dx = xx.unsqueeze(0) - (center_x + point_x).view(-1, 1, 1)
    dy = yy.unsqueeze(0) - (center_y + point_y).view(-1, 1, 1)
    density = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma2)).sum(dim=0)
    norm = normalization / (torch.sum(density) + 1.0e-8)
    return density * norm


def laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        torch.roll(field, shifts=1, dims=0)
        + torch.roll(field, shifts=-1, dims=0)
        + torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
        - 4.0 * field
    ) / (dx * dx)


def gradient_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=1) - torch.roll(field, shifts=1, dims=1)) / (2.0 * dx)


def gradient_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=0) - torch.roll(field, shifts=1, dims=0)) / (2.0 * dx)


def radial_flux_divergence(e: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor, dx: float) -> torch.Tensor:
    r = torch.sqrt(xx * xx + yy * yy) + 1.0e-6
    ux = xx / r
    uy = yy / r
    return gradient_x(e * ux, dx) + gradient_y(e * uy, dx)


def eccentricities(e: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor) -> tuple[float, float]:
    weight = torch.clamp(e, min=0.0)
    total = torch.sum(weight) + 1.0e-8
    r2 = xx * xx + yy * yy
    phi2_real = torch.sum(weight * (xx * xx - yy * yy)) / (torch.sum(weight * r2) + 1.0e-8)
    phi2_imag = torch.sum(weight * (2.0 * xx * yy)) / (torch.sum(weight * r2) + 1.0e-8)
    eps2 = torch.sqrt(phi2_real * phi2_real + phi2_imag * phi2_imag)
    rms_radius = torch.sqrt(torch.sum(weight * r2) / total)
    return float(eps2.detach().cpu()), float(rms_radius.detach().cpu())


def simulate(
    grid_size: int,
    extent: float,
    steps: int,
    frame_every: int,
    dt: float,
    mass_number: int,
    radius: float,
    skin_depth: float,
    impact_parameter: float,
    beam_velocity: float,
    collision_width: float,
    binary_fraction: float,
    initial_condition: str,
    nucleon_width: float,
    inelastic_cross_section: float,
    seed: int,
    norm: float,
    diffusion: float,
    radial_expansion: float,
    damping: float,
    device: torch.device,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[tuple[float, float, float]], dict[str, int]]:
    xx, yy, dx = make_grid(grid_size, extent, device)
    e = torch.zeros((grid_size, grid_size), device=device)
    frames = []
    observables = []
    start_offset = extent * 0.75
    rng = np.random.default_rng(seed)
    nucleons_a = None
    nucleons_b = None
    participants_a = None
    participants_b = None
    collision_centers = None
    n_part = 0
    n_coll = 0

    if initial_condition == "mc-glauber":
        ax, ay = sample_woods_saxon_nucleons(mass_number, radius, skin_depth, rng)
        bx, by = sample_woods_saxon_nucleons(mass_number, radius, skin_depth, rng)
        participant_a, participant_b, coll_x, coll_y = tag_glauber_collisions(
            ax, ay, bx, by, impact_parameter, inelastic_cross_section
        )
        nucleons_a = (
            torch.as_tensor(ax, device=device),
            torch.as_tensor(ay, device=device),
        )
        nucleons_b = (
            torch.as_tensor(bx, device=device),
            torch.as_tensor(by, device=device),
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
        n_part = int(participant_a.sum() + participant_b.sum())
        n_coll = int(coll_x.shape[0])
        print(f"MC Glauber event: N_part={n_part}, N_coll={n_coll}")

    cell_area = dx * dx
    for step in range(steps):
        center_a_x = torch.tensor(-start_offset + beam_velocity * step * dt, device=device)
        center_b_x = torch.tensor(start_offset - beam_velocity * step * dt, device=device)
        center_a_y = torch.tensor(-0.5 * impact_parameter, device=device)
        center_b_y = torch.tensor(0.5 * impact_parameter, device=device)

        if initial_condition == "mc-glauber":
            assert nucleons_a is not None and nucleons_b is not None
            assert participants_a is not None and participants_b is not None and collision_centers is not None
            t_a = gaussian_point_thickness(
                xx, yy, nucleons_a[0], nucleons_a[1], center_a_x, center_a_y, nucleon_width, mass_number
            )
            t_b = gaussian_point_thickness(
                xx, yy, nucleons_b[0], nucleons_b[1], center_b_x, center_b_y, nucleon_width, mass_number
            )
            participant_a_thickness = gaussian_point_thickness(
                xx,
                yy,
                participants_a[0],
                participants_a[1],
                center_a_x,
                center_a_y,
                nucleon_width,
                max(participants_a[0].numel(), 1),
            )
            participant_b_thickness = gaussian_point_thickness(
                xx,
                yy,
                participants_b[0],
                participants_b[1],
                center_b_x,
                center_b_y,
                nucleon_width,
                max(participants_b[0].numel(), 1),
            )
            binary = gaussian_point_thickness(
                xx,
                yy,
                collision_centers[0],
                collision_centers[1],
                0.5 * (center_a_x + center_b_x),
                torch.tensor(0.0, device=device),
                nucleon_width,
                max(n_coll, 1),
            )
        else:
            # Nucleons per cell; sigma * T needs T per fm^2, hence / cell_area.
            t_a = woods_saxon_thickness(xx, yy, center_a_x, center_a_y, radius, skin_depth, mass_number)
            t_b = woods_saxon_thickness(xx, yy, center_b_x, center_b_y, radius, skin_depth, mass_number)
            collision_depth_a = inelastic_cross_section * t_a / cell_area
            collision_depth_b = inelastic_cross_section * t_b / cell_area
            participant_a_thickness = t_a * (1.0 - torch.exp(-collision_depth_b))
            participant_b_thickness = t_b * (1.0 - torch.exp(-collision_depth_a))
            binary = t_a * collision_depth_b
        overlap_time = torch.exp(-(center_a_x - center_b_x) ** 2 / max(collision_width * collision_width, 1.0e-6))
        wounded = participant_a_thickness + participant_b_thickness
        source = norm * ((1.0 - binary_fraction) * wounded + binary_fraction * binary) * overlap_time

        e = e + source * dt
        if step > int(steps * 0.20):
            e = e + diffusion * laplacian(e, dx) * dt
            e = e - radial_expansion * radial_flux_divergence(e, xx, yy, dx) * dt
            e = e * np.exp(-damping * dt)
            e = torch.clamp(e, min=0.0)

        if step % frame_every == 0:
            eps2, rms_radius = eccentricities(e, xx, yy)
            frames.append((
                e.detach().cpu().numpy().astype(np.float32),
                t_a.detach().cpu().numpy().astype(np.float32),
                t_b.detach().cpu().numpy().astype(np.float32),
            ))
            observables.append((float(step * dt), eps2, rms_radius))

    return frames, observables, {"n_part": n_part, "n_coll": n_coll}


def build_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    observables: list[tuple[float, float, float]],
    frame_duration_ms: int,
    title: str,
) -> go.Figure:
    first_e, first_a, first_b = frames[0]
    e_max = max(float(np.percentile(e, 99.8)) for e, _, _ in frames)
    thickness_max = max(float(np.percentile(a + b, 99.5)) for _, a, b in frames)
    e_max = max(e_max, 1.0e-6)
    thickness_max = max(thickness_max, 1.0e-6)
    times = np.array([x[0] for x in observables], dtype=np.float32)
    eps2 = np.array([x[1] for x in observables], dtype=np.float32)
    rms = np.array([x[2] for x in observables], dtype=np.float32)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("energy density evolution", "incoming nuclear thickness", "eps2 / rms radius"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "xy"}]],
        horizontal_spacing=0.055,
    )
    fig.add_trace(
        go.Heatmap(z=first_e, colorscale="Inferno", zmin=0.0, zmax=e_max, colorbar={"title": "e"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=first_a + first_b, colorscale="Viridis", zmin=0.0, zmax=thickness_max, colorbar={"title": "T"}),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(x=times, y=eps2, mode="lines", line={"color": "#60a5fa", "width": 3}, name="eps2"), row=1, col=3)
    fig.add_trace(go.Scatter(x=times, y=rms, mode="lines", line={"color": "#f97316", "width": 3}, name="rms"), row=1, col=3)
    fig.add_trace(
        go.Scatter(x=[times[0]], y=[eps2[0]], mode="markers", marker={"color": "#e0f2fe", "size": 10}, name="current eps2"),
        row=1,
        col=3,
    )

    fig.frames = [
        go.Frame(
            data=[
                go.Heatmap(z=e, colorscale="Inferno", zmin=0.0, zmax=e_max),
                go.Heatmap(z=a + b, colorscale="Viridis", zmin=0.0, zmax=thickness_max),
                go.Scatter(x=times, y=eps2, mode="lines", line={"color": "#60a5fa", "width": 3}),
                go.Scatter(x=times, y=rms, mode="lines", line={"color": "#f97316", "width": 3}),
                go.Scatter(x=[times[index]], y=[eps2[index]], mode="markers", marker={"color": "#e0f2fe", "size": 10}),
            ],
            name=str(index),
            layout=go.Layout(title_text=f"{title} | t={observables[index][0]:.2f}, eps2={observables[index][1]:.3f}"),
        )
        for index, (e, a, b) in enumerate(frames)
    ]
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(title_text="time", row=1, col=3)
    fig.update_yaxes(title_text="value", row=1, col=3)
    fig.update_layout(
        title=title,
        width=1360,
        height=540,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        plot_bgcolor="#07111f",
        paper_bgcolor="#07111f",
        font={"color": "#e5edf7"},
        updatemenus=[
            {
                "type": "buttons",
                "x": 0.5,
                "y": -0.08,
                "xanchor": "center",
                "direction": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
    )
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2D heavy-ion collision initial condition/evolution toy model.")
    parser.add_argument("--grid-size", type=int, default=180, help="2D transverse grid size.")
    parser.add_argument("--extent", type=float, default=12.0, help="Half-width of the transverse plane in fm-like units.")
    parser.add_argument("--steps", type=int, default=520, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=4, help="Save one viewer frame every N steps.")
    parser.add_argument("--dt", type=float, default=0.025, help="Simulation time step.")
    parser.add_argument("--mass-number", type=int, default=197, help="Nuclear mass number, e.g. Au/Pb around 197-208.")
    parser.add_argument("--radius", type=float, default=6.4, help="Nuclear radius.")
    parser.add_argument("--skin-depth", type=float, default=0.55, help="Woods-Saxon surface diffuseness.")
    parser.add_argument("--impact-parameter", type=float, default=7.0, help="Offset between nuclei in transverse direction.")
    parser.add_argument("--beam-velocity", type=float, default=2.4, help="Visual beam velocity across the grid.")
    parser.add_argument("--collision-width", type=float, default=1.8, help="Longitudinal overlap width controlling source duration.")
    parser.add_argument("--binary-fraction", type=float, default=0.16, help="Mix between wounded and binary collision scaling.")
    parser.add_argument(
        "--initial-condition",
        choices=("smooth", "mc-glauber"),
        default="smooth",
        help="Use continuous Woods-Saxon profiles or fluctuating sampled nucleons.",
    )
    parser.add_argument("--nucleon-width", type=float, default=0.45, help="Gaussian nucleon width for mc-glauber mode.")
    parser.add_argument(
        "--inelastic-cross-section",
        type=float,
        default=4.2,
        help="Nucleon-nucleon inelastic cross section in fm^2 (participant tagging and optical Glauber densities).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for event-by-event nucleon sampling.")
    parser.add_argument("--norm", type=float, default=2.5, help="Overall energy-density normalization.")
    parser.add_argument("--diffusion", type=float, default=0.18, help="Diffusion-like smoothing of energy density.")
    parser.add_argument("--radial-expansion", type=float, default=0.20, help="Radial expansion strength.")
    parser.add_argument("--damping", type=float, default=0.035, help="Energy-density decay rate.")
    parser.add_argument("--frame-duration-ms", type=int, default=35, help="Animation frame duration in milliseconds.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output", type=Path, default=Path("outputs/nuclear_collision_2d.html"), help="Output HTML path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames, observables, event_info = simulate(
        grid_size=args.grid_size,
        extent=args.extent,
        steps=args.steps,
        frame_every=args.frame_every,
        dt=args.dt,
        mass_number=args.mass_number,
        radius=args.radius,
        skin_depth=args.skin_depth,
        impact_parameter=args.impact_parameter,
        beam_velocity=args.beam_velocity,
        collision_width=args.collision_width,
        binary_fraction=args.binary_fraction,
        initial_condition=args.initial_condition,
        nucleon_width=args.nucleon_width,
        inelastic_cross_section=args.inelastic_cross_section,
        seed=args.seed,
        norm=args.norm,
        diffusion=args.diffusion,
        radial_expansion=args.radial_expansion,
        damping=args.damping,
        device=device,
    )
    event_suffix = ""
    if args.initial_condition == "mc-glauber":
        event_suffix = f", N_part={event_info['n_part']}, N_coll={event_info['n_coll']}"

    fig = build_figure(
        frames,
        observables,
        args.frame_duration_ms,
        title=(
            f"2D heavy-ion collision toy evolution | "
            f"{args.initial_condition}, b={args.impact_parameter:.1f}{event_suffix}"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved heavy-ion collision viewer: {args.output}")


if __name__ == "__main__":
    main()
