# 2D heavy-ion collision initial-condition and evolution toy model using PyTorch GPU acceleration.
# Initial condition:
#   T_A(x,y,t) and T_B(x,y,t) are moving Woods-Saxon-like thickness profiles.
#   e0(x,y,t) = norm * [(1-alpha) * wounded + alpha * binary]
# Evolution:
#   de/dt = diffusion * laplacian(e) - radial_expansion * div(e * r_hat) - damping * e
# This is a qualitative GPU prototype, not a validated relativistic hydro code.
# You can handle the parameters:
# grid_size, steps, frame_every, mass_number, radius, skin_depth,
# impact_parameter, beam_velocity, collision_width, binary_fraction,
# diffusion, radial_expansion, damping, output
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


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
    norm: float,
    diffusion: float,
    radial_expansion: float,
    damping: float,
    device: torch.device,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[tuple[float, float, float]]]:
    xx, yy, dx = make_grid(grid_size, extent, device)
    e = torch.zeros((grid_size, grid_size), device=device)
    frames = []
    observables = []
    start_offset = extent * 0.75

    for step in range(steps):
        t = (step - steps * 0.22) * dt
        center_a_x = torch.tensor(-start_offset + beam_velocity * step * dt, device=device)
        center_b_x = torch.tensor(start_offset - beam_velocity * step * dt, device=device)
        center_a_y = torch.tensor(-0.5 * impact_parameter, device=device)
        center_b_y = torch.tensor(0.5 * impact_parameter, device=device)

        t_a = woods_saxon_thickness(xx, yy, center_a_x, center_a_y, radius, skin_depth, mass_number)
        t_b = woods_saxon_thickness(xx, yy, center_b_x, center_b_y, radius, skin_depth, mass_number)
        overlap_time = torch.exp(-(center_a_x - center_b_x) ** 2 / max(collision_width * collision_width, 1.0e-6))
        wounded = t_a * (1.0 - torch.exp(-t_b)) + t_b * (1.0 - torch.exp(-t_a))
        binary = t_a * t_b
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

    return frames, observables


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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--norm", type=float, default=2.5, help="Overall energy-density normalization.")
    parser.add_argument("--diffusion", type=float, default=0.18, help="Diffusion-like smoothing of energy density.")
    parser.add_argument("--radial-expansion", type=float, default=0.20, help="Radial expansion strength.")
    parser.add_argument("--damping", type=float, default=0.035, help="Energy-density decay rate.")
    parser.add_argument("--frame-duration-ms", type=int, default=35, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/nuclear_collision_2d.html"), help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames, observables = simulate(
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
        norm=args.norm,
        diffusion=args.diffusion,
        radial_expansion=args.radial_expansion,
        damping=args.damping,
        device=device,
    )
    fig = build_figure(
        frames,
        observables,
        args.frame_duration_ms,
        title=f"2D heavy-ion collision toy evolution | b={args.impact_parameter:.1f}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved heavy-ion collision viewer: {args.output}")


if __name__ == "__main__":
    main()
