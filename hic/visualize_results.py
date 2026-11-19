import argparse
import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = 0.0
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def column(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows if name in row], dtype=np.float64)


def build_report(args: argparse.Namespace) -> go.Figure:
    hydro = read_csv(args.hydro)
    freezeout = read_csv(args.freezeout)
    particles = read_csv(args.particles)
    pipeline = read_csv(args.pipeline_summary)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Hydro energy and flow",
            "Hydro geometry",
            "Freeze-out surface",
            "Particle pT spectrum",
            "Particle azimuth",
            "Event observables",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "scatter"}],
            [{"type": "xy"}, {"type": "polar"}, {"type": "xy"}],
        ],
        horizontal_spacing=0.075,
        vertical_spacing=0.13,
    )

    if hydro:
        time = column(hydro, "time")
        fig.add_trace(
            go.Scatter(x=time, y=column(hydro, "max_energy_density"), mode="lines+markers", name="max e"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=column(hydro, "max_velocity"), mode="lines+markers", name="max |v|"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=column(hydro, "eps2"), mode="lines+markers", name="eps2"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=time, y=column(hydro, "rms_radius"), mode="lines+markers", name="rms radius"),
            row=1,
            col=2,
        )

    if freezeout:
        color = column(freezeout, "temperature") if "temperature" in freezeout[0] else column(freezeout, "tau")
        fig.add_trace(
            go.Scatter(
                x=column(freezeout, "x"),
                y=column(freezeout, "y"),
                mode="markers",
                marker={"size": 5, "color": color, "colorscale": "Viridis", "colorbar": {"title": "T/tau"}},
                name="freeze-out",
            ),
            row=1,
            col=3,
        )

    if particles:
        pt = column(particles, "pt")
        phi = column(particles, "phi")
        fig.add_trace(
            go.Histogram(x=pt, nbinsx=50, name="pT", marker={"color": "#38bdf8"}),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Barpolar(
                theta=np.degrees(phi),
                r=np.ones_like(phi),
                marker={"color": "#f97316", "opacity": 0.45},
                name="phi",
            ),
            row=2,
            col=2,
        )

    if pipeline:
        event = column(pipeline, "event")
        fig.add_trace(
            go.Scatter(x=event, y=column(pipeline, "mean_pt"), mode="lines+markers", name="mean pT"),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(x=event, y=column(pipeline, "v2"), mode="lines+markers", name="v2"),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(x=event, y=column(pipeline, "v3"), mode="lines+markers", name="v3"),
            row=2,
            col=3,
        )

    fig.update_xaxes(title_text="tau", row=1, col=1)
    fig.update_xaxes(title_text="tau", row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=3)
    fig.update_yaxes(title_text="y", scaleanchor="x3", scaleratio=1, row=1, col=3)
    fig.update_xaxes(title_text="pT", row=2, col=1)
    fig.update_xaxes(title_text="event", row=2, col=3)
    fig.update_layout(
        title="HIC Simulation Result Report",
        width=1450,
        height=850,
        template="plotly_dark",
        bargap=0.05,
        legend={"orientation": "h", "y": -0.08},
    )
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive HTML report from HIC simulation outputs.")
    parser.add_argument("--hydro", type=Path, default=Path("outputs/ideal_hydro_2d.csv"), help="Hydro diagnostics CSV.")
    parser.add_argument("--freezeout", type=Path, default=Path("outputs/freezeout_surface.csv"), help="Freeze-out surface CSV.")
    parser.add_argument("--particles", type=Path, default=Path("outputs/particles.csv"), help="Sampled particles CSV.")
    parser.add_argument("--pipeline-summary", type=Path, default=Path("outputs/pipeline_summary.csv"), help="Event pipeline summary CSV.")
    parser.add_argument("--output", type=Path, default=Path("outputs/simulation_report.html"), help="Output HTML report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fig = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved simulation report: {args.output}")


if __name__ == "__main__":
    main()
