import argparse
import math
import random
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk


STAGES = (
    ("incoming nuclei", 0.00, 0.20, "#68d4e7"),
    ("pre-equilibrium", 0.20, 0.32, "#b9a7ff"),
    ("QGP hydrodynamics", 0.32, 0.56, "#f2533f"),
    ("mixed / freeze-out", 0.56, 0.76, "#f6c453"),
    ("hadron gas", 0.76, 1.01, "#9ee7c3"),
)


@dataclass
class Particle:
    angle: float
    radius: float
    speed: float
    size: float
    phase: float


@dataclass
class SimulationState:
    progress: float = 0.0
    impact_parameter: float = 5.7
    running: bool = True
    speed: float = 1.0
    particles: list[Particle] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(17)
        self.particles = [
            Particle(
                angle=index * 2.399963,
                radius=0.18 + rng.random() * 0.85,
                speed=0.32 + rng.random() * 0.52,
                size=1.0 + rng.random() * 2.0,
                phase=rng.random(),
            )
            for index in range(220)
        ]

    @property
    def stage(self) -> str:
        for name, start, end, _ in STAGES:
            if start <= self.progress < end:
                return name
        return STAGES[-1][0]

    @property
    def tau(self) -> float:
        return max(0.0, (self.progress - 0.20) * 14.0)

    @property
    def temperature(self) -> float:
        heat_up = max(0.0, min(1.0, (self.progress - 0.18) / 0.18))
        return max(0.03, 0.55 * heat_up * math.exp(-max(0.0, self.progress - 0.34) * 2.2))

    @property
    def radial_flow(self) -> float:
        return max(0.0, min(0.75, (self.progress - 0.30) * 1.25))

    @property
    def emitted_hadrons(self) -> int:
        hadron_fraction = max(0.0, min(1.0, (self.progress - 0.62) / 0.26))
        return round(hadron_fraction * len(self.particles))

    @property
    def eccentricity(self) -> float:
        return min(0.85, self.impact_parameter / 12.0 * 1.05)

    @property
    def overlap(self) -> float:
        return max(0.0, 1.0 - self.impact_parameter / 12.0)

    @property
    def multiplicity_proxy(self) -> int:
        return round(900 * self.overlap * self.overlap + 70)

    def step(self, dt: float) -> None:
        if self.running:
            self.progress = (self.progress + dt * self.speed) % 1.0


class HICSimulatorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Heavy-Ion Collision Educational Simulator")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.state = SimulationState()
        self._build_ui()
        self._tick()

    def _build_ui(self) -> None:
        self.root.configure(bg="#07100f")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#07100f")
        style.configure("Panel.TFrame", background="#101817", borderwidth=1, relief="solid")
        style.configure("TLabel", background="#101817", foreground="#eef7f4")
        style.configure("Muted.TLabel", background="#101817", foreground="#a6b8b3")
        style.configure("TButton", background="#15211f", foreground="#eef7f4")
        style.configure("Horizontal.TScale", background="#101817")

        outer = ttk.Frame(self.root, style="TFrame")
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(outer, bg="#020504", highlightthickness=1, highlightbackground="#2a3a37")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        panel = ttk.Frame(outer, style="Panel.TFrame", padding=14)
        panel.grid(row=0, column=1, sticky="ns")

        title = ttk.Label(panel, text="HIC Simulator", font=("Segoe UI", 22, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        subtitle = ttk.Label(panel, text="desktop educational mode", style="Muted.TLabel")
        subtitle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 14))

        self.stage_label = ttk.Label(panel, text="", font=("Segoe UI", 14, "bold"))
        self.stage_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        self.play_button = ttk.Button(panel, text="Pause", command=self._toggle_running)
        self.play_button.grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
        ttk.Button(panel, text="Reset", command=self._reset).grid(row=3, column=1, sticky="ew", pady=(0, 8))

        ttk.Label(panel, text="time", style="Muted.TLabel").grid(row=4, column=0, sticky="w")
        self.time_scale = ttk.Scale(panel, from_=0.0, to=1.0, command=self._set_progress)
        self.time_scale.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        ttk.Label(panel, text="impact parameter b", style="Muted.TLabel").grid(row=6, column=0, sticky="w")
        self.impact_scale = ttk.Scale(panel, from_=0.0, to=12.0, command=self._set_impact)
        self.impact_scale.set(self.state.impact_parameter)
        self.impact_scale.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        ttk.Label(panel, text="speed", style="Muted.TLabel").grid(row=8, column=0, sticky="w")
        self.speed_scale = ttk.Scale(panel, from_=0.2, to=3.0, command=self._set_speed)
        self.speed_scale.set(self.state.speed)
        self.speed_scale.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        self.readouts: dict[str, ttk.Label] = {}
        labels = [
            ("proper time", "tau"),
            ("temperature", "temperature"),
            ("radial flow", "radial_flow"),
            ("emitted hadrons", "emitted_hadrons"),
            ("overlap", "overlap"),
            ("eccentricity", "eccentricity"),
            ("multiplicity", "multiplicity"),
        ]
        row = 10
        for text, key in labels:
            ttk.Label(panel, text=text, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            value = ttk.Label(panel, text="0", font=("Segoe UI", 12, "bold"))
            value.grid(row=row, column=1, sticky="e", pady=4)
            self.readouts[key] = value
            row += 1

        self.stage_canvas = tk.Canvas(panel, width=310, height=110, bg="#101817", highlightthickness=1, highlightbackground="#2a3a37")
        self.stage_canvas.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(16, 0))

    def _toggle_running(self) -> None:
        self.state.running = not self.state.running
        self.play_button.configure(text="Pause" if self.state.running else "Play")

    def _reset(self) -> None:
        self.state.progress = 0.0
        self.state.running = True
        self.play_button.configure(text="Pause")
        self.time_scale.set(0.0)

    def _set_progress(self, value: str) -> None:
        self.state.progress = float(value)
        self.state.running = False
        self.play_button.configure(text="Play")

    def _set_impact(self, value: str) -> None:
        self.state.impact_parameter = float(value)

    def _set_speed(self, value: str) -> None:
        self.state.speed = float(value)

    def _tick(self) -> None:
        self.state.step(0.004)
        if self.state.running:
            self.time_scale.set(self.state.progress)
        self._draw()
        self.root.after(24, self._tick)

    def _draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self._draw_collision(width, height)
        self._draw_panel()
        self._draw_stage_strip()

    def _draw_collision(self, width: int, height: int) -> None:
        p = self.state.progress
        cx = width * 0.50
        cy = height * 0.52
        unit = min(width, height)
        nuclear_r = unit * 0.22
        approach = max(0.0, 1.0 - p / 0.28)
        pass_by = max(0.0, (p - 0.28) / 0.72)
        sep = unit * (0.50 * approach + 0.35 * pass_by)
        impact = unit * (0.04 + 0.018 * self.state.impact_parameter)
        opacity = max(0.08, 1.0 - p * 1.35)

        self._draw_lorentz_nucleus(cx - sep, cy - impact, nuclear_r, opacity)
        self._draw_lorentz_nucleus(cx + sep, cy + impact, nuclear_r, opacity)

        fire = max(0.0, min(1.0, (p - 0.18) / 0.24)) * max(0.0, 1.0 - max(0.0, p - 0.72) / 0.45)
        mixed = max(0.0, 1.0 - abs(p - 0.66) / 0.18)
        hadron = max(0.0, min(1.0, (p - 0.62) / 0.26))

        if fire > 0.01:
            r = unit * (0.10 + 0.28 * min(1.0, p * 1.35))
            self._draw_fireball(cx, cy, r, fire, mixed, hadron)

        if 0.34 < p < 0.84:
            for index in range(34):
                angle = index * 2.399963
                radius = unit * (0.08 + (index * 23 % 100) / 330)
                x = cx + math.cos(angle) * radius * (1.25 + p * 0.3)
                y = cy + math.sin(angle) * radius * 0.85
                length = unit * (0.025 + 0.030 * p)
                self.canvas.create_line(x, y, x + math.cos(angle) * length, y + math.sin(angle) * length, fill="#9fb1ad", width=1)

        if hadron > 0.02:
            for particle in self.state.particles:
                radius = unit * (0.05 + particle.radius * 0.34 * hadron)
                angle = particle.angle + particle.phase * 0.8
                x = cx + math.cos(angle) * radius * particle.speed * 1.35
                y = cy + math.sin(angle) * radius * particle.speed * 0.92
                color = self._rgba("#9ee7c3", 0.18 + 0.72 * hadron)
                self.canvas.create_oval(x - particle.size, y - particle.size, x + particle.size, y + particle.size, fill=color, outline="")

        self.canvas.create_line(width * 0.16, height * 0.78, width * 0.84, height * 0.28, fill="#eef7f4", dash=(9, 7), width=1)

    def _draw_lorentz_nucleus(self, x: float, y: float, r: float, opacity: float) -> None:
        for step in range(18, 0, -1):
            frac = step / 18
            color = self._rgba("#68d4e7", opacity * 0.055 * frac)
            self.canvas.create_oval(
                x - r * 0.34 * frac,
                y - r * frac,
                x + r * 0.34 * frac,
                y + r * frac,
                fill=color,
                outline="",
            )

    def _draw_fireball(self, x: float, y: float, r: float, fire: float, mixed: float, hadron: float) -> None:
        layers = [
            (1.35, "#9ee7c3", 0.10 + 0.28 * max(mixed, hadron)),
            (1.05, "#f6c453", 0.18 + 0.45 * max(fire, mixed)),
            (0.68, "#f2533f", 0.28 + 0.48 * fire),
        ]
        for scale, color, alpha in layers:
            self.canvas.create_oval(
                x - r * scale * 1.30,
                y - r * scale * 0.78,
                x + r * scale * 1.30,
                y + r * scale * 0.78,
                fill=self._rgba(color, alpha),
                outline="",
            )
        self.canvas.create_oval(x - r * 0.95, y - r * 0.52, x + r * 0.95, y + r * 0.52, outline="#fff6d6", width=2)
        self.canvas.create_oval(x - r * 1.26, y - r * 0.74, x + r * 1.26, y + r * 0.74, outline="#f6c453", width=2)

    def _draw_panel(self) -> None:
        self.stage_label.configure(text=self.state.stage)
        self.readouts["tau"].configure(text=f"{self.state.tau:.1f} fm/c")
        self.readouts["temperature"].configure(text=f"{self.state.temperature:.2f}")
        self.readouts["radial_flow"].configure(text=f"{self.state.radial_flow:.2f}")
        self.readouts["emitted_hadrons"].configure(text=str(self.state.emitted_hadrons))
        self.readouts["overlap"].configure(text=f"{self.state.overlap * 100:.0f}%")
        self.readouts["eccentricity"].configure(text=f"{self.state.eccentricity:.2f}")
        self.readouts["multiplicity"].configure(text=str(self.state.multiplicity_proxy))

    def _draw_stage_strip(self) -> None:
        self.stage_canvas.delete("all")
        width = max(self.stage_canvas.winfo_width(), 310)
        height = 110
        x = 10
        y = 18
        segment_width = (width - 20) / len(STAGES)
        for name, start, end, color in STAGES:
            active = start <= self.state.progress < end
            fill = color if active else "#15211f"
            outline = "#f6c453" if active else "#2a3a37"
            self.stage_canvas.create_rectangle(x, y, x + segment_width - 5, y + 36, fill=fill, outline=outline)
            self.stage_canvas.create_text(x + segment_width / 2 - 3, y + 58, text=name.split()[0], fill="#eef7f4", font=("Segoe UI", 8))
            x += segment_width

    @staticmethod
    def _rgba(hex_color: str, alpha: float) -> str:
        alpha = max(0.0, min(1.0, alpha))
        red = int(hex_color[1:3], 16)
        green = int(hex_color[3:5], 16)
        blue = int(hex_color[5:7], 16)
        red = round(red * alpha)
        green = round(green * alpha)
        blue = round(blue * alpha)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def run(self) -> None:
        self.root.mainloop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native desktop educational heavy-ion collision simulator.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a non-GUI logic smoke test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke_test:
        state = SimulationState()
        for _ in range(160):
            state.step(0.004)
        print(
            "smoke ok",
            f"stage={state.stage}",
            f"tau={state.tau:.2f}",
            f"temperature={state.temperature:.2f}",
            f"hadrons={state.emitted_hadrons}",
        )
        return
    HICSimulatorApp().run()


if __name__ == "__main__":
    main()
