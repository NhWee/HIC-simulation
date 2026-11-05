import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch

_TABLE_CACHE: dict[tuple, torch.Tensor] = {}


@dataclass(frozen=True)
class EquationOfState:
    kind: str
    speed_of_sound2: float
    low_speed_of_sound2: float
    high_speed_of_sound2: float
    transition_energy: float
    transition_width: float
    table_energy: tuple[float, ...] = ()
    table_pressure: tuple[float, ...] = ()
    table_sound_speed2: tuple[float, ...] = ()
    table_temperature: tuple[float, ...] = ()

    def _table_tensor(self, values: tuple[float, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Cache converted tables, keyed by their contents, dtype and device, instead of rebuilding a
        # tensor on every call (this dominated the cost of the table-EoS primitive recovery).
        key = (values, dtype, str(device))
        tensor = _TABLE_CACHE.get(key)
        if tensor is None:
            tensor = torch.as_tensor(values, dtype=dtype, device=device)
            _TABLE_CACHE[key] = tensor
        return tensor

    def _interpolate_table(self, energy_density: torch.Tensor, values: tuple[float, ...]) -> torch.Tensor:
        table_energy = self._table_tensor(self.table_energy, energy_density.dtype, energy_density.device)
        table_values = self._table_tensor(values, energy_density.dtype, energy_density.device)
        clipped = torch.clamp(energy_density, min=float(table_energy[0]), max=float(table_energy[-1]))
        upper = torch.bucketize(clipped.reshape(-1), table_energy)
        upper = torch.clamp(upper, min=1, max=table_energy.numel() - 1)
        lower = upper - 1
        e0 = table_energy[lower]
        e1 = table_energy[upper]
        v0 = table_values[lower]
        v1 = table_values[upper]
        fraction = (clipped.reshape(-1) - e0) / torch.clamp(e1 - e0, min=1.0e-12)
        interpolated = v0 + fraction * (v1 - v0)
        return interpolated.reshape_as(energy_density)

    def pressure(self, energy_density: torch.Tensor) -> torch.Tensor:
        if self.kind == "conformal":
            return self.speed_of_sound2 * energy_density
        if self.kind == "table":
            # Inside the table: linear interpolation. Below it, p = (p_0/e_0) e so that the pressure of
            # dilute/vacuum cells goes to zero instead of staying at the first tabulated value (which
            # would give p > e). Above it, continue with the last tabulated sound speed.
            interpolated = self._interpolate_table(energy_density, self.table_pressure)
            e_low, p_low = self.table_energy[0], self.table_pressure[0]
            e_high, p_high = self.table_energy[-1], self.table_pressure[-1]
            below = energy_density * (p_low / e_low)
            above = p_high + self.table_sound_speed2[-1] * (energy_density - e_high)
            result = torch.where(energy_density < e_low, below, interpolated)
            return torch.where(energy_density > e_high, above, result)

        low_pressure = self.low_speed_of_sound2 * energy_density
        high_pressure = self.high_speed_of_sound2 * energy_density
        switch = torch.sigmoid((energy_density - self.transition_energy) / max(self.transition_width, 1.0e-8))
        return (1.0 - switch) * low_pressure + switch * high_pressure

    def sound_speed2(self, energy_density: torch.Tensor) -> torch.Tensor:
        if self.kind == "conformal":
            return torch.full_like(energy_density, self.speed_of_sound2)
        if self.kind == "table":
            interpolated = self._interpolate_table(energy_density, self.table_sound_speed2)
            dilute = torch.full_like(energy_density, self.table_pressure[0] / self.table_energy[0])
            return torch.where(energy_density < self.table_energy[0], dilute, interpolated)

        switch = torch.sigmoid((energy_density - self.transition_energy) / max(self.transition_width, 1.0e-8))
        return (1.0 - switch) * self.low_speed_of_sound2 + switch * self.high_speed_of_sound2

    def temperature(self, energy_density: torch.Tensor, fallback_temperature: float = 0.145) -> torch.Tensor:
        if self.kind == "table" and self.table_temperature:
            return self._interpolate_table(energy_density, self.table_temperature)
        return torch.full_like(energy_density, fallback_temperature)

    def max_sound_speed(self) -> float:
        if self.kind == "conformal":
            return float(self.speed_of_sound2**0.5)
        if self.kind == "table":
            return float(max(self.table_sound_speed2) ** 0.5)
        return float(max(self.low_speed_of_sound2, self.high_speed_of_sound2) ** 0.5)

    def bjorken_cooling_factor(self, tau: float, dt: float) -> float:
        if self.kind != "conformal":
            raise ValueError("Analytic Bjorken cooling is only available for conformal EoS.")
        return (tau / (tau + dt)) ** (1.0 + self.speed_of_sound2)


def add_eos_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eos",
        choices=("conformal", "crossover", "table"),
        default="conformal",
        help="Equation of state model.",
    )
    parser.add_argument("--speed-of-sound2", type=float, default=1.0 / 3.0, help="Conformal EoS p = cs2 * e.")
    parser.add_argument("--low-speed-of-sound2", type=float, default=0.15, help="Low-density sound speed squared.")
    parser.add_argument("--high-speed-of-sound2", type=float, default=1.0 / 3.0, help="High-density sound speed squared.")
    parser.add_argument("--transition-energy", type=float, default=1.0, help="Crossover midpoint energy density.")
    parser.add_argument("--transition-width", type=float, default=0.25, help="Crossover smoothness in energy density.")
    parser.add_argument(
        "--eos-table",
        type=Path,
        default=Path("data/eos_table.csv"),
        help="CSV table with energy_density, pressure, sound_speed2 (and temperature); default is the HotQCD lattice EoS.",
    )


def read_eos_table(path: Path) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"EoS table not found: {path}")

    rows = []
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"energy_density", "pressure", "sound_speed2"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("EoS table must contain energy_density, pressure, and sound_speed2 columns.")
        has_temperature = "temperature" in set(reader.fieldnames)
        for row in reader:
            rows.append(
                (
                    float(row["energy_density"]),
                    float(row["pressure"]),
                    float(row["sound_speed2"]),
                    float(row["temperature"]) if has_temperature else 0.145,
                )
            )

    if len(rows) < 2:
        raise ValueError("EoS table must contain at least two rows.")
    rows.sort(key=lambda item: item[0])
    energy = tuple(item[0] for item in rows)
    if any(next_energy <= current_energy for current_energy, next_energy in zip(energy, energy[1:])):
        raise ValueError("EoS table energy_density values must be strictly increasing.")
    pressure = tuple(item[1] for item in rows)
    sound_speed2 = tuple(item[2] for item in rows)
    temperature = tuple(item[3] for item in rows)
    return energy, pressure, sound_speed2, temperature


def build_eos(args: argparse.Namespace) -> EquationOfState:
    table_energy: tuple[float, ...] = ()
    table_pressure: tuple[float, ...] = ()
    table_sound_speed2: tuple[float, ...] = ()
    table_temperature: tuple[float, ...] = ()
    if args.eos == "table":
        table_energy, table_pressure, table_sound_speed2, table_temperature = read_eos_table(args.eos_table)

    return EquationOfState(
        kind=args.eos,
        speed_of_sound2=args.speed_of_sound2,
        low_speed_of_sound2=args.low_speed_of_sound2,
        high_speed_of_sound2=args.high_speed_of_sound2,
        transition_energy=args.transition_energy,
        transition_width=args.transition_width,
        table_energy=table_energy,
        table_pressure=table_pressure,
        table_sound_speed2=table_sound_speed2,
        table_temperature=table_temperature,
    )
