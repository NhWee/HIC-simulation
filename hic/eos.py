"""Equations of state for the transverse hydrodynamics prototypes.

Units: energy density and pressure in GeV/fm^3, temperature in GeV.

Tabulated EoS (``kind="table"``)
--------------------------------
The pressure is interpolated linearly between the tabulated (e, p) nodes, so
the squared sound speed used by the solver is the piecewise-constant slope
dp/de of that interpolation. Below the first node the pressure goes linearly to
zero at e = 0; above the last node it continues with the last slope. The
equation of state therefore never produces a vacuum pressure and never
becomes infinitely soft outside the table.

At zero baryon chemical potential thermodynamic consistency requires

    e + p = T s,        d ln s = de / (e + p),        cs2 = dp/de.

For a linear segment p = p_i + k (e - e_i) these integrate exactly to

    T(e) = T_i * ((e + p(e)) / (e_i + p_i)) ** (k / (1 + k)).

``read_eos_table`` warns when a table's temperature or sound-speed column
violates these relations, and ``python -m hic.eos`` prints a consistency
report or writes a consistent table.
"""

import argparse
import csv
import functools
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch


CONSISTENCY_TOLERANCE = 1.0e-3


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

    def pressure(self, energy_density: torch.Tensor) -> torch.Tensor:
        if self.kind == "conformal":
            return self.speed_of_sound2 * energy_density
        if self.kind == "table":
            arrays = _table_arrays(self, energy_density.dtype, energy_density.device)
            segment = _table_segment(arrays, energy_density)
            return arrays.node_pressure[segment] + arrays.slope[segment] * (
                energy_density - arrays.node_energy[segment]
            )

        switch = self._crossover_switch(energy_density)
        return ((1.0 - switch) * self.low_speed_of_sound2 + switch * self.high_speed_of_sound2) * energy_density

    def sound_speed2(self, energy_density: torch.Tensor) -> torch.Tensor:
        """Return dp/de, the squared sound speed implied by ``pressure``."""
        if self.kind == "conformal":
            return torch.full_like(energy_density, self.speed_of_sound2)
        if self.kind == "table":
            arrays = _table_arrays(self, energy_density.dtype, energy_density.device)
            return arrays.slope[_table_segment(arrays, energy_density)]

        # p = c2_eff(e) * e with a sigmoid c2_eff, so dp/de also contains
        # e * dc2_eff/de. It peaks near the transition energy.
        switch = self._crossover_switch(energy_density)
        width = max(self.transition_width, 1.0e-8)
        delta = self.high_speed_of_sound2 - self.low_speed_of_sound2
        return (
            (1.0 - switch) * self.low_speed_of_sound2
            + switch * self.high_speed_of_sound2
            + energy_density * delta * switch * (1.0 - switch) / width
        )

    def temperature(self, energy_density: torch.Tensor, fallback_temperature: float = 0.145) -> torch.Tensor:
        if self.kind != "table" or not self.table_temperature:
            return torch.full_like(energy_density, fallback_temperature)

        arrays = _table_arrays(self, energy_density.dtype, energy_density.device)
        safe_energy = torch.clamp(energy_density, min=0.0)
        segment = _table_segment(arrays, safe_energy)
        pressure = arrays.node_pressure[segment] + arrays.slope[segment] * (safe_energy - arrays.node_energy[segment])
        enthalpy = torch.clamp(safe_energy + pressure, min=0.0)
        inside = arrays.reference_temperature[segment] * torch.pow(
            enthalpy / arrays.reference_enthalpy[segment],
            arrays.temperature_exponent[segment],
        )
        above = arrays.last_temperature * torch.pow(enthalpy / arrays.last_enthalpy, arrays.extrapolation_exponent)
        return torch.where(safe_energy > arrays.last_energy, above, inside)

    def max_sound_speed(self) -> float:
        """Upper bound of sqrt(dp/de); used as the Rusanov signal speed."""
        if self.kind == "conformal":
            return float(self.speed_of_sound2**0.5)
        return float(_max_sound_speed2(self) ** 0.5)

    def bjorken_cooling_factor(self, tau: float, dt: float) -> float:
        if self.kind != "conformal":
            raise ValueError("Analytic Bjorken cooling is only available for conformal EoS.")
        return (tau / (tau + dt)) ** (1.0 + self.speed_of_sound2)

    def _crossover_switch(self, energy_density: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((energy_density - self.transition_energy) / max(self.transition_width, 1.0e-8))


class _TableArrays(NamedTuple):
    boundaries: torch.Tensor
    node_energy: torch.Tensor
    node_pressure: torch.Tensor
    slope: torch.Tensor
    reference_enthalpy: torch.Tensor
    reference_temperature: torch.Tensor
    temperature_exponent: torch.Tensor
    last_energy: float
    last_enthalpy: float
    last_temperature: float
    extrapolation_exponent: float


def table_segment_slopes(energy: tuple[float, ...], pressure: tuple[float, ...]) -> tuple[float, ...]:
    """Slopes dp/de of the origin segment, the interior segments, in order."""
    nodes_e = (0.0, *energy)
    nodes_p = (0.0, *pressure)
    return tuple(
        (nodes_p[index + 1] - nodes_p[index]) / (nodes_e[index + 1] - nodes_e[index])
        for index in range(len(nodes_e) - 1)
    )


def _segment_temperature_exponents(
    energy: tuple[float, ...],
    pressure: tuple[float, ...],
    temperature: tuple[float, ...],
) -> tuple[float, ...]:
    slopes = table_segment_slopes(energy, pressure)
    exponents = [slopes[0] / (1.0 + slopes[0])]
    for index in range(len(energy) - 1):
        enthalpy_ratio = (energy[index + 1] + pressure[index + 1]) / (energy[index] + pressure[index])
        exponents.append(math.log(temperature[index + 1] / temperature[index]) / math.log(enthalpy_ratio))
    return tuple(exponents)


@functools.lru_cache(maxsize=64)
def _table_arrays(eos: EquationOfState, dtype: torch.dtype, device: torch.device) -> _TableArrays:
    energy = eos.table_energy
    pressure = eos.table_pressure
    slopes = table_segment_slopes(energy, pressure)
    node_energy = (0.0, *energy[:-1])
    node_pressure = (0.0, *pressure[:-1])
    if eos.table_temperature:
        temperature = eos.table_temperature
        exponents = _segment_temperature_exponents(energy, pressure, temperature)
        # Segment 0 runs from the origin to the first node, so it is anchored
        # at the first node. Interior segments are anchored at their left node.
        reference_temperature = (temperature[0], *temperature[:-1])
        reference_enthalpy = (energy[0] + pressure[0], *(e + p for e, p in zip(energy[:-1], pressure[:-1])))
        last_temperature = temperature[-1]
    else:
        exponents = (0.0,) * len(slopes)
        reference_temperature = (1.0,) * len(slopes)
        reference_enthalpy = (1.0,) * len(slopes)
        last_temperature = 1.0

    def tensor(values: tuple[float, ...]) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    return _TableArrays(
        boundaries=tensor(energy[:-1]),
        node_energy=tensor(node_energy),
        node_pressure=tensor(node_pressure),
        slope=tensor(slopes),
        reference_enthalpy=tensor(reference_enthalpy),
        reference_temperature=tensor(reference_temperature),
        temperature_exponent=tensor(exponents),
        last_energy=float(energy[-1]),
        last_enthalpy=float(energy[-1] + pressure[-1]),
        last_temperature=float(last_temperature),
        extrapolation_exponent=float(slopes[-1] / (1.0 + slopes[-1])),
    )


def _table_segment(arrays: _TableArrays, energy_density: torch.Tensor) -> torch.Tensor:
    flat = energy_density.reshape(-1).contiguous()
    return torch.bucketize(flat, arrays.boundaries, right=True).reshape(energy_density.shape)


@functools.lru_cache(maxsize=64)
def _max_sound_speed2(eos: EquationOfState) -> float:
    if eos.kind == "table":
        return max(table_segment_slopes(eos.table_energy, eos.table_pressure))
    width = max(eos.transition_width, 1.0e-8)
    upper = max(eos.transition_energy, 0.0) + 60.0 * width
    probe = torch.linspace(0.0, upper, 200_001, dtype=torch.float64)
    sampled = float(torch.max(eos.sound_speed2(probe)))
    return max(sampled, eos.low_speed_of_sound2, eos.high_speed_of_sound2)


@dataclass(frozen=True)
class ConsistencyReport:
    """Deviations of a table from mu_B = 0 thermodynamics.

    ``temperature_log_errors[i]`` compares ln(T_{i+1}/T_i) with the exact
    value k/(1+k) ln(w_{i+1}/w_i) for segment i, w = e + p.
    ``sound_speed2_errors[i]`` is the distance of the tabulated cs2 at node i
    from the interval spanned by the adjacent slopes dp/de.
    """

    slopes: tuple[float, ...]
    temperature_log_errors: tuple[float, ...]
    sound_speed2_errors: tuple[float, ...]

    @property
    def max_temperature_log_error(self) -> float:
        return max((abs(value) for value in self.temperature_log_errors), default=0.0)

    @property
    def max_sound_speed2_error(self) -> float:
        return max(self.sound_speed2_errors, default=0.0)

    def is_consistent(self, tolerance: float = CONSISTENCY_TOLERANCE) -> bool:
        return self.max_temperature_log_error <= tolerance and self.max_sound_speed2_error <= tolerance


def thermodynamic_consistency(
    energy: tuple[float, ...],
    pressure: tuple[float, ...],
    sound_speed2: tuple[float, ...] = (),
    temperature: tuple[float, ...] = (),
) -> ConsistencyReport:
    slopes = table_segment_slopes(energy, pressure)
    temperature_errors: list[float] = []
    if temperature:
        for index in range(len(energy) - 1):
            k = slopes[index + 1]
            enthalpy_ratio = (energy[index + 1] + pressure[index + 1]) / (energy[index] + pressure[index])
            expected = k / (1.0 + k) * math.log(enthalpy_ratio)
            temperature_errors.append(math.log(temperature[index + 1] / temperature[index]) - expected)
    sound_speed_errors: list[float] = []
    if sound_speed2:
        for index, value in enumerate(sound_speed2):
            left = slopes[index]
            right = slopes[index + 1] if index + 1 < len(slopes) else slopes[-1]
            lower, upper = min(left, right), max(left, right)
            sound_speed_errors.append(max(lower - value, value - upper, 0.0))
    return ConsistencyReport(slopes, tuple(temperature_errors), tuple(sound_speed_errors))


def consistent_temperatures(
    energy: tuple[float, ...],
    pressure: tuple[float, ...],
    anchor_index: int,
    anchor_temperature: float,
) -> tuple[float, ...]:
    """Temperatures implied by p(e), fixed to ``anchor_temperature`` at one node."""
    slopes = table_segment_slopes(energy, pressure)
    enthalpy = [e + p for e, p in zip(energy, pressure)]
    temperature = [0.0] * len(energy)
    temperature[anchor_index] = anchor_temperature
    for index in range(anchor_index, len(energy) - 1):
        k = slopes[index + 1]
        temperature[index + 1] = temperature[index] * (enthalpy[index + 1] / enthalpy[index]) ** (k / (1.0 + k))
    for index in range(anchor_index, 0, -1):
        k = slopes[index]
        temperature[index - 1] = temperature[index] * (enthalpy[index - 1] / enthalpy[index]) ** (k / (1.0 + k))
    return tuple(temperature)


def centered_sound_speed2(energy: tuple[float, ...], pressure: tuple[float, ...]) -> tuple[float, ...]:
    """Node values of cs2: the mean of the slopes dp/de on both sides of each node."""
    slopes = table_segment_slopes(energy, pressure)
    return tuple(
        0.5 * (slopes[index] + (slopes[index + 1] if index + 1 < len(slopes) else slopes[-1]))
        for index in range(len(energy))
    )


def add_eos_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eos",
        choices=("conformal", "crossover", "table"),
        default="conformal",
        help="Equation of state model.",
    )
    parser.add_argument("--speed-of-sound2", type=float, default=1.0 / 3.0, help="Conformal EoS p = cs2 * e.")
    parser.add_argument("--low-speed-of-sound2", type=float, default=0.15, help="Low-density limit of cs2 and p/e (crossover EoS).")
    parser.add_argument("--high-speed-of-sound2", type=float, default=1.0 / 3.0, help="High-density limit of cs2 and p/e (crossover EoS).")
    parser.add_argument("--transition-energy", type=float, default=1.0, help="Crossover midpoint energy density.")
    parser.add_argument("--transition-width", type=float, default=0.25, help="Crossover smoothness in energy density.")
    parser.add_argument(
        "--eos-table",
        type=Path,
        default=Path("data/eos_table.csv"),
        help="CSV table with energy_density and pressure columns; temperature and sound_speed2 are optional.",
    )


def read_eos_table(path: Path) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Read and validate an EoS table.

    Returns (energy, pressure, sound_speed2, temperature) sorted by energy.
    A missing sound_speed2 column is filled with the slopes of p(e). A missing
    temperature column yields an empty tuple.
    """
    if not path.exists():
        raise FileNotFoundError(f"EoS table not found: {path}")

    rows = []
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"energy_density", "pressure"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("EoS table must contain energy_density and pressure columns.")
        has_sound_speed = "sound_speed2" in set(reader.fieldnames)
        has_temperature = "temperature" in set(reader.fieldnames)
        for row in reader:
            rows.append(
                (
                    float(row["energy_density"]),
                    float(row["pressure"]),
                    float(row["sound_speed2"]) if has_sound_speed else math.nan,
                    float(row["temperature"]) if has_temperature else math.nan,
                )
            )

    if len(rows) < 2:
        raise ValueError("EoS table must contain at least two rows.")
    rows.sort(key=lambda item: item[0])
    energy = tuple(item[0] for item in rows)
    if energy[0] <= 0.0:
        raise ValueError("EoS table energy_density values must be positive.")
    if any(next_energy <= current_energy for current_energy, next_energy in zip(energy, energy[1:])):
        raise ValueError("EoS table energy_density values must be strictly increasing.")
    pressure = tuple(item[1] for item in rows)
    if pressure[0] < 0.0:
        raise ValueError("EoS table pressure values must be non-negative.")
    slopes = table_segment_slopes(energy, pressure)
    if any(slope < 0.0 or slope > 1.0 for slope in slopes):
        raise ValueError(
            "EoS table must have 0 <= dp/de <= 1 between nodes (mechanical stability and causality); "
            f"found slopes {', '.join(f'{slope:.4g}' for slope in slopes)}."
        )
    sound_speed2 = (
        tuple(item[2] for item in rows) if has_sound_speed else centered_sound_speed2(energy, pressure)
    )
    temperature: tuple[float, ...] = ()
    if has_temperature:
        temperature = tuple(item[3] for item in rows)
        if any(value <= 0.0 for value in temperature):
            raise ValueError("EoS table temperature values must be positive.")

    report = thermodynamic_consistency(energy, pressure, sound_speed2 if has_sound_speed else (), temperature)
    if not report.is_consistent():
        warnings.warn(
            f"EoS table {path} is not thermodynamically consistent: max |d ln T| error "
            f"{report.max_temperature_log_error:.3g}, max cs2 error {report.max_sound_speed2_error:.3g}. "
            "Pressure is used as given; temperature interpolation follows the tabulated nodes. "
            f"Run 'python -m hic.eos --eos-table {path}' for a report.",
            stacklevel=2,
        )
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check or rewrite a tabulated EoS for thermodynamic consistency.")
    parser.add_argument("--eos-table", type=Path, default=Path("data/eos_table.csv"), help="EoS table to check.")
    parser.add_argument(
        "--write-consistent",
        type=Path,
        default=None,
        help="Write a copy whose temperature and sound_speed2 columns follow from p(e).",
    )
    parser.add_argument(
        "--anchor-energy",
        type=float,
        default=0.35,
        help="Node energy density whose tabulated temperature is kept when writing a consistent table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy, pressure, sound_speed2, temperature = read_eos_table(args.eos_table)
    report = thermodynamic_consistency(energy, pressure, sound_speed2, temperature)
    anchor_index = min(range(len(energy)), key=lambda index: abs(energy[index] - args.anchor_energy))
    consistent = consistent_temperatures(energy, pressure, anchor_index, temperature[anchor_index]) if temperature else ()

    print(f"EoS table: {args.eos_table}")
    print(f"{'e [GeV/fm^3]':>13} {'p':>9} {'cs2 col':>8} {'dp/de right':>11} {'T col':>8} {'T from p(e)':>11}")
    for index, (e, p) in enumerate(zip(energy, pressure)):
        right_slope = report.slopes[index + 1] if index + 1 < len(report.slopes) else report.slopes[-1]
        t_col = f"{temperature[index]:8.4f}" if temperature else f"{'-':>8}"
        t_consistent = f"{consistent[index]:11.4f}" if consistent else f"{'-':>11}"
        print(f"{e:13.4f} {p:9.4f} {sound_speed2[index]:8.4f} {right_slope:11.4f} {t_col} {t_consistent}")
    print(f"max dp/de = {max(report.slopes):.4f} (Rusanov signal speed {max(report.slopes) ** 0.5:.4f})")
    print(f"max |d ln T| error = {report.max_temperature_log_error:.3g}")
    print(f"max cs2 column error = {report.max_sound_speed2_error:.3g}")
    print("consistent" if report.is_consistent() else "NOT consistent")

    if args.write_consistent is not None:
        if not temperature:
            raise SystemExit("A temperature column is required to anchor a consistent table.")
        args.write_consistent.parent.mkdir(parents=True, exist_ok=True)
        node_sound_speed2 = centered_sound_speed2(energy, pressure)
        with args.write_consistent.open("w", newline="") as output_file:
            writer = csv.writer(output_file, lineterminator="\n")
            writer.writerow(["energy_density", "pressure", "sound_speed2", "temperature"])
            for row in zip(energy, pressure, node_sound_speed2, consistent):
                writer.writerow([f"{row[0]:.6g}", f"{row[1]:.6g}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
        print(f"Saved consistent EoS table: {args.write_consistent} (anchored at e={energy[anchor_index]:g})")


if __name__ == "__main__":
    main()
