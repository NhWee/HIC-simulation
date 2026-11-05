"""Generate the tabulated (2+1)-flavor lattice-QCD equation of state used by the hydro code.

The pressure is the HotQCD parametrization,

    p/T^4 = 1/2 [1 + tanh(c_t (t - t_0))]
            * (p_id + a_n/t + b_n/t^2 + c_n/t^3 + d_n/t^4) / (1 + a_d/t + b_d/t^2 + c_d/t^3 + d_d/t^4),

with t = T/T_c, T_c = 154 MeV and p_id = 95 pi^2/180 (the Stefan-Boltzmann value for 2+1 flavors),
from A. Bazavov et al. (HotQCD), Phys. Rev. D 90, 094503 (2014), arXiv:1407.6387. The
parametrization was fitted to lattice results for 130 MeV <= T <= 400 MeV and is constructed to
agree with the hadron resonance gas below that range; rows above 400 MeV use it as an extrapolation
toward the Stefan-Boltzmann limit.

NOTE: the coefficients below were transcribed without access to the paper in the session that added
this file. They reproduce the paper's quoted crossover energy density (0.18-0.5 GeV/fm^3 for
145-163 MeV), a trace-anomaly peak of about 4 near T = 200 MeV and a sound-speed minimum of about
0.14-0.15 near T_c (see tests/test_eos.py), but they should be checked against the paper's table once.

All other columns follow from thermodynamic identities, so the table is consistent by construction:

    s = dp/dT,   e = T s - p,   c_s^2 = dp/de = s / (T ds/dT).

Run ``python -m hic.lattice_eos`` to regenerate data/eos_table.csv.
"""
import argparse
import csv
import math
from pathlib import Path

HBARC = 0.1973269804  # GeV fm

HOTQCD_2014 = {
    "c_t": 3.8706,
    "a_n": -8.7704,
    "b_n": 3.9200,
    "c_n": 0.0,
    "d_n": 0.3419,
    "t_0": 0.9761,
    "a_d": -1.2600,
    "b_d": 0.8425,
    "c_d": 0.0,
    "d_d": -0.0475,
    "T_c": 0.154,  # GeV
}
P_IDEAL = 95.0 * math.pi**2 / 180.0


def pressure_over_t4(temperature: float, params: dict[str, float] = HOTQCD_2014) -> float:
    t = temperature / params["T_c"]
    numerator = P_IDEAL + params["a_n"] / t + params["b_n"] / t**2 + params["c_n"] / t**3 + params["d_n"] / t**4
    denominator = 1.0 + params["a_d"] / t + params["b_d"] / t**2 + params["c_d"] / t**3 + params["d_d"] / t**4
    return 0.5 * (1.0 + math.tanh(params["c_t"] * (t - params["t_0"]))) * numerator / denominator


def pressure(temperature: float) -> float:
    """Pressure in GeV/fm^3 at temperature in GeV."""
    return pressure_over_t4(temperature) * temperature**4 / HBARC**3


def thermodynamics(temperature: float, step: float = 1.0e-5) -> dict[str, float]:
    """Return p, e (GeV/fm^3), s (1/fm^3) and c_s^2 at a temperature in GeV."""
    p_minus2, p_minus, p_0, p_plus, p_plus2 = (
        pressure(temperature + k * step) for k in (-2, -1, 0, 1, 2)
    )
    # Fourth-order central differences.
    entropy = (p_minus2 - 8.0 * p_minus + 8.0 * p_plus - p_plus2) / (12.0 * step)
    d_entropy = (-p_minus2 + 16.0 * p_minus - 30.0 * p_0 + 16.0 * p_plus - p_plus2) / (12.0 * step * step)
    energy = temperature * entropy - p_0
    return {
        "temperature": temperature,
        "pressure": p_0,
        "energy_density": energy,
        "entropy_density": entropy,
        "sound_speed2": entropy / (temperature * d_entropy),
    }


def build_table(t_min: float = 0.070, t_max: float = 0.560, t_step: float = 0.001) -> list[dict[str, float]]:
    count = int(round((t_max - t_min) / t_step)) + 1
    return [thermodynamics(t_min + index * t_step) for index in range(count)]


def write_table(rows: list[dict[str, float]], path: Path) -> None:
    fields = ["energy_density", "pressure", "sound_speed2", "temperature", "entropy_density"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            writer.writerow([f"{row[field]:.8g}" for field in fields])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/eos_table.csv"))
    parser.add_argument("--t-min", type=float, default=0.070, help="GeV")
    parser.add_argument("--t-max", type=float, default=0.560, help="GeV")
    parser.add_argument("--t-step", type=float, default=0.001, help="GeV")
    args = parser.parse_args()
    rows = build_table(args.t_min, args.t_max, args.t_step)
    write_table(rows, args.output)
    print(
        f"Wrote {len(rows)} rows to {args.output}: "
        f"T = {rows[0]['temperature']:.3f}-{rows[-1]['temperature']:.3f} GeV, "
        f"e = {rows[0]['energy_density']:.4g}-{rows[-1]['energy_density']:.4g} GeV/fm^3"
    )


if __name__ == "__main__":
    main()
