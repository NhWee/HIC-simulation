"""Reproduce the Song & Heinz Cu+Cu viscous-hydro benchmark (PRC 77, 064901 (2008), arXiv:0712.3715).

Setup as in their Sec. II C-D: optical-Glauber wounded-nucleon profile for Cu+Cu (R = 4.2 fm, xi = 0.596 fm,
rho0 = 0.17 fm^-3, sigma = 40 mb), b = 7 fm, zero initial flow, e(0,0; b=0) = 30 GeV/fm^3 at tau0 = 0.6 fm/c,
EOS I (massless QGP, Nf = 2.5), eta/s = 0.08, tau_pi = 3 eta/(sT), Navier-Stokes initial shear.

Records <|v_x| - |v_y|> (gamma e weighted), eps_p = <T0^xx - T0^yy>/<T0^xx + T0^yy>, the total eps'_p (with pi)
and eps_x versus tau - tau0, for the ideal fluid and for several forms of the shear relaxation equation:
  * "song-heinz": D pi = -(pi - 2 eta sigma)/tau_pi - projector terms (their Eq. 5, no pi theta term)
  * "dnmr-theta": the same plus -(4/3) pi theta (the default in hic.viscous_hydro)

    python tools/song_heinz_cucu.py --grid-size 161 --output docs/results/song_heinz_cucu.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from hic import viscous_hydro as vh
from hic.eos import EquationOfState

HBARC = 0.1973269804
DEGENERACY = 16.0 + 10.5 * 2.5  # gluons + (21/2) Nf, Nf = 2.5
A_EOS_I = DEGENERACY * math.pi**2 / 30.0 / HBARC**3  # e = A T^4 (GeV/fm^3, T in GeV)
CONFORMAL = EquationOfState(kind="conformal", speed_of_sound2=1 / 3, low_speed_of_sound2=1 / 3,
                            high_speed_of_sound2=1 / 3, transition_energy=1.0, transition_width=1.0)


def temperature_of(energy: torch.Tensor) -> torch.Tensor:
    return (torch.clamp(energy, min=1.0e-14) / A_EOS_I) ** 0.25


def thickness(r: np.ndarray, radius=4.2, diffuseness=0.596, rho0=0.17, zmax=15.0, nz=601) -> np.ndarray:
    z = np.linspace(-zmax, zmax, nz)
    dist = np.sqrt(r[..., None] ** 2 + z**2)
    return np.trapezoid(rho0 / (1.0 + np.exp((dist - radius) / diffuseness)), z, axis=-1)


def wounded_nucleons(x: np.ndarray, y: np.ndarray, b: float, mass_number=63, sigma=4.0) -> np.ndarray:
    ta = thickness(np.hypot(x + b / 2, y))
    tb = thickness(np.hypot(x - b / 2, y))
    a = mass_number
    return ta * (1.0 - (1.0 - sigma * tb / a) ** a) + tb * (1.0 - (1.0 - sigma * ta / a) ** a)


def observables(state: vh.ViscousState, xx: torch.Tensor, yy: torch.Tensor, t_dec: float) -> dict[str, float]:
    prim = vh.recover_primitive(state, CONFORMAL, 1.0e-14)
    e, vx, vy, gamma = prim["e"], prim["vx"], prim["vy"], prim["ut"]
    p = e / 3.0
    w = (e + p) * gamma * gamma
    txx, tyy = w * vx * vx + p, w * vy * vy + p
    pxx, pyy = state.shear[0], state.shear[2]
    weight = gamma * e
    hot = temperature_of(e) > t_dec
    out = {
        "flow_anisotropy": float(torch.sum(weight * (vx.abs() - vy.abs())) / torch.sum(weight)),
        "eps_p": float(torch.sum(txx - tyy) / torch.sum(txx + tyy)),
        "eps_p_total": float(torch.sum(txx - tyy + pxx - pyy) / torch.sum(txx + tyy + pxx + pyy)),
        "eps_x": float(torch.sum(e * (yy**2 - xx**2)) / torch.sum(e * (yy**2 + xx**2))),
        "mean_vt": float(torch.sum(weight * torch.hypot(vx, vy)) / torch.sum(weight)),
        "max_temperature": float(temperature_of(e).max()),
        "hot_cells": int(hot.sum()),
        "pi_over_w": float(torch.sum((pxx.abs() + pyy.abs())[hot]) / torch.sum((e + p)[hot]).clamp(min=1e-30)) if hot.any() else 0.0,
    }
    return out


def run(case: str, params: vh.ViscousParameters, energy: torch.Tensor, xx, yy, dx, tau0, tau_end, cfl, t_dec,
        sample_every: float, initial_shear: str) -> list[dict]:
    zero = torch.zeros_like(energy)
    state = vh.initial_state(energy, zero, zero, CONFORMAL, params, temperature_of, tau0, dx, initial_shear=initial_shear)
    tau, dt = tau0, cfl * dx
    rows, next_sample = [], tau0
    while True:
        if tau >= next_sample - 1.0e-9:
            rows.append({"case": case, "tau_minus_tau0": round(tau - tau0, 4), **observables(state, xx, yy, t_dec)})
            next_sample += sample_every
            print(f"{case:>14s} tau-tau0={tau - tau0:5.2f} <|vx|-|vy|>={rows[-1]['flow_anisotropy']:.4f} "
                  f"eps_p={rows[-1]['eps_p']:.4f} eps'_p={rows[-1]['eps_p_total']:.4f} Tmax={rows[-1]['max_temperature']:.3f}",
                  flush=True)
            if rows[-1]["hot_cells"] == 0:
                break
        if tau >= tau_end - 1.0e-9:
            break
        h = min(dt, tau_end - tau)
        state = vh.viscous_step(state, tau, h, dx, CONFORMAL, params, temperature_of, energy_floor=1.0e-14)
        tau += h
    return rows


CASES = {
    # name: (eta/s, c_pi, theta coefficient, initial shear)
    "ideal": (1.0e-9, 3.0, 4.0 / 3.0, "ns"),
    "song-heinz": (0.08, 3.0, 0.0, "ns"),
    "song-heinz-pi0": (0.08, 3.0, 0.0, "zero"),
    "dnmr-theta": (0.08, 3.0, 4.0 / 3.0, "ns"),
    "dnmr-theta-c5": (0.08, 5.0, 4.0 / 3.0, "ns"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--extent", type=float, default=10.0)
    parser.add_argument("--impact-parameter", type=float, default=7.0)
    parser.add_argument("--tau-end", type=float, default=8.6)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--freezeout-temperature", type=float, default=0.130)
    parser.add_argument("--sample-every", type=float, default=0.5)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    axis = np.linspace(-args.extent, args.extent, args.grid_size)
    dx = float(axis[1] - axis[0])
    xg, yg = np.meshgrid(axis, axis)
    scale = 30.0 / float(wounded_nucleons(np.zeros(1), np.zeros(1), 0.0)[0])
    energy = torch.as_tensor(scale * wounded_nucleons(xg, yg, args.impact_parameter) + 1.0e-12)
    xx, yy = torch.as_tensor(xg), torch.as_tensor(yg)
    print(f"Cu+Cu b={args.impact_parameter} fm: e_max={float(energy.max()):.2f} GeV/fm^3, "
          f"T_max={float(temperature_of(energy).max()):.3f} GeV, dx={dx:.3f} fm")
    rows = []
    for case in args.cases:
        eta_over_s, c_pi, theta_coefficient, initial = CASES[case]
        params = vh.ViscousParameters(eta_over_s=eta_over_s, shear_relaxation=c_pi,
                                      shear_theta_coefficient=theta_coefficient)
        rows += run(case, params, energy, xx, yy, dx, 0.6, args.tau_end, args.cfl, args.freezeout_temperature,
                    args.sample_every, initial)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
