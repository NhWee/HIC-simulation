"""Physics checks that the current code is known to FAIL (docs/ROADMAP.md, P0 items).

They are marked ``xfail(strict=True, raises=AssertionError)``: the suite stays green while the bug exists, and turns red
(XPASS) once a fix lands, so the marker has to be removed together with the fix. Any other exception is a real failure.
"""
import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from hic.eos import build_eos, read_eos_table
from hic.ideal_hydro_2d import apply_bjorken_cooling, conserved_from_primitive, primitive_from_conserved

EOS_TABLE = Path(__file__).resolve().parents[1] / "data" / "eos_table.csv"


def conformal_eos():
    return build_eos(
        argparse.Namespace(
            eos="conformal",
            speed_of_sound2=1.0 / 3.0,
            low_speed_of_sound2=0.15,
            high_speed_of_sound2=1.0 / 3.0,
            transition_energy=1.0,
            transition_width=0.25,
            eos_table=None,
        )
    )


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="ROADMAP P0-3: Bjorken source split keeps v fixed instead of Milne sources")
def test_bjorken_source_conserves_tau_times_transverse_momentum():
    # For a uniform, transversely moving, boost-invariant fluid the Milne-coordinate equations give
    # d(T^{tau x})/dtau = -T^{tau x}/tau exactly, i.e. tau * T^{tau x} is constant.
    eos = conformal_eos()
    energy = torch.full((8, 8), 10.0, dtype=torch.float64)
    velocity_x = torch.full_like(energy, 0.4)
    state = conserved_from_primitive(energy, velocity_x, torch.zeros_like(energy), eos)
    tau, tau_end, steps = 0.6, 1.2, 600
    dt = (tau_end - tau) / steps
    initial = tau * float(state[1, 0, 0])
    for step in range(steps):
        state = apply_bjorken_cooling(state, tau + step * dt, dt, eos, 1.0e-12)
    final = tau_end * float(state[1, 0, 0])
    assert final == pytest.approx(initial, rel=1.0e-3)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="ROADMAP P0-1: EoS table sound_speed2 column is not dp/de")
def test_eos_table_sound_speed_matches_pressure_slope():
    energy, pressure, sound_speed2, _ = map(np.asarray, read_eos_table(EOS_TABLE))
    slope = np.diff(pressure) / np.diff(energy)
    midpoint_cs2 = 0.5 * (sound_speed2[1:] + sound_speed2[:-1])
    np.testing.assert_allclose(midpoint_cs2, slope, rtol=0.05)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="ROADMAP P0-1: EoS table violates dp/dT = s = (e + p)/T")
def test_eos_table_satisfies_thermodynamic_identity():
    energy, pressure, _, temperature = map(np.asarray, read_eos_table(EOS_TABLE))
    dp_dt = np.diff(pressure) / np.diff(temperature)
    entropy = (energy + pressure) / temperature
    np.testing.assert_allclose(dp_dt, 0.5 * (entropy[1:] + entropy[:-1]), rtol=0.10)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="ROADMAP P0-1/P0-2: EoS table stops at 6 GeV/fm^3")
def test_eos_table_covers_rhic_initial_energy_densities():
    energy, *_ = read_eos_table(EOS_TABLE)
    assert max(energy) >= 50.0


def test_bjorken_cooling_at_rest_matches_analytic_law():
    # Control case that the current implementation does get right: v = 0, conformal EoS,
    # e(tau) = e0 (tau0/tau)^(4/3).
    eos = conformal_eos()
    energy = torch.full((4, 4), 10.0, dtype=torch.float64)
    zero = torch.zeros_like(energy)
    state = conserved_from_primitive(energy, zero, zero, eos)
    tau, tau_end, steps = 0.6, 1.2, 200
    dt = (tau_end - tau) / steps
    for step in range(steps):
        state = apply_bjorken_cooling(state, tau + step * dt, dt, eos, 1.0e-12)
    final_energy, *_ = primitive_from_conserved(state[0], state[1], state[2], eos, 1.0e-12)
    assert float(final_energy[0, 0]) == pytest.approx(10.0 * (tau / tau_end) ** (4.0 / 3.0), rel=1.0e-6)
