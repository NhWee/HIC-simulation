"""The tabulated lattice-QCD equation of state (ROADMAP P0-1)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hic.blender_event_pipeline import build_table_eos
from hic.lattice_eos import HBARC, P_IDEAL, build_table, thermodynamics
from hic.eos import read_eos_table

EOS_TABLE = Path(__file__).resolve().parents[1] / "data" / "eos_table.csv"


@pytest.fixture(scope="module")
def table():
    names = ("energy_density", "pressure", "sound_speed2", "temperature", "entropy_density")
    data = np.genfromtxt(EOS_TABLE, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=np.float64) for name in names}


def test_table_is_regenerated_by_lattice_eos_module(table):
    rows = build_table()
    assert len(rows) == table["temperature"].size
    for key in table:
        np.testing.assert_allclose([row[key] for row in rows], table[key], rtol=1.0e-7, err_msg=key)


def test_sound_speed_matches_pressure_slope(table):
    slope = np.diff(table["pressure"]) / np.diff(table["energy_density"])
    midpoint = 0.5 * (table["sound_speed2"][1:] + table["sound_speed2"][:-1])
    np.testing.assert_allclose(midpoint, slope, rtol=0.02)


def test_satisfies_thermodynamic_identities(table):
    e, p, t, s = (table[k] for k in ("energy_density", "pressure", "temperature", "entropy_density"))
    np.testing.assert_allclose(e + p, t * s, rtol=1.0e-6)
    dp_dt = np.diff(p) / np.diff(t)
    np.testing.assert_allclose(dp_dt, 0.5 * (s[1:] + s[:-1]), rtol=0.01)
    assert np.all(np.diff(e) > 0) and np.all(np.diff(p) > 0) and np.all(np.diff(s) > 0)


def test_covers_rhic_and_lhc_initial_energy_densities(table):
    assert table["energy_density"].max() >= 100.0
    assert table["temperature"].min() <= 0.080


def test_is_causal(table):
    assert np.all(table["sound_speed2"] > 0.0)
    assert np.all(table["sound_speed2"] < 1.0 / 3.0)


def test_reproduces_published_hotqcd_features():
    # Bazavov et al. (HotQCD), PRD 90, 094503 (2014): crossover region 145-163 MeV has
    # energy density 0.18-0.5 GeV/fm^3; trace anomaly peaks near 4 around T ~ 200 MeV;
    # the sound speed has its minimum (~0.14-0.15) near T_c.
    assert 0.15 < thermodynamics(0.145)["energy_density"] < 0.25
    assert 0.40 < thermodynamics(0.163)["energy_density"] < 0.60
    temperatures = np.linspace(0.12, 0.40, 281)
    rows = [thermodynamics(float(t)) for t in temperatures]
    trace = np.array([(r["energy_density"] - 3.0 * r["pressure"]) * HBARC**3 / r["temperature"] ** 4 for r in rows])
    assert 3.8 < trace.max() < 4.3 and 0.19 < temperatures[trace.argmax()] < 0.21
    cs2 = np.array([r["sound_speed2"] for r in rows])
    assert 0.13 < cs2.min() < 0.16 and 0.14 < temperatures[cs2.argmin()] < 0.155
    high = thermodynamics(0.56)
    ratio = high["pressure"] * HBARC**3 / 0.56**4 / P_IDEAL
    assert 0.8 < ratio < 1.0  # approaches but stays below the Stefan-Boltzmann limit


def test_pressure_outside_table_is_physical():
    eos = build_table_eos(EOS_TABLE)
    energy, pressure, sound_speed2, _ = read_eos_table(EOS_TABLE)
    e_low, e_high = energy[0], energy[-1]
    probe = torch.tensor([1.0e-8, 0.5 * e_low, e_low, e_high, 2.0 * e_high], dtype=torch.float64)
    p = eos.pressure(probe)
    assert torch.all(p >= 0.0) and torch.all(p <= probe / 3.0 + 1.0e-12)
    assert float(p[0]) < 1.0e-8  # vacuum cells carry no pressure
    assert float(p[2]) == pytest.approx(pressure[0], rel=1.0e-9)
    assert float(p[3]) == pytest.approx(pressure[-1], rel=1.0e-9)
    assert float(p[4]) == pytest.approx(pressure[-1] + sound_speed2[-1] * e_high, rel=1.0e-9)
    # Continuity just inside/outside both edges.
    eps = 1.0e-9
    edges = torch.tensor([e_low * (1 - eps), e_low * (1 + eps), e_high * (1 - eps), e_high * (1 + eps)], dtype=torch.float64)
    edge_p = eos.pressure(edges)
    assert float(edge_p[0]) == pytest.approx(float(edge_p[1]), rel=1.0e-6)
    assert float(edge_p[2]) == pytest.approx(float(edge_p[3]), rel=1.0e-6)
