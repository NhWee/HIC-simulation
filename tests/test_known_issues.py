"""Physics checks that the current code is known to FAIL (docs/ROADMAP.md, P0 items).

They are marked ``xfail(strict=True, raises=AssertionError)``: the suite stays green while the bug exists, and turns red
(XPASS) once a fix lands, so the marker has to be removed together with the fix. Any other exception is a real failure.
"""
from pathlib import Path

import numpy as np
import pytest

from hic.eos import read_eos_table

EOS_TABLE = Path(__file__).resolve().parents[1] / "data" / "eos_table.csv"


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
