"""Hadron resonance list, decays and hadron-resonance-gas thermodynamics (ROADMAP P2-2)."""
import math
from fractions import Fraction as F

import numpy as np
import pytest

from hic.hadrons import (
    clebsch_gordan,
    decay_all,
    final_state_mask,
    hadron_table,
    hrg_thermodynamics,
)
from hic.lattice_eos import thermodynamics

STATES, INDEX = hadron_table()


def channel_fraction(parent: str, products: set[tuple[str, ...]]) -> float:
    total = 0.0
    for ratio, channel in STATES[INDEX[parent]].decays:
        if tuple(sorted(STATES[p].name for p in channel)) in {tuple(sorted(c)) for c in products}:
            total += ratio
    return total


def test_clebsch_gordan_standard_values():
    h = F(1, 2)
    assert clebsch_gordan(h, h, h, -h, F(1), F(0)) == pytest.approx(1.0 / math.sqrt(2.0))
    assert clebsch_gordan(F(1), F(1), F(1), F(-1), F(0), F(0)) == pytest.approx(1.0 / math.sqrt(3.0))
    assert clebsch_gordan(F(1), F(0), F(1), F(0), F(0), F(0)) == pytest.approx(-1.0 / math.sqrt(3.0))
    assert clebsch_gordan(F(1), F(0), F(1), F(0), F(1), F(0)) == pytest.approx(0.0)
    assert clebsch_gordan(F(1), F(1), h, -h, F(3, 2), h) == pytest.approx(1.0 / math.sqrt(3.0))
    # Orthonormality of a full coupling table: 1 x 1/2.
    for j in (F(1, 2), F(3, 2)):
        for m in np.arange(-float(j), float(j) + 1):
            norm = sum(
                clebsch_gordan(F(1), F(m1), h, F(m) - F(m1), j, F(m)) ** 2 for m1 in (-1, 0, 1)
            )
            assert norm == pytest.approx(1.0)


def test_every_decay_conserves_quantum_numbers_and_is_open():
    for hadron in STATES:
        if not hadron.decays:
            continue
        assert sum(r for r, _ in hadron.decays) == pytest.approx(1.0)
        for _, products in hadron.decays:
            daughters = [STATES[p] for p in products]
            assert sum(d.charge for d in daughters) == hadron.charge, hadron.name
            assert sum(d.baryon for d in daughters) == hadron.baryon, hadron.name
            assert sum(d.strangeness for d in daughters) == hadron.strangeness, hadron.name
            assert sum(d.mass for d in daughters) < hadron.mass, hadron.name


def test_final_states_are_the_strong_and_em_stable_hadrons():
    final = {STATES[i].name for i in np.flatnonzero(final_state_mask())}
    expected = {
        "pi+", "pi0", "pi-", "K+", "K0", "K-", "anti-K0", "p", "n", "anti-p", "anti-n",
        "Lambda", "anti-Lambda", "Sigma+", "Sigma-", "anti-Sigma+", "anti-Sigma-",
        "Xi0", "Xi-", "anti-Xi0", "anti-Xi-", "Omega-", "anti-Omega-",
    }
    assert final == expected


def test_isospin_branching_fractions():
    assert channel_fraction("Delta+(1232)", {("p", "pi0")}) == pytest.approx(2.0 / 3.0)
    assert channel_fraction("Delta++(1232)", {("p", "pi+")}) == pytest.approx(1.0)
    assert channel_fraction("K*+(892)", {("K0", "pi+")}) == pytest.approx(2.0 / 3.0)
    assert channel_fraction("rho0", {("pi+", "pi-")}) == pytest.approx(1.0)
    assert channel_fraction("rho+", {("pi+", "pi0")}) == pytest.approx(1.0)
    pipi = channel_fraction("f2(1270)", {("pi+", "pi-"), ("pi0", "pi0")})
    assert channel_fraction("f2(1270)", {("pi0", "pi0")}) == pytest.approx(pipi / 3.0)
    for state in ("Sigma+", "Sigma0", "Sigma-"):
        pion = {"Sigma+": "pi-", "Sigma0": "pi0", "Sigma-": "pi+"}[state]
        assert channel_fraction("Lambda(1405)", {(state, pion)}) == pytest.approx(1.0 / 3.0)


def test_antiparticles_are_conjugates():
    for hadron in STATES:
        if hadron.baryon == 0 and hadron.strangeness == 0:
            continue
        partner = [h for h in STATES if h.multiplet in (f"anti-{hadron.multiplet}", hadron.multiplet.removeprefix("anti-"))
                   and h is not hadron and h.charge == -hadron.charge and h.mass == hadron.mass
                   and h.baryon == -hadron.baryon and h.strangeness == -hadron.strangeness]
        assert partner, hadron.name
        conjugate_decays = sorted(
            (round(r, 12), tuple(sorted(-STATES[p].charge for p in prods))) for r, prods in partner[0].decays
        )
        decays = sorted((round(r, 12), tuple(sorted(STATES[p].charge for p in prods))) for r, prods in hadron.decays)
        assert [r for r, _ in conjugate_decays] == [r for r, _ in decays]


def test_decay_engine_conserves_four_momentum_and_ends_in_final_states():
    rng = np.random.default_rng(0)
    parents = np.repeat(np.flatnonzero([bool(h.decays) for h in STATES]), 50)
    masses = np.asarray([STATES[i].mass for i in parents])
    momentum = rng.normal(0.0, 1.5, size=(parents.size, 3))
    four = np.column_stack((np.sqrt((momentum**2).sum(axis=1) + masses**2), momentum))
    state, final, origin = decay_all(parents, four, rng)
    totals = np.zeros_like(four)
    np.add.at(totals, origin, final)
    np.testing.assert_allclose(totals, four, atol=1.0e-10)
    final_masses = np.asarray([STATES[i].mass for i in state])
    np.testing.assert_allclose(final[:, 0] ** 2 - (final[:, 1:] ** 2).sum(axis=1), final_masses**2, atol=1.0e-9)
    assert all(STATES[i].stable for i in np.unique(state))
    charge = np.zeros(parents.size)
    np.add.at(charge, origin, [STATES[i].charge for i in state])
    np.testing.assert_array_equal(charge, [STATES[i].charge for i in parents])


def test_three_body_decay_stays_inside_the_dalitz_region():
    rng = np.random.default_rng(1)
    eta = INDEX["eta"]
    four = np.tile([STATES[eta].mass, 0.0, 0.0, 0.0], (4000, 1))
    state, final, origin = decay_all(np.full(4000, eta), four, rng)
    names = np.asarray([STATES[i].name for i in state])
    for event in np.unique(origin)[:400]:
        rows = np.flatnonzero(origin == event)
        if sorted(names[rows]) != ["pi+", "pi-", "pi0"]:
            continue
        plus, minus = final[rows[names[rows] == "pi+"][0]], final[rows[names[rows] == "pi-"][0]]
        pair = plus + minus
        m12 = math.sqrt(pair[0] ** 2 - (pair[1:] ** 2).sum())
        assert 2 * 0.13957 - 1.0e-9 <= m12 <= STATES[eta].mass - 0.13498 + 1.0e-9
    fraction_3pi0 = np.mean([sorted(names[origin == e]) == ["pi0"] * 3 for e in range(2000)])
    assert fraction_3pi0 == pytest.approx(0.3268, abs=0.03)


def test_resonance_gas_entropy_is_close_to_the_lattice_eos_at_freezeout():
    # The list ends near 1.8 GeV and uses Boltzmann statistics, so it falls short of the lattice
    # entropy (which effectively contains every state) by ~10% at T_sw = 145 MeV and less below.
    ratio_145 = hrg_thermodynamics(0.145)["s"] / thermodynamics(0.145)["entropy_density"]
    ratio_130 = hrg_thermodynamics(0.130)["s"] / thermodynamics(0.130)["entropy_density"]
    assert 0.85 < ratio_145 < 1.05
    assert ratio_145 < ratio_130 < 1.05
