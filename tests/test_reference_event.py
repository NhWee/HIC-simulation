"""Regression test of the reference Au+Au event against REPRODUCIBILITY.md.

The reference run uses the defaults: b = 5.7 fm, seed 17, 160 x 160 grid,
48 pre-equilibrium streams, 900 display particles, 1e6 Cooper-Frye candidates.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hic import blender_event_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = {
    "n_part": 236,
    "n_coll": 574,
    "initial_max_landau_effective_temperature_gev": 0.4748,
    "handoff_max_landau_effective_temperature_gev": 0.28024,
    "energy_weighted_mean_preequilibrium_flow": 0.1487,
    "qgp_cells_at_hydrodynamization": 3349,
    "freezeout_cells": 4039,
    "final_hydro_tau_fm_c": 4.657,
    "mean_pt_gev": 0.4629,
    "v2_reaction_plane": 0.0142,
}


@pytest.fixture(scope="module")
def reference_event(tmp_path_factory):
    output = tmp_path_factory.mktemp("reference") / "blender_physics_event.npz"
    pipeline.main(["--cpu", "--eos-table", str(ROOT / "data" / "eos_table.csv"), "--output", str(output)])
    audit = json.loads(output.with_suffix(".audit.json").read_text())
    with np.load(output) as cache:
        arrays = {key: cache[key] for key in cache.files}
    return audit, arrays


def test_reference_audit_passes(reference_event):
    audit, _ = reference_event
    assert audit["passed"] is True
    assert audit["n_part"] == REFERENCE["n_part"]
    assert audit["n_coll"] == REFERENCE["n_coll"]
    assert audit["sampled_particles"] == 900
    assert audit["observable_candidates"] == 1_000_000


def test_reference_temperatures_and_preflow(reference_event):
    audit, _ = reference_event
    for key in ("initial_max_landau_effective_temperature_gev", "handoff_max_landau_effective_temperature_gev"):
        assert audit[key] == pytest.approx(REFERENCE[key], rel=2.0e-3)
    assert audit["energy_weighted_mean_preequilibrium_flow"] == pytest.approx(
        REFERENCE["energy_weighted_mean_preequilibrium_flow"], abs=1.0e-3
    )
    assert audit["qgp_cells_at_hydrodynamization"] == pytest.approx(REFERENCE["qgp_cells_at_hydrodynamization"], rel=0.01)
    assert audit["freezeout_cells"] == pytest.approx(REFERENCE["freezeout_cells"], rel=0.01)


def test_reference_hydro_lifetime(reference_event):
    _, cache = reference_event
    assert float(cache["hydro_tau"][-1]) == pytest.approx(REFERENCE["final_hydro_tau_fm_c"], abs=0.05)
    assert np.all(np.diff(cache["max_temperature"]) <= 2.0e-3)


def test_reference_observables(reference_event):
    audit, _ = reference_event
    pt_tolerance = max(4.0 * audit["mean_pt_stat_error_gev"], 1.5e-3)
    v2_tolerance = max(4.0 * audit["v2_stat_error"], 2.0e-3)
    assert audit["mean_pt_gev"] == pytest.approx(REFERENCE["mean_pt_gev"], abs=pt_tolerance)
    assert audit["v2_reaction_plane"] == pytest.approx(REFERENCE["v2_reaction_plane"], abs=v2_tolerance)
    assert audit["v2_stat_error"] < 2.0e-3


def test_reference_cache_layout(reference_event):
    _, cache = reference_event
    assert str(cache["format_version"]) == "hic_blender_event_v3"
    assert cache["particle_species"].shape == (900,)
    for key in ("mean_pt_stat_error", "v2_stat_error", "v3_stat_error", "observable_candidates"):
        assert key in cache
