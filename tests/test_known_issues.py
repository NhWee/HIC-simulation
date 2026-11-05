"""Physics checks that the current code is known to FAIL (see docs/ROADMAP.md).

Mark them ``xfail(strict=True, raises=AssertionError)``: the suite stays green while the defect
exists and turns red (XPASS) once a fix lands, so the marker is removed together with the fix.
Any other exception is a real failure.

The EoS (P0-1) and Milne source-term (P0-3) checks that used to live here now pass and moved to
tests/test_eos.py and tests/test_hydro_sources.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import hic.blender_event_pipeline as pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="ROADMAP P0-4: freeze-out ignores cells that first exceed T_sw after tau0",
)
def test_every_cell_that_was_hot_is_on_the_freezeout_surface(monkeypatch):
    # With the lattice EoS the reference geometry at 24 GeV/fm^3 has ~10% of its hot cells heated
    # only after tau0 (the expanding edge, which carries the strongest in-plane flow). The
    # `initially_hot` mask in evolve_hydrodynamics drops them from the Cooper-Frye surface.
    monkeypatch.setattr(
        sys, "argv", ["x", "--cpu", "--grid-size", "60", "--initial-peak-energy", "24",
                      "--eos-table", str(REPO_ROOT / "data" / "eos_table.csv")]
    )
    args = pipeline.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    xx, yy, dx = pipeline.make_grid(args.grid_size, args.extent, torch.device("cpu"))
    eos = pipeline.build_table_eos(args.eos_table)
    geometry = pipeline.build_event_geometry(args, xx, yy, rng)
    preequilibrium = pipeline.evolve_pre_equilibrium(args, geometry.source, eos)
    hydro = pipeline.evolve_hydrodynamics(args, preequilibrium, eos, xx, yy, dx)
    ever_hot = np.any(hydro.temperature >= args.freezeout_temperature, axis=0)
    on_surface = np.isfinite(hydro.freezeout_tau)
    missing = int(np.count_nonzero(ever_hot & ~on_surface))
    assert missing == 0, f"{missing} of {int(ever_hot.sum())} hot cells are missing from the freeze-out surface"
