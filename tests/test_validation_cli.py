"""Smoke tests for the existing validation command-line tools (CPU)."""
import csv

import pytest

from tests.conftest import assert_ok, run_module


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    "eos_args",
    [
        pytest.param([], id="conformal"),
        pytest.param(["--eos", "crossover"], id="crossover"),
        pytest.param(["--eos", "table", "--eos-table", "data/eos_table.csv"], id="table"),
    ],
)
def test_hydro_validation_suite_passes(tmp_path, eos_args):
    output = tmp_path / "hydro_validation.csv"
    result = run_module("hic.hydro_validation", "--cpu", "--output", output, *eos_args)
    assert_ok(result)
    rows = read_rows(output)
    assert {row["test"] for row in rows} == {
        "uniform_rest",
        "uniform_boost",
        "gaussian_fireball",
        "bjorken_uniform_cooling",
        "bjorken_crossover_convergence",
        "gubser_flow",
    }
    assert all(row["status"] in {"PASS", "SKIP"} for row in rows), rows


def test_hydro_convergence_scan_passes(tmp_path):
    output = tmp_path / "hydro_convergence.csv"
    result = run_module(
        "hic.hydro_convergence",
        "--cpu",
        "--grid-sizes", "40,60",
        "--cfl-values", "0.15,0.25",
        "--steps", "20",
        "--output", output,
    )
    assert_ok(result)
    rows = read_rows(output)
    assert len(rows) == 4
    assert all(row["status"] == "PASS" for row in rows), rows


def test_shock_tube_conserves_energy(tmp_path):
    profile = tmp_path / "shock.csv"
    diagnostics = tmp_path / "shock_diagnostics.csv"
    result = run_module(
        "hic.hydro_shock_tube", "--cpu", "--grid-size", "120",
        "--output", profile, "--diagnostics-output", diagnostics,
    )
    assert_ok(result)
    (row,) = read_rows(diagnostics)
    assert abs(float(row["relative_energy_drift"])) < 1.0e-6
    assert 0.0 < float(row["center_max_velocity"]) < 1.0
    energies = [float(r["energy_density"]) for r in read_rows(profile)]
    assert min(energies) > 0.0


def test_glauber_ensemble_and_centrality_analysis(tmp_path):
    ensemble = tmp_path / "glauber.csv"
    assert_ok(run_module("hic.glauber_ensemble", "--events", "6", "--grid-size", "60", "--output", ensemble))
    rows = read_rows(ensemble)
    assert len(rows) == 6
    for row in rows:
        assert 0 <= int(float(row["n_part"])) <= 2 * 197
        assert 0.0 <= float(row["eps2"]) <= 1.0

    analysis = tmp_path / "centrality.csv"
    assert_ok(run_module("hic.centrality_analysis", "--input", ensemble, "--output", analysis))
    assert read_rows(analysis)
