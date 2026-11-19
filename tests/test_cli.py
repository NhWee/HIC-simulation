"""Smoke tests of the standalone command-line tools with small settings."""

import csv
from pathlib import Path

import pytest

from hic import (
    analyze_pipeline,
    educational_visual,
    event_pipeline,
    hydro_shock_tube,
    ideal_hydro_2d,
    nuclear_collision_2d,
    visual_simulation,
    visualize_results,
)


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def toy_chain(tmp_path_factory):
    folder = tmp_path_factory.mktemp("chain")
    hydro, freezeout = folder / "hydro.csv", folder / "freezeout.csv"
    # Module-scoped fixtures run before the per-test chdir; event_pipeline starts
    # "python -m hic...." subprocesses that need the repository root as cwd.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)
        ideal_hydro_2d.main(
            [
                "--cpu", "--bjorken-expansion", "--eos", "table", "--grid-size", "48", "--steps", "40",
                "--freezeout-output", str(freezeout), "--output", str(hydro),
            ]
        )
        summary = folder / "pipeline_summary.csv"
        event_pipeline.main(
            [
                "--events", "1", "--particles", "200", "--grid-size", "32", "--steps", "20",
                "--output-dir", str(folder / "events"), "--summary-output", str(summary),
            ]
        )
    return folder


def test_ideal_hydro_writes_freezeout_cells(toy_chain):
    assert len(read_rows(toy_chain / "hydro.csv")) > 1
    cells = read_rows(toy_chain / "freezeout.csv")
    assert cells and all(0.1 < float(row["temperature"]) < 0.2 for row in cells)


def test_event_pipeline_and_analysis(toy_chain, tmp_path):
    rows = read_rows(toy_chain / "pipeline_summary.csv")
    assert len(rows) == 1 and float(rows[0]["mean_pt"]) > 0.0
    analysis = tmp_path / "analysis.csv"
    analyze_pipeline.main(["--input", str(toy_chain / "pipeline_summary.csv"), "--output", str(analysis)])
    assert float(read_rows(analysis)[0]["events"]) == 1.0


def test_results_report(toy_chain, tmp_path):
    report = tmp_path / "report.html"
    visualize_results.main(
        [
            "--hydro", str(toy_chain / "hydro.csv"),
            "--freezeout", str(toy_chain / "freezeout.csv"),
            "--particles", str(toy_chain / "events" / "event_0000_particles.csv"),
            "--pipeline-summary", str(toy_chain / "pipeline_summary.csv"),
            "--output", str(report),
        ]
    )
    assert report.stat().st_size > 10_000


def test_shock_tube_conserves_energy(tmp_path):
    profile, diagnostics = tmp_path / "profile.csv", tmp_path / "diagnostics.csv"
    hydro_shock_tube.main(
        ["--cpu", "--grid-size", "64", "--steps", "30", "--output", str(profile), "--diagnostics-output", str(diagnostics)]
    )
    row = read_rows(diagnostics)[0]
    assert float(row["relative_energy_drift"]) < 1.0e-5
    assert 0.0 < float(row["center_max_velocity"]) < 1.0


@pytest.mark.parametrize("mode", ["smooth", "mc-glauber"])
def test_collision_viewer(tmp_path, mode):
    output = tmp_path / f"{mode}.html"
    nuclear_collision_2d.main(
        ["--cpu", "--initial-condition", mode, "--grid-size", "48", "--steps", "80", "--frame-every", "20", "--output", str(output)]
    )
    assert output.stat().st_size > 10_000


def test_visual_simulation_and_educational_board(tmp_path):
    simulation = tmp_path / "visual_simulation.html"
    visual_simulation.main(
        [
            "--device", "cpu", "--grid-size", "64", "--output-grid-size", "32", "--steps", "160",
            "--frame-every", "40", "--output", str(simulation),
        ]
    )
    board = tmp_path / "educational.html"
    educational_visual.main(["--simulation", str(simulation), "--output", str(board)])
    summary = educational_visual.read_visual_simulation_summary(simulation)
    assert summary["nPart"] > 0
    assert board.stat().st_size > 1_000
