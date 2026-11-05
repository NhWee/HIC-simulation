"""The normalization calibration recovers a known initial energy density (ROADMAP P0-2)."""
import pytest

from hic.calibrate_normalization import calibrate, evaluate

BASE = ["--cpu", "--grid-size", "64", "--particles", "0", "--impact-parameter", "2.2"]


@pytest.mark.parametrize(
    ("observable", "tolerance", "peak_rel"),
    [("entropy", 0.002, 0.01), ("dnch_deta_direct", 0.01, 0.04)],  # the latter is a Monte Carlo estimate
)
def test_calibration_recovers_known_normalization(observable, tolerance, peak_rel):
    truth = 14.0
    target = evaluate(BASE, truth, [3], observable)
    result = calibrate(BASE, [3], observable, target, initial_peak=6.0, tolerance=tolerance)
    assert result["calibrated_initial_peak_energy_gev_fm3"] == pytest.approx(truth, rel=peak_rel)
    assert len(result["history"]) <= 6
