"""Headless checks of the desktop applications; skipped when their GUI toolkits are unavailable."""

import pytest


def import_or_skip(module: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as error:  # missing package or missing system library such as libEGL
        pytest.skip(f"{module} unavailable: {error}")


@pytest.fixture
def pyside_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return import_or_skip("hic.pyside_simulator_app")


def test_pyside_analytic_audits(pyside_app):
    assert pyside_app.run_physics_audit()
    assert pyside_app.run_visual_audit()


def test_pyside_hydro_cache_audits(pyside_app, tmp_path):
    cache_path = pyside_app.generate_hydro_cache(
        output=tmp_path / "cache.npz",
        impact_parameter=5.7,
        grid_size=96,
        extent=8.0,
        steps=240,
        sample_every=4,
        seed=7,
        cpu=True,
        target_max_energy=28.0,
        freezeout_energy=0.35,
    )
    cache = pyside_app.HydroFrameCache.load(cache_path)
    assert pyside_app.run_hydro_cache_audit(cache)
    assert pyside_app.run_visual_audit(cache)


def test_tkinter_app_smoke(capsys):
    app = import_or_skip("hic.educational_simulator_app")
    app.main(["--smoke-test"])
    assert "smoke ok" in capsys.readouterr().out
