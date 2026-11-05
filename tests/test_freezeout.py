"""Freeze-out hypersurface and Cooper-Frye sampling (ROADMAP P0-4)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import hic.blender_event_pipeline as pipeline
from hic import hadrons

REPO_ROOT = Path(__file__).resolve().parents[1]
EOS_TABLE = REPO_ROOT / "data" / "eos_table.csv"


def run_event(monkeypatch, grid_size: int, peak_energy: float):
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--cpu", "--grid-size", str(grid_size), "--initial-peak-energy", str(peak_energy),
         "--eos-table", str(EOS_TABLE)],
    )
    args = pipeline.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    xx, yy, dx = pipeline.make_grid(args.grid_size, args.extent, torch.device("cpu"))
    axis = np.linspace(-args.extent, args.extent, args.grid_size, dtype=np.float32)
    eos = pipeline.build_table_eos(args.eos_table)
    geometry = pipeline.build_event_geometry(args, xx, yy, rng)
    preequilibrium = pipeline.evolve_pre_equilibrium(args, geometry.source, eos)
    hydro = pipeline.evolve_hydrodynamics(args, preequilibrium, eos, xx, yy, dx)
    surface = pipeline.build_freezeout_surface(hydro, axis, dx)
    return args, hydro, surface, dx


@pytest.fixture(scope="module")
def event_24(request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    return run_event(monkeypatch, grid_size=60, peak_energy=24.0)


def test_isotherm_crossing_normals_for_moving_planar_fronts():
    # T(tau, x) = T_sw + k (c tau - x): a heating front moving toward +x at speed c.
    # d_mu T = k (c, -1, 0), so dsigma_mu = -tau dx^2 d_mu T / |d_tau T| = tau dx^2 (-1, 1/c, 0).
    k, c, threshold, dx, dt, tau0 = 0.05, 0.4, 0.145, 0.1, 0.3, 2.0  # front moves 0.12 > dx per step
    x = torch.arange(-5, 16, dtype=torch.float64) * dx  # spans the front near x = 0.8-0.9
    xx = x.view(1, -1).expand(5, -1).clone()
    field = lambda tau: threshold + k * (c * tau - xx) + 1.0e-4  # offset puts the front between cells
    zero = torch.zeros_like(xx)
    elements = pipeline.isotherm_crossings(field(tau0), field(tau0 + dt), threshold, tau0, dt, dx, (zero, zero), (zero, zero))
    assert elements["tau"].numel() > 0 and bool(torch.all(elements["heating"]))
    area = elements["tau"] * dx * dx
    torch.testing.assert_close(elements["d_sigma_tau"], -area)
    torch.testing.assert_close(elements["d_sigma_x"], area / c)
    torch.testing.assert_close(elements["d_sigma_y"], torch.zeros_like(area))
    # The time-reversed field cools: the normal flips to the familiar tau dx^2 (1, -d_x tau_f, 0).
    cooling = pipeline.isotherm_crossings(field(tau0 + dt), field(tau0), threshold, tau0, dt, dx, (zero, zero), (zero, zero))
    assert not bool(torch.any(cooling["heating"]))
    torch.testing.assert_close(cooling["d_sigma_tau"], cooling["tau"] * dx * dx)


def test_every_cell_that_was_hot_is_on_the_freezeout_surface(event_24):
    args, hydro, surface, _ = event_24
    ever_hot = np.any(hydro.temperature >= args.freezeout_temperature, axis=0)
    on_surface = np.isfinite(hydro.freezeout_tau)
    assert np.count_nonzero(ever_hot & ~on_surface) == 0
    assert np.count_nonzero(surface.heating) > 0  # the expanding edge is represented


def test_entropy_flux_through_freezeout_surface_matches_initial_entropy(event_24):
    # Fluid elements only cool in ideal hydro, so all entropy that is above T_sw at tau0 leaves
    # through the isotherm: S(tau0, T >= T_sw) = integral of s(T_sw) u^mu dsigma_mu.
    args, hydro, surface, dx = event_24
    table = np.genfromtxt(EOS_TABLE, delimiter=",", names=True)
    entropy = lambda t: np.interp(t, table["temperature"], table["entropy_density"])
    hot = hydro.temperature[0] >= args.freezeout_temperature
    gamma0 = 1.0 / np.sqrt(1.0 - hydro.velocity_x[0] ** 2 - hydro.velocity_y[0] ** 2)
    initial = float(np.sum((args.tau0 * entropy(hydro.temperature[0]) * gamma0)[hot]) * dx * dx)
    velocity_x = surface.velocity_x.astype(np.float64)
    velocity_y = surface.velocity_y.astype(np.float64)
    gamma = 1.0 / np.sqrt(1.0 - velocity_x**2 - velocity_y**2)
    u_dsigma = gamma * (surface.d_sigma_tau + velocity_x * surface.d_sigma_x + velocity_y * surface.d_sigma_y)
    flux = entropy(args.freezeout_temperature) * u_dsigma
    assert float(flux.sum()) == pytest.approx(initial, rel=0.03)
    assert float(-flux[flux < 0].sum()) < 0.01 * float(flux.sum())  # negative Cooper-Frye flux is small


def quadrature_pion_v2(surface, temperature: float) -> float:
    mass = 0.13957
    pt = np.linspace(0.02, 2.5, 25)
    phi = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    eta = np.linspace(-3.5, 3.5, 36)
    v = (surface.velocity_x.astype(np.float64), surface.velocity_y.astype(np.float64))
    gamma = 1.0 / np.sqrt(1.0 - v[0] ** 2 - v[1] ** 2)
    numerator = denominator = 0.0
    for p in pt:
        mt = math.sqrt(p * p + mass * mass)
        cosh = np.cosh(eta)[None, None, :]
        cos_phi, sin_phi = np.cos(phi)[None, :, None], np.sin(phi)[None, :, None]
        p_dot_sigma = (
            mt * cosh * surface.d_sigma_tau[:, None, None]
            + p * cos_phi * surface.d_sigma_x[:, None, None]
            + p * sin_phi * surface.d_sigma_y[:, None, None]
        )
        p_dot_u = mt * cosh * gamma[:, None, None] - p * gamma[:, None, None] * (
            cos_phi * v[0][:, None, None] + sin_phi * v[1][:, None, None]
        )
        weight = (np.maximum(p_dot_sigma, 0.0) * np.exp(-p_dot_u / temperature)).sum(axis=(0, 2)) * p
        numerator += float((weight * np.cos(2.0 * (phi - 0.5 * math.pi))).sum())
        denominator += float(weight.sum())
    return numerator / denominator


def test_cooper_frye_sampler_agrees_with_direct_quadrature(event_24):
    args, _, surface, _ = event_24
    expected = quadrature_pion_v2(surface, args.freezeout_temperature)
    states, index = hadrons.hadron_table()
    pions = np.asarray([index["pi+"], index["pi-"]])
    rng = np.random.default_rng(5)
    samples = np.array([
        pipeline.sample_cooper_frye_particles(args, surface, rng, species=pions, decays=False).v2 for _ in range(6)
    ])
    error = samples.std(ddof=1) / math.sqrt(samples.size)
    assert expected > 0.01  # a real in-plane signal, not noise
    assert abs(samples.mean() - expected) < 4.0 * error + 0.003


def test_auto_particle_count_follows_cooper_frye_multiplicity(event_24, monkeypatch):
    args, _, surface, _ = event_24
    monkeypatch.setattr(args, "particles", 0)
    counts, expected = [], []
    rng = np.random.default_rng(11)
    for _ in range(5):
        particles = pipeline.sample_cooper_frye_particles(args, surface, rng)
        counts.append(particles.primaries)
        expected.append(particles.expected_in_window)
        assert particles.species.size >= particles.primaries  # decay products are added
    mean_expected = float(np.mean(expected))
    assert np.std(expected) < 0.05 * mean_expected  # the Monte Carlo integral is stable
    assert abs(np.mean(counts) - mean_expected) < 4.0 * math.sqrt(mean_expected / len(counts)) + 1.0


def test_resonance_decays_raise_multiplicity_and_order_mean_pt_by_mass(event_24):
    args, _, surface, _ = event_24
    with_decays = pipeline.sample_cooper_frye_particles(args, surface, np.random.default_rng(2))
    without = pipeline.sample_cooper_frye_particles(args, surface, np.random.default_rng(2), decays=False)
    assert with_decays.dnch_deta > 1.3 * without.dnch_deta  # feed-down from rho, omega, K*, Delta, ...
    pion, kaon, proton = with_decays.mean_pt_identified
    assert pion < kaon < proton
    assert with_decays.dnch_deta < with_decays.dnch_dy  # pseudorapidity Jacobian pT/mT < 1
