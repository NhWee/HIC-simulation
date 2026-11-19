import argparse
import math

import numpy as np
import pytest

from hic import blender_event_pipeline as pipeline
from hic import particlization


def exact_mean_pt(temperature: float, mass: float) -> float:
    pt = np.linspace(0.0, 40.0 * temperature + 4.0 * mass, 400_001)
    weight = pt * np.exp(-np.sqrt(pt * pt + mass * mass) / temperature)
    return float(np.trapezoid(pt * weight, pt) / np.trapezoid(weight, pt))


@pytest.mark.parametrize(("temperature", "mass"), [(0.145, 0.13957), (0.12, 0.49368), (0.17, 0.93827)])
def test_thermal_pt_sampler_matches_boltzmann_spectrum(temperature, mass):
    samples = particlization.sample_thermal_pt(np.random.default_rng(1), temperature, mass, 200_000)
    error = samples.std() / math.sqrt(samples.size)
    assert abs(samples.mean() - exact_mean_pt(temperature, mass)) < 4.0 * error


def test_thermal_pt_array_uses_each_temperature():
    temperatures = np.where(np.arange(200_000) % 2 == 0, 0.12, 0.17)
    samples = particlization.sample_thermal_pt_array(np.random.default_rng(2), temperatures, 0.13957)
    for parity, temperature in ((0, 0.12), (1, 0.17)):
        subset = samples[parity::2]
        assert abs(subset.mean() - exact_mean_pt(temperature, 0.13957)) < 4.0 * subset.std() / math.sqrt(subset.size)


def test_particlization_cli(tmp_path):
    surface = tmp_path / "surface.csv"
    rows = ["tau,x,y,energy_density,temperature,velocity_x,velocity_y,d_sigma_tau,d_sigma_x,d_sigma_y"]
    for index in range(40):
        angle = 2.0 * math.pi * index / 40.0
        rows.append(
            f"1.0,{3 * math.cos(angle)},{3 * math.sin(angle)},0.35,0.145,"
            f"{0.4 * math.cos(angle)},{0.4 * math.sin(angle)},0.02,0.0,0.0"
        )
    surface.write_text("\n".join(rows) + "\n")
    output, summary = tmp_path / "particles.csv", tmp_path / "summary.csv"
    particlization.main(
        ["--freezeout-surface", str(surface), "--particles", "500", "--output", str(output), "--summary-output", str(summary)]
    )
    assert len(output.read_text().splitlines()) == 501
    assert "mean_pt" in summary.read_text()


def test_lorentz_boost_preserves_invariant_mass():
    rng = np.random.default_rng(3)
    momentum = rng.normal(size=(1000, 3))
    mass = 0.5
    energy = np.sqrt(np.sum(momentum * momentum, axis=1) + mass * mass)
    beta = rng.uniform(-0.55, 0.55, size=(1000, 3))
    lab_energy, lab_momentum = pipeline.lorentz_boost(energy, momentum, beta.copy())
    invariant = lab_energy**2 - np.sum(lab_momentum * lab_momentum, axis=1)
    assert np.allclose(invariant, mass * mass, rtol=1.0e-10)
    rest = np.zeros((1, 3))
    boosted_energy, boosted_momentum = pipeline.lorentz_boost(np.array([mass]), rest, np.array([[0.6, 0.0, 0.0]]))
    gamma = 1.0 / math.sqrt(1.0 - 0.36)
    assert boosted_energy[0] == pytest.approx(gamma * mass)
    assert boosted_momentum[0, 0] == pytest.approx(gamma * 0.6 * mass)


def test_thermal_species_ratios_are_ordered_by_mass():
    probabilities = pipeline.thermal_species_probabilities(0.145)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2] > 0.0


def test_weighted_mean_and_error():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mean, error = pipeline.weighted_mean_and_error(values, np.ones(4))
    assert mean == pytest.approx(2.5)
    assert error == pytest.approx(math.sqrt(np.sum((values - 2.5) ** 2)) / 4.0)


def static_surface(cells: int = 400) -> pipeline.SurfaceResult:
    rng = np.random.default_rng(4)
    radius = np.sqrt(rng.uniform(0.0, 16.0, cells)).astype(np.float32)
    angle = rng.uniform(0.0, 2.0 * math.pi, cells).astype(np.float32)
    zeros = np.zeros(cells, dtype=np.float32)
    return pipeline.SurfaceResult(
        tau=np.full(cells, 5.0, dtype=np.float32),
        x=radius * np.cos(angle),
        y=radius * np.sin(angle),
        velocity_x=zeros,
        velocity_y=zeros,
        d_sigma_tau=np.full(cells, 0.05, dtype=np.float32),
        d_sigma_x=zeros,
        d_sigma_y=zeros,
    )


def test_cooper_frye_sampler_on_a_static_surface():
    args = argparse.Namespace(particles=300, freezeout_temperature=0.145, eta_window=0.8, observable_candidates=200_000)
    particles = pipeline.sample_cooper_frye_particles(args, static_surface(), np.random.default_rng(5))
    assert particles.species.size == 300
    assert np.all(particles.cooper_frye_weight >= 0.0)
    assert particles.observable_candidates == 200_000
    assert abs(particles.v2) < 4.0 * particles.v2_error
    assert abs(particles.v3) < 4.0 * particles.v3_error
    assert 0.3 < particles.mean_pt < 0.6
    assert particles.mean_pt_error < 2.0e-3
