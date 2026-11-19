import math

import pytest
import torch

from hic import blender_event_pipeline as pipeline
from hic.nuclear_collision_2d import make_grid


def perfect_fluid_tensor(energy, pressure, velocity_x, velocity_y):
    gamma2 = 1.0 / (1.0 - velocity_x**2 - velocity_y**2)
    enthalpy = (energy + pressure) * gamma2
    return (
        enthalpy - pressure,
        enthalpy * velocity_x,
        enthalpy * velocity_y,
        enthalpy * velocity_x * velocity_x + pressure,
        enthalpy * velocity_x * velocity_y,
        enthalpy * velocity_y * velocity_y + pressure,
    )


@pytest.mark.parametrize("pressure_ratio", [1.0 / 3.0, 0.5])
def test_landau_matching_recovers_a_boosted_fluid(pressure_ratio):
    generator = torch.Generator().manual_seed(8)
    energy = 0.5 + 5.0 * torch.rand(500, generator=generator, dtype=torch.float64)
    speed = 0.85 * torch.rand(500, generator=generator, dtype=torch.float64)
    angle = 2.0 * math.pi * torch.rand(500, generator=generator, dtype=torch.float64)
    velocity_x, velocity_y = speed * torch.cos(angle), speed * torch.sin(angle)
    tensor = perfect_fluid_tensor(energy, pressure_ratio * energy, velocity_x, velocity_y)
    matched, matched_x, matched_y, residual = pipeline.landau_match_transverse_tensor(*tensor, energy_floor=1.0e-10)
    assert torch.allclose(matched, energy, rtol=1.0e-8)
    assert torch.allclose(matched_x, velocity_x, atol=1.0e-8)
    assert torch.allclose(matched_y, velocity_y, atol=1.0e-8)
    assert float(residual.max()) < 1.0e-8


def test_free_streaming_is_traceless_and_flows_outward(table_eos, cpu):
    args = pipeline.parse_args(["--grid-size", "64", "--preeq-angles", "24"])
    xx, yy, dx = make_grid(args.grid_size, args.extent, cpu)
    source = args.initial_peak_energy * torch.exp(-(xx**2 / 8.0 + yy**2 / 4.0))
    result = pipeline.evolve_pre_equilibrium(args, source, table_eos)
    # Massless transverse streams carry T^xx + T^yy = T^00 and no longitudinal pressure.
    trace = result.stress_xx + result.stress_yy
    assert float(abs(trace - result.energy_density).max()) < 1.0e-5 * float(result.energy_density.max())
    # tau * dE/deta is conserved by free streaming while the profile stays on the grid.
    totals = [float(tau) * float(frame.sum()) for tau, frame in zip(result.tau, result.energy_density)]
    assert max(totals) == pytest.approx(min(totals), rel=2.0e-3)
    assert float(abs(result.velocity_x[0]).max()) < 1.0e-6
    radial = result.velocity_x[-1] * xx.numpy() + result.velocity_y[-1] * yy.numpy()
    hot = result.landau_energy_density[-1] > 0.05 * result.landau_energy_density[-1].max()
    assert (radial[hot] >= -1.0e-6).all()
    # The source is narrower along y, so the energy-weighted preflow is larger along y.
    weight = result.landau_energy_density[-1] * hot
    flow_x = float((weight * abs(result.velocity_x[-1])).sum() / weight.sum())
    flow_y = float((weight * abs(result.velocity_y[-1])).sum() / weight.sum())
    assert flow_y > 1.2 * flow_x > 0.0
    assert float(result.landau_energy_density[-1].max()) == pytest.approx(args.initial_peak_energy, rel=1.0e-6)
