import math

import pytest
import torch

from hic import hydro_convergence, hydro_validation
from hic.ideal_hydro_2d import (
    apply_bjorken_cooling,
    apply_interface_fluxes,
    boundary_outflow,
    conserved_from_primitive,
    conserved_totals,
    primitive_from_conserved,
    rusanov_interface_fluxes,
    rusanov_step,
)
from hic.nuclear_collision_2d import make_grid


def random_primitives(count: int, dtype=torch.float64, max_speed: float = 0.95):
    generator = torch.Generator().manual_seed(11)
    energy = 0.05 + 5.0 * torch.rand(count, generator=generator, dtype=torch.float64)
    speed = max_speed * torch.rand(count, generator=generator, dtype=torch.float64)
    angle = 2.0 * math.pi * torch.rand(count, generator=generator, dtype=torch.float64)
    return energy.to(dtype), (speed * torch.cos(angle)).to(dtype), (speed * torch.sin(angle)).to(dtype)


@pytest.mark.parametrize("name", ["conformal_eos", "crossover_eos", "table_eos"])
def test_primitive_recovery_round_trip(name, request):
    model = request.getfixturevalue(name)
    energy, velocity_x, velocity_y = random_primitives(5000)
    state = conserved_from_primitive(energy, velocity_x, velocity_y, model)
    recovered, pressure, recovered_x, recovered_y = primitive_from_conserved(state[0], state[1], state[2], model, 1.0e-12)
    assert torch.allclose(recovered, energy, rtol=1.0e-5)
    assert torch.allclose(pressure, model.pressure(energy), rtol=1.0e-5)
    assert torch.allclose(recovered_x, velocity_x, atol=1.0e-6)
    assert torch.allclose(recovered_y, velocity_y, atol=1.0e-6)


@pytest.mark.parametrize("velocity", [0.0, 0.4])
def test_uniform_state_is_preserved(conformal_eos, cpu, velocity):
    _, _, dx = make_grid(32, 5.0, cpu)
    energy = torch.full((32, 32), 1.0)
    state = conserved_from_primitive(energy, torch.full_like(energy, velocity), torch.zeros_like(energy), conformal_eos)
    evolved = state
    for _ in range(10):
        evolved = rusanov_step(evolved, dx, 0.25 * dx, conformal_eos, 1.0e-8)
    assert torch.equal(evolved, state)


def test_boundary_outflow_closes_the_energy_budget(table_eos, cpu):
    xx, yy, dx = make_grid(48, 6.0, cpu)
    energy = 1.0e-8 + 2.0 * torch.exp(-((xx - 3.5) ** 2 + yy**2) / 2.0)
    state = conserved_from_primitive(energy, torch.zeros_like(energy), torch.zeros_like(energy), table_eos)
    initial_energy, _, _ = conserved_totals(state, dx)
    dt = 0.25 * dx
    outflow = torch.zeros(3)
    for _ in range(40):
        interface_x, interface_y = rusanov_interface_fluxes(state, table_eos, 1.0e-8)
        outflow += boundary_outflow(interface_x, interface_y, dx, dt)
        state = apply_interface_fluxes(state, interface_x, interface_y, dx, dt, 1.0e-8)
    final_energy, _, _ = conserved_totals(state, dx)
    assert float(outflow[0]) > 1.0e-3 * initial_energy
    assert final_energy + float(outflow[0]) == pytest.approx(initial_energy, rel=1.0e-5)


def test_source_step_reproduces_bjorken_cooling(conformal_eos):
    energy = torch.full((4,), 2.0, dtype=torch.float64)
    zero = torch.zeros_like(energy)
    state = conserved_from_primitive(energy, zero, zero, conformal_eos)
    tau, dt = 0.6, 0.05
    for _ in range(100):
        state = apply_bjorken_cooling(state, tau, dt, conformal_eos, 1.0e-12)
        tau += dt
    expected = 2.0 * (0.6 / tau) ** (4.0 / 3.0)
    assert torch.allclose(state[0], torch.full_like(energy, expected), rtol=1.0e-8)


@pytest.mark.parametrize("name", ["conformal_eos", "crossover_eos", "table_eos"])
def test_source_step_conserves_tau_times_momentum(name, request):
    """With no gradients, d(tau T^{tau i})/dtau = 0 and d(tau T^{tau tau})/dtau = -p."""
    model = request.getfixturevalue(name)
    energy, velocity_x, velocity_y = random_primitives(200, max_speed=0.8)
    initial = conserved_from_primitive(energy, velocity_x, velocity_y, model)
    tau0, tau1 = 0.6, 2.0

    def evolve(steps: int) -> torch.Tensor:
        state, tau, dt = initial, tau0, (tau1 - tau0) / steps
        for _ in range(steps):
            state = apply_bjorken_cooling(state, tau, dt, model, 1.0e-12)
            tau += dt
        return state

    # 40 steps match the reference event's dtau ~ 0.036 fm/c. RK4 drops to
    # second order across the kinks of the tabulated p(e), so coarser steps
    # need a looser tolerance.
    coarse, fine = evolve(40), evolve(640)
    assert torch.allclose(tau1 * coarse[1:], tau0 * initial[1:], rtol=1.0e-12, atol=1.0e-14)
    assert torch.allclose(coarse[0], fine[0], rtol=2.0e-5)
    # The old fixed-velocity cooling damped momentum as tau^-(1+cs2); check that
    # the transverse velocity instead grows as longitudinal work removes energy.
    _, _, final_x, final_y = primitive_from_conserved(coarse[0], coarse[1], coarse[2], model, 1.0e-12)
    moving = torch.hypot(velocity_x, velocity_y) > 0.05
    assert torch.all(torch.hypot(final_x, final_y)[moving] > torch.hypot(velocity_x, velocity_y)[moving])


def test_gubser_profile_solves_the_milne_equations():
    """Finite-difference residuals of the analytic flow against the exact sources."""
    q, tau0 = 1.0, 1.0
    points = [(1.2, 0.7, 0.3), (1.5, 1.4, -0.9), (2.0, 0.2, 2.1)]
    step = 1.0e-5

    def conserved(tau, x, y):
        xx = torch.tensor([[x]], dtype=torch.float64)
        yy = torch.tensor([[y]], dtype=torch.float64)
        energy, vx, vy = hydro_validation.gubser_profile(tau, xx, yy, q, tau0)
        pressure = energy / 3.0
        gamma2 = 1.0 / (1.0 - vx * vx - vy * vy)
        total = (energy + pressure) * gamma2
        return [float(value) for value in (total - pressure, total * vx, total * vy, pressure, vx, vy)]

    for tau, x, y in points:
        energy_density, momentum_x, _, pressure, _, _ = conserved(tau, x, y)
        d_tau = [(a - b) / (2 * step) for a, b in zip(conserved(tau + step, x, y), conserved(tau - step, x, y))]
        plus_x, minus_x = conserved(tau, x + step, y), conserved(tau, x - step, y)
        plus_y, minus_y = conserved(tau, x, y + step), conserved(tau, x, y - step)
        divergence_m = (plus_x[1] - minus_x[1] + plus_y[2] - minus_y[2]) / (2 * step)
        flux_xx = lambda c: c[1] * c[4] + c[3]  # noqa: E731
        flux_yx = lambda c: c[1] * c[5]  # noqa: E731
        divergence_txx = (flux_xx(plus_x) - flux_xx(minus_x) + flux_yx(plus_y) - flux_yx(minus_y)) / (2 * step)
        assert d_tau[0] + divergence_m == pytest.approx(-(energy_density + pressure) / tau, rel=1.0e-6)
        assert d_tau[1] + divergence_txx == pytest.approx(-momentum_x / tau, rel=1.0e-6)


def test_gubser_flow_converges_at_first_order(cpu):
    args = hydro_validation.parse_args(["--cpu"])
    row = hydro_validation.gubser_flow_test(args, cpu)
    assert row["status"] == "PASS"
    assert row["value"] < 0.15
    assert row["convergence_order"] > 0.8


def test_validation_suite_passes():
    rows = hydro_validation.run_validation(hydro_validation.parse_args(["--cpu"]))
    assert {row["test"] for row in rows} >= {"uniform_rest", "uniform_boost", "gaussian_fireball", "gubser_flow"}
    assert all(row["status"] in {"PASS", "SKIP"} for row in rows), rows


def test_convergence_scan_runs(tmp_path):
    output = tmp_path / "scan.csv"
    hydro_convergence.main(["--cpu", "--grid-sizes", "40", "--cfl-values", "0.25,0.35", "--output", str(output)])
    assert output.read_text().count("PASS") >= 2
