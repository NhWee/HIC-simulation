"""Boost-invariant source terms and the Gubser-flow benchmark (ROADMAP P0-3)."""
import math

import pytest
import torch

from hic.eos import EquationOfState
from hic.hydro_validation import gubser_flow_error, gubser_ideal_solution
from hic.ideal_hydro_2d import apply_bjorken_cooling, conserved_from_primitive, primitive_from_conserved

CONFORMAL = EquationOfState(
    kind="conformal",
    speed_of_sound2=1.0 / 3.0,
    low_speed_of_sound2=1.0 / 3.0,
    high_speed_of_sound2=1.0 / 3.0,
    transition_energy=1.0,
    transition_width=1.0,
)
FLOOR = 1.0e-12


def integrate_source(state, tau, tau_end, steps):
    dt = (tau_end - tau) / steps
    for step in range(steps):
        state = apply_bjorken_cooling(state, tau + step * dt, dt, CONFORMAL, FLOOR)
    return state


def test_bjorken_cooling_at_rest_matches_analytic_law():
    energy = torch.full((4, 4), 10.0, dtype=torch.float64)
    zero = torch.zeros_like(energy)
    state = integrate_source(conserved_from_primitive(energy, zero, zero, CONFORMAL), 0.6, 1.2, 20)
    final_energy, *_ = primitive_from_conserved(state[0], state[1], state[2], CONFORMAL, FLOOR)
    assert float(final_energy[0, 0]) == pytest.approx(10.0 * (0.6 / 1.2) ** (4.0 / 3.0), rel=1.0e-8)


@pytest.mark.parametrize("velocity", [0.2, 0.4, 0.8])
def test_bjorken_source_conserves_tau_times_transverse_momentum(velocity):
    # Milne equations for a uniform moving fluid: d(T^{tau x})/dtau = -T^{tau x}/tau exactly.
    energy = torch.full((4, 4), 10.0, dtype=torch.float64)
    state = conserved_from_primitive(energy, torch.full_like(energy, velocity), torch.zeros_like(energy), CONFORMAL)
    final = integrate_source(state, 0.6, 1.2, 15)
    assert 1.2 * float(final[1, 0, 0]) == pytest.approx(0.6 * float(state[1, 0, 0]), rel=1.0e-12)


@pytest.mark.parametrize("velocity", [0.2, 0.6])
def test_bjorken_source_energy_equation_matches_fine_integration(velocity):
    # d(T^tt)/dtau = -(T^tt + p)/tau, checked against a 100x finer integration of the same ODE.
    energy = torch.full((2, 2), 5.0, dtype=torch.float64)
    state = conserved_from_primitive(energy, torch.full_like(energy, velocity), torch.zeros_like(energy), CONFORMAL)
    coarse = integrate_source(state, 0.6, 3.0, 4)
    fine = integrate_source(state, 0.6, 3.0, 400)
    assert float(coarse[0, 0, 0]) == pytest.approx(float(fine[0, 0, 0]), rel=1.0e-7)


def test_gubser_solution_satisfies_milne_equations():
    """Independent check of the analytic formula with centered finite differences."""
    h = 1.0e-4
    points = [(1.0, 0.7, 0.3), (1.5, 1.2, -0.8), (2.0, 0.1, 2.5)]

    def tensor(tau, x, y):
        e, vx, vy = gubser_ideal_solution(tau, torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), 1.0)
        p = e / 3.0
        w = (e + p) / (1.0 - vx * vx - vy * vy)
        return [w - p, w * vx, w * vy, w * vx * vx + p, w * vx * vy, w * vy * vy + p, p]

    for tau, x, y in points:
        def derivative(component, axis):
            plus = [tau, x, y]
            minus = [tau, x, y]
            plus[axis] += h
            minus[axis] -= h
            return float((tensor(*plus)[component] - tensor(*minus)[component]) / (2.0 * h))

        t = [float(v) for v in tensor(tau, x, y)]
        residuals = (
            derivative(0, 0) + derivative(1, 1) + derivative(2, 2) + (t[0] + t[6]) / tau,
            derivative(1, 0) + derivative(3, 1) + derivative(4, 2) + t[1] / tau,
            derivative(2, 0) + derivative(4, 1) + derivative(5, 2) + t[2] / tau,
        )
        for residual in residuals:
            assert abs(residual) < 1.0e-6 * max(1.0, t[0])


def test_gubser_flow_converges_at_first_order():
    kwargs = dict(extent=8.0, cfl=0.25, tau0=1.0, tau_end=2.0, q=1.0, compare_radius=3.0, device=torch.device("cpu"))
    coarse_energy, coarse_velocity = gubser_flow_error(80, **kwargs)
    fine_energy, fine_velocity = gubser_flow_error(160, **kwargs)
    # dx shrinks by (2*8/79)/(2*8/159) ~ 2.01 between the two grids.
    ratio = (16.0 / 79.0) / (16.0 / 159.0)
    assert math.log(coarse_energy / fine_energy) / math.log(ratio) > 0.8
    assert math.log(coarse_velocity / fine_velocity) / math.log(ratio) > 0.8
    assert fine_energy < 0.18 and fine_velocity < 0.03
