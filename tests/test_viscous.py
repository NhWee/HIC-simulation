"""Second-order viscous hydrodynamics (ROADMAP P2-1) against independent solutions."""
import numpy as np
import pytest
import torch

import hic.viscous_hydro as vh
from hic.eos import EquationOfState

HBARC = 0.1973269804
CONFORMAL = EquationOfState(kind="conformal", speed_of_sound2=1 / 3, low_speed_of_sound2=1 / 3,
                            high_speed_of_sound2=1 / 3, transition_energy=1.0, transition_width=1.0)
A_SB = 10.0 / 0.4**4  # e = a T^4, so T(e = 10 GeV/fm^3) = 0.4 GeV


def temperature_of(energy):
    return (torch.clamp(energy, min=1.0e-12) / A_SB) ** 0.25


def bjorken_ode(e0, phi0, tau0, tau1, eta_over_s, steps=20000):
    """MIS Bjorken flow written independently in physical units (phi = -tau^2 pi^etaeta)."""
    h = (tau1 - tau0) / steps

    def rate(t, y):
        e, phi = y
        eta_over_w = eta_over_s * HBARC / (e / A_SB) ** 0.25  # eta / (e + p) in fm
        phi_ns = (4.0 / 3.0) * eta_over_w * (4.0 / 3.0) * e / t
        return np.array((-(4.0 / 3.0 * e - phi) / t, -(phi - phi_ns) / (5.0 * eta_over_w) - (4.0 / 3.0) * phi / t))

    y, t = np.array((e0, phi0)), tau0
    for _ in range(steps):
        k1 = rate(t, y)
        k2 = rate(t + h / 2, y + h / 2 * k1)
        k3 = rate(t + h / 2, y + h / 2 * k2)
        k4 = rate(t + h, y + h * k3)
        y, t = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + h
    return y


@pytest.mark.parametrize(("eta_over_s", "initial"), [(0.08, "ns"), (0.2, "ns"), (0.2, "zero")])
def test_viscous_bjorken_flow_matches_mis_ode(eta_over_s, initial):
    params = vh.ViscousParameters(eta_over_s=eta_over_s)
    energy = torch.full((4, 4), 10.0, dtype=torch.float64)
    zero = torch.zeros_like(energy)
    tau0, tau1, dt = 0.6, 3.0, 0.005
    state = vh.initial_state(energy, zero, zero, CONFORMAL, params, temperature_of, tau0, 0.2, initial_shear=initial)
    phi0 = float(state.shear[0, 0, 0] + state.shear[2, 0, 0])
    tau = tau0
    for _ in range(round((tau1 - tau0) / dt)):
        state = vh.viscous_step(state, tau, dt, 0.2, CONFORMAL, params, temperature_of)
        tau += dt
    prim = vh.recover_primitive(state, CONFORMAL, 1.0e-12)
    e_exact, phi_exact = bjorken_ode(10.0, phi0, tau0, tau1, eta_over_s)
    assert float(prim["e"][2, 2]) == pytest.approx(e_exact, rel=2.0e-3)  # first order in dt (operator splitting)
    assert float(state.shear[0, 2, 2] + state.shear[2, 2, 2]) == pytest.approx(phi_exact, rel=1.0e-2)
    # Viscous heating: noticeably hotter than ideal Bjorken cooling.
    assert float(prim["e"][2, 2]) > 1.1 * 10.0 * (tau0 / tau1) ** (4.0 / 3.0)


def linear_mis_solution(x, profile, t, eta_over_w, tau_pi, kind):
    """Exact linearized MIS evolution of a static conformal fluid, mode by mode in Fourier space."""
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    amplitudes = np.fft.fft(profile)
    out = np.zeros(k.size, dtype=complex)
    for i, kk in enumerate(k):
        if kind == "shear":
            # y = (u, pi^xy/(e+p)):  u' = -i k q,  tau_pi q' = -q - i k eta/(e+p) u   (pi_NS = -eta d_x u)
            matrix = np.array([[0.0, -1j * kk], [-1j * kk * eta_over_w / tau_pi, -1.0 / tau_pi]])
            y0 = np.array([1.0, -1j * kk * eta_over_w])
        else:
            # y = (de/(e+p), u, pi^xx/(e+p)):  de' = -i k u,  u' = -i k (cs2 de + q),
            # tau_pi q' = -q + 2 eta/(e+p) sigma^xx with sigma^xx = -(2/3) d_x u.
            cs2 = 1.0 / 3.0
            matrix = np.array([[0.0, -1j * kk, 0.0],
                               [-1j * kk * cs2, 0.0, -1j * kk],
                               [0.0, -(4.0 / 3.0) * 1j * kk * eta_over_w / tau_pi, -1.0 / tau_pi]])
            y0 = np.array([0.0, 1.0, -(4.0 / 3.0) * 1j * kk * eta_over_w])
        values, vectors = np.linalg.eig(matrix)
        coefficients = np.linalg.solve(vectors, y0)
        y_t = vectors @ (coefficients * np.exp(values * t))
        out[i] = amplitudes[i] * (y_t[0] if kind == "shear" else y_t[1])
    return np.fft.ifft(out).real


@pytest.mark.parametrize("kind", ["shear", "sound"])
def test_linear_waves_match_exact_mis_solution(kind):
    eta_over_s = 0.2
    nx, extent, dt, t_end, width, amplitude = 241, 12.0, 0.02, 3.0, 1.0, 0.01
    x = np.linspace(-extent, extent, nx)
    dx = x[1] - x[0]
    profile = amplitude * np.exp(-x**2 / (2 * width**2))
    energy = torch.full((5, nx), 10.0, dtype=torch.float64)
    u = torch.as_tensor(np.tile(profile, (5, 1)))
    vx, vy = (u, torch.zeros_like(u)) if kind == "sound" else (torch.zeros_like(u), u)
    params = vh.ViscousParameters(eta_over_s=eta_over_s, bjorken=False)
    state = vh.initial_state(energy, vx, vy, CONFORMAL, params, temperature_of, 1.0, dx, initial_shear="ns")
    t = 0.0
    for _ in range(round(t_end / dt)):
        state = vh.viscous_step(state, 1.0 + t, dt, dx, CONFORMAL, params, temperature_of)
        t += dt
    prim = vh.recover_primitive(state, CONFORMAL, 1.0e-12)
    numeric = (prim["vx"] if kind == "sound" else prim["vy"])[2].numpy()
    eta_over_w = eta_over_s * HBARC / 0.4
    exact = linear_mis_solution(x, profile, t_end, eta_over_w, 5.0 * eta_over_w, kind)
    ideal = linear_mis_solution(x, profile, t_end, 1.0e-9, 5.0e-9, kind)
    error = np.max(np.abs(numeric - exact)) / amplitude
    viscous_effect = np.max(np.abs(exact - ideal)) / amplitude
    assert viscous_effect > 0.04  # the test is sensitive to viscosity
    assert error < 0.1 * viscous_effect, (error, viscous_effect)


def gubser_is_ode(eta_over_s, rho0=-6.0, rho1=3.0, steps=36000, t_hat0=0.035):
    """Conformal Israel-Stewart Gubser flow (Marrochio et al., PRC 91, 014903 (2015)) in de Sitter time rho:
    dT/drho = T tanh(rho) (pibar/3 - 2/3),  tau_pi [dpibar/drho + (4/3) pibar^2 tanh rho] + pibar = (4/3)(eta/s)/T tanh rho,
    with tau_pi = 5 (eta/s)/T and pibar = -pi^varsigma_varsigma/(e+p). Any solution is an exact solution."""
    rho = np.linspace(rho0, rho1, steps + 1)
    h = rho[1] - rho[0]

    def rate(r, y):
        t_hat, pibar = y
        th = np.tanh(r)
        return np.array((t_hat * th * (pibar / 3.0 - 2.0 / 3.0),
                         ((4.0 / 3.0) * eta_over_s / t_hat * th - pibar) / (5.0 * eta_over_s / t_hat) - (4.0 / 3.0) * pibar**2 * th))

    y = np.array((t_hat0, (4.0 / 3.0) * eta_over_s / t_hat0 * np.tanh(rho0)))
    out = [y]
    for r in rho[:-1]:
        k1 = rate(r, y)
        k2 = rate(r + h / 2, y + h / 2 * k1)
        k3 = rate(r + h / 2, y + h / 2 * k2)
        k4 = rate(r + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y)
    return rho, np.array(out)


def gubser_is_fields(tau, xx, yy, rho, solution, q=1.0):
    """e, v and pi^{xx, xy, yy} of the Gubser IS solution in Milne coordinates.

    In the local rest frame the shear tensor is diag(-pibar w/2, -pibar w/2, pibar w) along (radial, azimuthal,
    eta), with tau^2 pi^etaeta = pibar w; boosting the radial leg gives pi^ij = -(pibar w/2)(cosh^2 kappa r^i r^j + phi^i phi^j).
    """
    r2 = xx**2 + yy**2
    r = np.sqrt(r2)
    rho_here = np.arcsinh(-(1.0 - q * q * tau * tau + q * q * r2) / (2.0 * q * tau))
    t_hat = np.interp(rho_here, rho, solution[:, 0])
    pibar = np.interp(rho_here, rho, solution[:, 1])
    energy = A_SB * (t_hat / tau * HBARC) ** 4
    vr = 2.0 * q * q * tau * r / (1.0 + q * q * tau * tau + q * q * r2)
    cosh2 = 1.0 / (1.0 - vr**2)
    safe = np.where(r > 0, r, 1.0)
    cx, cy = np.where(r > 0, xx / safe, 1.0), np.where(r > 0, yy / safe, 0.0)
    lam = -pibar * (4.0 / 3.0) * energy / 2.0
    shear = np.stack((lam * (cosh2 * cx * cx + cy * cy), lam * (cosh2 - 1.0) * cx * cy, lam * (cosh2 * cy * cy + cx * cx)))
    return energy, vr * cx, vr * cy, shear


def gubser_viscous_errors(n, eta_over_s=0.2, tau0=1.0, tau1=2.0, extent=5.0, radius=2.0):
    rho, solution = gubser_is_ode(eta_over_s)
    x = np.linspace(-extent, extent, n)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x)
    params = vh.ViscousParameters(eta_over_s=eta_over_s, regulator=10.0)
    energy, vx, vy, shear = (torch.as_tensor(a) for a in gubser_is_fields(tau0, xx, yy, rho, solution))
    state = vh.initial_state(energy, vx, vy, CONFORMAL, params, temperature_of, tau0, dx, initial_shear="zero")
    state.shear = shear.clone()
    gamma = 1.0 / torch.sqrt(1.0 - vx**2 - vy**2)
    state.conserved = vh.total_tensor(energy, gamma * vx, gamma * vy, state.shear, state.bulk, CONFORMAL)[0]
    back = 1.0e-3  # the exact flow just before tau0, for d_tau u in sigma^{mu nu}
    _, vx_b, vy_b, _ = (torch.as_tensor(a) for a in gubser_is_fields(tau0 - back, xx, yy, rho, solution))
    gamma_b = 1.0 / torch.sqrt(1.0 - vx_b**2 - vy_b**2)
    state.previous_u, state.previous_tau = torch.stack((gamma_b, gamma_b * vx_b, gamma_b * vy_b)), tau0 - back
    tau, dt = tau0, 0.25 * dx
    while tau < tau1 - 1.0e-9:
        h = min(dt, tau1 - tau)
        state = vh.viscous_step(state, tau, h, dx, CONFORMAL, params, temperature_of)
        tau += h
    prim = vh.recover_primitive(state, CONFORMAL, 1.0e-12)
    energy1, vx1, vy1, shear1 = gubser_is_fields(tau1, xx, yy, rho, solution)
    inside = xx**2 + yy**2 <= radius**2
    enthalpy = (4.0 / 3.0) * np.max(energy1[inside])
    return (np.max(np.abs(prim["e"].numpy() - energy1)[inside]) / np.max(energy1[inside]),
            np.max(np.hypot(prim["vx"].numpy() - vx1, prim["vy"].numpy() - vy1)[inside]),
            np.max(np.abs(state.shear.numpy() - shear1)[:, inside]) / enthalpy,
            np.max(np.abs(shear1)[:, inside]) / enthalpy)


def test_viscous_gubser_flow_converges_to_israel_stewart_solution():
    """Radial + longitudinal expansion with large flow: tests the projector (u pi a) terms, sigma with
    acceleration and the primitive inversion. Without the projector terms the shear error is ~7x larger."""
    coarse = gubser_viscous_errors(81)
    fine = gubser_viscous_errors(121)
    assert fine[3] > 0.05  # the shear stress is a sizeable fraction of e + p
    assert fine[0] < 0.015 and fine[1] < 0.002, fine
    assert fine[2] < 0.15 * fine[3], fine
    assert fine[0] < 0.7 * coarse[0] and fine[2] < 0.8 * coarse[2], (coarse, fine)


def test_shear_delta_f_is_traceless_and_orthogonal_in_moving_fluid():
    """Averaged over local-rest-frame directions at fixed energy, p^mu p^nu pi_mu_nu vanishes for a
    traceless pi orthogonal to u; this checks pi^{tau mu} and tau^2 pi^etaeta built from (pi^xx, pi^xy, pi^yy),
    the boost to the cell's eta_s and the momentum layout (z, x, y) used by the sampler."""
    from types import SimpleNamespace

    import hic.blender_event_pipeline as bep

    rng = np.random.default_rng(3)
    count = 400000
    vx, vy, eta = 0.55, -0.3, 0.7
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    rest_energy = np.full(count, 0.8)  # massless, LRF energy 0.8 GeV
    rest_momentum = rest_energy[:, None] * direction
    cosh = np.cosh(eta)
    beta = np.tile((np.tanh(eta), vx / cosh, vy / cosh), (count, 1))
    lab_energy, lab_momentum = bep.lorentz_boost(rest_energy, rest_momentum, beta)
    one = np.ones(1)
    surface = SimpleNamespace(velocity_x=vx * one, velocity_y=vy * one,
                              shear_xx=0.006 * one, shear_xy=-0.004 * one, shear_yy=0.01 * one)
    args = SimpleNamespace(freezeout_temperature=0.156)
    factor = bep.shear_delta_f_factor(args, surface, np.zeros(count, dtype=int), np.full(count, eta),
                                      lab_energy, lab_momentum)
    delta = factor - 1.0
    assert np.all(factor > 0.0)  # small stresses: no clipping, so the linear identity applies exactly
    assert np.std(delta) > 0.05  # the correction is not trivially zero
    assert abs(np.mean(delta)) < 5.0 * np.std(delta) / np.sqrt(count)


def test_viscous_event_produces_entropy_and_passes_audit():
    import hic.blender_event_pipeline as bep

    common = ["--cpu", "--grid-size", "64", "--extent", "12", "--particles", "1", "--candidate-pool", "20000",
              "--impact-parameter", "6.0", "--seed", "11"]
    ideal = bep.simulate(bep.parse_args(common), verbose=False).audit
    viscous = bep.simulate(bep.parse_args(common + ["--eta-over-s", "0.12"]), verbose=False).audit
    assert ideal["passed"] and viscous["passed"]
    assert abs(ideal["entropy_flux_ratio"] - 1.0) < 0.05
    assert 1.03 < viscous["entropy_flux_ratio"] < 1.5  # shear viscosity produces entropy
    assert viscous["dnch_deta"] > ideal["dnch_deta"]
    assert viscous["eta_over_s"] == 0.12
