"""Second-order (Israel-Stewart / 14-moment type) viscous hydrodynamics in 2+1D, boost invariant.

Conserved variables are the full T^{tau mu} = (e + p + Pi) u^tau u^mu - (p + Pi) g^{tau mu} + pi^{tau mu}.
The shear stress is carried by its transverse components pi^xx, pi^xy, pi^yy; the others follow from
orthogonality (u_mu pi^{mu nu} = 0) and tracelessness:

    pi^{tau i} = v_x pi^{x i} + v_y pi^{y i},   pi^{tau tau} = v_x pi^{tau x} + v_y pi^{tau y},
    tau^2 pi^{eta eta} = pi^{tau tau} - pi^xx - pi^yy.

Relaxation equations (D = u^mu d_mu, theta = d_mu u^mu with the Milne term u^tau / tau):

    D pi^ij = -(pi^ij - 2 eta sigma^ij) / tau_pi - (4/3) pi^ij theta,    tau_pi = c_pi eta / (e + p),
    D Pi    = -(Pi + zeta theta) / tau_Pi  - (2/3) Pi theta,           tau_Pi = zeta / (15 (1/3 - cs2)^2 (e + p)),

with sigma^ij = -(d_i u^j + d_j u^i)/2 - (u^i a^j + u^j a^i)/2 + (delta^ij + u^i u^j) theta / 3 and
a^j = D u^j. Other second-order terms (vorticity couplings, phi_7 pi pi, lambda_piPi ...) are omitted;
the relaxation is integrated exponentially, so tau_pi << dt is stable. Transport of T^{tau mu} uses the
Kurganov-Tadmor scheme of hic.ideal_hydro_2d with the viscous stresses reconstructed alongside e and u;
the viscous fields are advected with the same reconstruction. A regulator caps the stresses relative
to e + p in dilute regions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from hic.eos import EquationOfState
from hic.ideal_hydro_2d import _interface_states

HBARC = 0.1973269804  # GeV fm: eta = (eta/s) s hbar c in GeV/fm^2 when s is in 1/fm^3


@dataclass
class ViscousParameters:
    eta_over_s: float = 0.0
    zeta_over_s_max: float = 0.0
    zeta_peak_temperature: float = 0.165  # GeV
    zeta_width: float = 0.03  # GeV
    shear_relaxation: float = 5.0  # c_pi in tau_pi = c_pi eta / (e + p)
    regulator: float = 1.0  # cap sqrt(pi.pi + 3 Pi^2) <= regulator * (e + p)
    bjorken: bool = True  # include the Milne (longitudinal expansion) terms
    # delta_pipi / tau_pi in the -delta_pipi pi theta term (4/3: DNMR / conformal "full" IS;
    # 0: the simplified IS form of Song & Heinz, PRC 77, 064901 (2008), Eq. 5).
    shear_theta_coefficient: float = 4.0 / 3.0

    @property
    def active(self) -> bool:
        return self.eta_over_s > 0.0 or self.zeta_over_s_max > 0.0


@dataclass
class ViscousState:
    conserved: torch.Tensor  # (3, ny, nx): T^{tau tau}, T^{tau x}, T^{tau y}
    shear: torch.Tensor  # (3, ny, nx): pi^xx, pi^xy, pi^yy
    bulk: torch.Tensor  # (ny, nx)
    previous_u: torch.Tensor | None = None  # (3, ny, nx): u^tau, u^x, u^y at the previous step
    previous_tau: float | None = None
    primitive: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Kinematics of the viscous stresses
# ---------------------------------------------------------------------------------------------

def shear_time_components(shear: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor):
    """pi^{tau x}, pi^{tau y}, pi^{tau tau}, tau^2 pi^{eta eta} from the transverse components."""
    pxx, pxy, pyy = shear
    ptx = vx * pxx + vy * pxy
    pty = vx * pxy + vy * pyy
    ptt = vx * ptx + vy * pty
    return ptx, pty, ptt, ptt - pxx - pyy


def total_tensor(energy, ux, uy, shear, bulk, eos: EquationOfState):
    """T^{tau mu} and the x, y fluxes (T^{x mu}, T^{y mu}) of the full viscous fluid."""
    gamma = torch.sqrt(1.0 + ux * ux + uy * uy)
    vx, vy = ux / gamma, uy / gamma
    pressure = eos.pressure(energy) + bulk
    enthalpy = energy + pressure
    ptx, pty, ptt, _ = shear_time_components(shear, vx, vy)
    pxx, pxy, pyy = shear
    w = enthalpy * gamma * gamma
    t_tt = w - pressure + ptt
    t_tx = w * vx + ptx
    t_ty = w * vy + pty
    t_xx = w * vx * vx + pressure + pxx
    t_xy = w * vx * vy + pxy
    t_yy = w * vy * vy + pressure + pyy
    conserved = torch.stack((t_tt, t_tx, t_ty))
    flux_x = torch.stack((t_tx, t_xx, t_xy))
    flux_y = torch.stack((t_ty, t_xy, t_yy))
    return conserved, flux_x, flux_y, vx, vy


def recover_primitive(state: ViscousState, eos: EquationOfState, energy_floor: float, iterations: int = 24):
    """e, v from T^{tau mu} with the viscous parts removed (fixed point in v: pi^{tau mu} depends on v).

    Warm-started from the previous primitive state (e and v). The damped iteration must be run to
    convergence: an unconverged inversion is a first-order error that does not vanish with dx
    (it made the Gubser energy density 4x less accurate than the ideal solver).
    """
    t_tt, t_tx, t_ty = state.conserved
    bulk = state.bulk
    vx = state.primitive.get("vx", torch.zeros_like(t_tt))
    vy = state.primitive.get("vy", torch.zeros_like(t_tt))
    energy = state.primitive.get("e")
    if energy is None or energy.shape != t_tt.shape:
        energy = torch.clamp(t_tt, min=energy_floor)
    for _ in range(iterations):
        ptx, pty, ptt, _ = shear_time_components(state.shear, vx, vy)
        ideal_e = t_tt - ptt
        mx, my = t_tx - ptx, t_ty - pty
        pressure = eos.pressure(energy) + bulk
        denominator = torch.clamp(ideal_e + pressure, min=energy_floor)
        new_vx, new_vy = mx / denominator, my / denominator
        speed = torch.sqrt(new_vx * new_vx + new_vy * new_vy)
        limit = torch.clamp(0.999 / torch.clamp(speed, min=1.0e-12), max=1.0)
        new_vx, new_vy = new_vx * limit, new_vy * limit
        vx, vy = 0.5 * (vx + new_vx), 0.5 * (vy + new_vy)
        energy = torch.clamp(ideal_e - (mx * vx + my * vy), min=energy_floor)
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - vx * vx - vy * vy, min=1.0e-12))
    state.primitive = {"e": energy, "vx": vx, "vy": vy, "ut": gamma, "ux": gamma * vx, "uy": gamma * vy}
    return state.primitive


# ---------------------------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------------------------

def _kt_rhs(state: ViscousState, dx: float, eos: EquationOfState, energy_floor: float, theta: float = 1.5):
    prim = recover_primitive(state, eos, energy_floor)
    fields = torch.stack((prim["e"], prim["ux"], prim["uy"], state.shear[0], state.shear[1], state.shear[2], state.bulk))
    padded = torch.nn.functional.pad(fields.unsqueeze(0), (2, 2, 2, 2), mode="replicate").squeeze(0)
    rhs_conserved = torch.zeros_like(state.conserved)
    rhs_shear = torch.zeros_like(state.shear)
    rhs_bulk = torch.zeros_like(state.bulk)
    for axis in (1, 2):
        dim = 2 if axis == 1 else 1
        q = padded[:, 2:-2, :] if axis == 1 else padded[:, :, 2:-2]
        left, right = _interface_states(q, dim, theta)
        left = torch.cat((torch.clamp(left[:1], min=energy_floor), left[1:]))
        right = torch.cat((torch.clamp(right[:1], min=energy_floor), right[1:]))
        sides = []
        for s in (left, right):
            conserved, flux_x, flux_y, vx, vy = total_tensor(s[0], s[1], s[2], s[3:6], s[6], eos)
            v_axis = vx if axis == 1 else vy
            sound = torch.sqrt(torch.clamp(eos.sound_speed2(s[0]), 0.0, 1.0))
            speed = (v_axis.abs() + sound) / (1.0 + v_axis.abs() * sound)
            visc = s[3:7]
            sides.append((conserved, flux_x if axis == 1 else flux_y, speed, v_axis, visc))
        (cl, fl, sl, vl, ql), (cr, fr, sr, vr, qr) = sides
        a = torch.maximum(sl, sr).unsqueeze(0)
        interface = 0.5 * (fl + fr) - 0.5 * a * (cr - cl)
        # Viscous fields are advected as densities u^tau q: flux u^k q (upwinded with the same speed).
        visc_interface = 0.5 * (ql * vl + qr * vr) - 0.5 * a * (qr - ql)
        n = state.conserved.shape[dim]
        rhs_conserved -= (interface.narrow(dim, 1, n) - interface.narrow(dim, 0, n)) / dx
        div = (visc_interface.narrow(dim, 1, n) - visc_interface.narrow(dim, 0, n)) / dx
        # d_t q + v.grad q = (d_t q + div(v q)) - q div v: add back q div v below via the velocity divergence.
        rhs_shear -= div[:3]
        rhs_bulk -= div[3]
    vx, vy = prim["vx"], prim["vy"]
    div_v = _central_x(vx, dx) + _central_y(vy, dx)
    rhs_shear += state.shear * div_v
    rhs_bulk += state.bulk * div_v
    return rhs_conserved, rhs_shear, rhs_bulk


def _central_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    padded = torch.nn.functional.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0), mode="replicate")[0, 0]
    return (padded[:, 2:] - padded[:, :-2]) / (2.0 * dx)


def _central_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    padded = torch.nn.functional.pad(field.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode="replicate")[0, 0]
    return (padded[2:, :] - padded[:-2, :]) / (2.0 * dx)


def _transport(state: ViscousState, dx: float, dt: float, eos: EquationOfState, energy_floor: float) -> None:
    """SSP-RK2 for T^{tau mu} and the advection of the viscous fields (sources handled separately)."""
    c0, s0, b0 = state.conserved, state.shear, state.bulk
    rc, rs, rb = _kt_rhs(state, dx, eos, energy_floor)
    state.conserved, state.shear, state.bulk = c0 + dt * rc, s0 + dt * rs, b0 + dt * rb
    state.conserved = torch.stack((torch.clamp(state.conserved[0], min=energy_floor), state.conserved[1], state.conserved[2]))
    rc, rs, rb = _kt_rhs(state, dx, eos, energy_floor)
    state.conserved = 0.5 * c0 + 0.5 * (state.conserved + dt * rc)
    state.conserved = torch.stack((torch.clamp(state.conserved[0], min=energy_floor), state.conserved[1], state.conserved[2]))
    state.shear = 0.5 * s0 + 0.5 * (state.shear + dt * rs)
    state.bulk = 0.5 * b0 + 0.5 * (state.bulk + dt * rb)


def _milne_source(state: ViscousState, tau: float, dt: float, eos: EquationOfState, energy_floor: float, substeps: int = 4):
    """d(T^tt)/dtau = -(T^tt + p + Pi + tau^2 pi^etaeta)/tau, d(T^ti)/dtau = -T^ti/tau (RK2 substeps)."""
    h = dt / substeps
    for k in range(substeps):
        t = tau + k * h
        def rate(conserved, at):
            state.conserved = conserved
            prim = recover_primitive(state, eos, energy_floor, iterations=12)
            _, _, _, pietaeta = shear_time_components(state.shear, prim["vx"], prim["vy"])
            pressure = eos.pressure(prim["e"]) + state.bulk
            return torch.stack((-(conserved[0] + pressure + pietaeta) / at, -conserved[1] / at, -conserved[2] / at))
        c0 = state.conserved
        k1 = rate(c0, t)
        k2 = rate(c0 + h * k1, t + h)
        state.conserved = c0 + 0.5 * h * (k1 + k2)
        state.conserved = torch.stack((torch.clamp(state.conserved[0], min=energy_floor), state.conserved[1], state.conserved[2]))


# ---------------------------------------------------------------------------------------------
# Relaxation of the viscous stresses
# ---------------------------------------------------------------------------------------------

def zeta_over_s(params: ViscousParameters, temperature: torch.Tensor) -> torch.Tensor:
    if params.zeta_over_s_max <= 0.0:
        return torch.zeros_like(temperature)
    x = (temperature - params.zeta_peak_temperature) / params.zeta_width
    return params.zeta_over_s_max / (1.0 + x * x)


def gradients_and_shear_rate(prim: dict, previous_u: torch.Tensor | None, dtau: float | None, tau: float, dx: float, bjorken: bool):
    """theta, sigma^{xx, xy, yy} and the acceleration from the current and previous flow."""
    ut, ux, uy = prim["ut"], prim["ux"], prim["uy"]
    if previous_u is None or not dtau:
        dt_ut = dt_ux = dt_uy = torch.zeros_like(ut)
    else:
        dt_ut = (ut - previous_u[0]) / dtau
        dt_ux = (ux - previous_u[1]) / dtau
        dt_uy = (uy - previous_u[2]) / dtau
    dx_ux, dy_ux = _central_x(ux, dx), _central_y(ux, dx)
    dx_uy, dy_uy = _central_x(uy, dx), _central_y(uy, dx)
    theta = dt_ut + dx_ux + dy_uy + (ut / tau if bjorken else 0.0)
    ax = ut * dt_ux + ux * dx_ux + uy * dy_ux
    ay = ut * dt_uy + ux * dx_uy + uy * dy_uy
    third = theta / 3.0
    sigma_xx = -dx_ux - ux * ax + (1.0 + ux * ux) * third
    sigma_yy = -dy_uy - uy * ay + (1.0 + uy * uy) * third
    sigma_xy = -0.5 * (dx_uy + dy_ux) - 0.5 * (ux * ay + uy * ax) + ux * uy * third
    return theta, torch.stack((sigma_xx, sigma_xy, sigma_yy)), torch.stack((ax, ay))


def projector_terms(shear: torch.Tensor, prim: dict, acceleration: torch.Tensor) -> torch.Tensor:
    """-(u^i pi^{j lambda} + u^j pi^{i lambda}) a_lambda: keeps u_mu pi^{mu nu} = 0 along the flow.

    With a^tau = (u^x a^x + u^y a^y)/u^tau (u.a = 0) and metric (+,-,-,-),
    pi^{j lambda} a_lambda = pi^{tau j} a^tau - pi^{jx} a^x - pi^{jy} a^y (a^eta = 0).
    """
    ux, uy, ut = prim["ux"], prim["uy"], prim["ut"]
    ax, ay = acceleration
    at = (ux * ax + uy * ay) / ut
    pxx, pxy, pyy = shear
    ptx, pty, _, _ = shear_time_components(shear, prim["vx"], prim["vy"])
    cx = ptx * at - pxx * ax - pxy * ay
    cy = pty * at - pxy * ax - pyy * ay
    return -torch.stack((2.0 * ux * cx, ux * cy + uy * cx, 2.0 * uy * cy))


def _relax(state: ViscousState, params: ViscousParameters, eos: EquationOfState, temperature_of, tau: float, dt: float,
           dx: float, energy_floor: float) -> None:
    prim = state.primitive
    energy = prim["e"]
    temperature = torch.clamp(temperature_of(energy), min=1.0e-6)
    pressure = eos.pressure(energy)
    enthalpy = torch.clamp(energy + pressure, min=energy_floor)
    entropy = enthalpy / temperature
    eta = params.eta_over_s * entropy * HBARC
    zeta = zeta_over_s(params, temperature) * entropy * HBARC
    dtau = (tau - state.previous_tau) if state.previous_tau is not None else None
    theta, sigma, acceleration = gradients_and_shear_rate(prim, state.previous_u, dtau, tau, dx, params.bjorken)
    ut = prim["ut"]
    # Local proper-time step of the fluid element: d tau_fluid = dt / u^tau.
    step = dt / ut
    if params.eta_over_s > 0.0:
        tau_pi = params.shear_relaxation * eta / enthalpy
        target = 2.0 * eta * sigma
        decay = torch.exp(-step / torch.clamp(tau_pi, min=1.0e-12))
        relaxed = target + (state.shear - target) * decay
        state.shear = relaxed * torch.exp(-params.shear_theta_coefficient * theta * step)
        state.shear = state.shear + step * projector_terms(state.shear, prim, acceleration)
    else:
        state.shear = torch.zeros_like(state.shear)
    if params.zeta_over_s_max > 0.0:
        cs2 = torch.clamp(eos.sound_speed2(energy), 0.0, 1.0 / 3.0 - 1.0e-4)
        tau_bulk = zeta / (15.0 * (1.0 / 3.0 - cs2) ** 2 * enthalpy)
        target = -zeta * theta
        decay = torch.exp(-step / torch.clamp(tau_bulk, min=1.0e-12))
        state.bulk = (target + (state.bulk - target) * decay) * torch.exp(-(2.0 / 3.0) * theta * step)
    else:
        state.bulk = torch.zeros_like(state.bulk)
    # Regulator: keep the stresses below regulator * (e + p) (dilute edges, vacuum).
    pxx, pxy, pyy = state.shear
    ptx, pty, ptt, pee = shear_time_components(state.shear, prim["vx"], prim["vy"])
    norm = torch.sqrt(torch.clamp(pxx * pxx + 2 * pxy * pxy + pyy * pyy + pee * pee + ptt * ptt + 3.0 * state.bulk ** 2, min=0.0))
    scale = torch.clamp(params.regulator * enthalpy / torch.clamp(norm, min=1.0e-30), max=1.0)
    state.shear = state.shear * scale
    state.bulk = state.bulk * scale
    # No cavitation: the bulk pressure may not make the effective pressure p + Pi negative.
    state.bulk = torch.maximum(state.bulk, -pressure)


def viscous_step(
    state: ViscousState,
    tau: float,
    dt: float,
    dx: float,
    eos: EquationOfState,
    params: ViscousParameters,
    temperature_of,
    energy_floor: float = 1.0e-8,
) -> ViscousState:
    """tau -> tau + dt: Strang-split Milne source, KT transport, relaxation of pi and Pi."""
    if params.bjorken:
        _milne_source(state, tau, 0.5 * dt, eos, energy_floor)
    _transport(state, dx, dt, eos, energy_floor)
    if params.bjorken:
        _milne_source(state, tau + 0.5 * dt, 0.5 * dt, eos, energy_floor)
    prim = recover_primitive(state, eos, energy_floor)
    _relax(state, params, eos, temperature_of, tau + dt, dt, dx, energy_floor)
    state.previous_u = torch.stack((prim["ut"], prim["ux"], prim["uy"]))
    state.previous_tau = tau + dt
    return state


def initial_state(
    energy: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    eos: EquationOfState,
    params: ViscousParameters,
    temperature_of,
    tau0: float,
    dx: float,
    initial_shear: str = "ns",
) -> ViscousState:
    """Ideal T^{tau mu} plus Navier-Stokes (or zero) stresses at tau0."""
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - vx * vx - vy * vy, min=1.0e-12))
    zero = torch.zeros_like(energy)
    shear = torch.zeros((3,) + energy.shape, dtype=energy.dtype, device=energy.device)
    state = ViscousState(conserved=torch.zeros((3,) + energy.shape, dtype=energy.dtype, device=energy.device),
                         shear=shear, bulk=zero.clone())
    state.primitive = {"e": energy, "vx": vx, "vy": vy, "ut": gamma, "ux": gamma * vx, "uy": gamma * vy}
    if initial_shear == "ns" and params.active:
        temperature = torch.clamp(temperature_of(energy), min=1.0e-6)
        entropy = (energy + eos.pressure(energy)) / temperature
        theta, sigma, _ = gradients_and_shear_rate(state.primitive, None, None, tau0, dx, params.bjorken)
        state.shear = 2.0 * params.eta_over_s * entropy * HBARC * sigma
        state.bulk = -zeta_over_s(params, temperature) * entropy * HBARC * theta
    elif initial_shear not in ("ns", "zero"):
        raise ValueError(f"unknown initial shear {initial_shear!r}")
    state.conserved, _, _, _, _ = total_tensor(energy, gamma * vx, gamma * vy, state.shear, state.bulk, eos)
    state.previous_u = torch.stack((gamma, gamma * vx, gamma * vy))
    state.previous_tau = tau0
    return state
