"""Calibrate the initial energy-density normalization of the reference event to a target multiplicity.

The single free normalization of the initial condition (``--initial-peak-energy``, the peak
Landau-frame energy density at tau0) is tuned so that the event-averaged value of one observable
matches a target:

* ``entropy``           -- dS/deta_s of the matter above T_sw at tau0. Ideal hydrodynamics conserves
                           it, and it equals the entropy flux through the freeze-out surface to
                           within the audit tolerance, but unlike that flux it is a smooth,
                           deterministic function of the normalization. It does not depend on which
                           hadron species are sampled; compare with dS/dy estimated from data.
* ``dnch_deta``         -- charged-hadron pseudorapidity density at midrapidity after resonance
                           decays (weak decays not performed), comparable with a measured
                           dN_ch/deta. It is a Monte Carlo estimate (relative noise ~0.5-1%).

The target must be supplied explicitly. Tune in one centrality class and keep the others as
predictions. Usage:

    python -m hic.calibrate_normalization --observable dnch_deta --target <measured dNch/deta> \\
        --impact-parameter 2.2 --seeds 1 2 3 --output outputs/calibration.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from hic import blender_event_pipeline as pipeline

OBSERVABLES = {
    "entropy": "entropy_at_tau0_above_tsw",
    "dnch_deta": "dnch_deta",
}


def evaluate(base_args: list[str], peak_energy: float, seeds: list[int], observable: str) -> float:
    values = []
    for seed in seeds:
        args = pipeline.parse_args(base_args + ["--seed", str(seed), "--initial-peak-energy", f"{peak_energy:.10g}"])
        audit = pipeline.simulate(args, verbose=False).audit
        if not audit["passed"]:
            raise RuntimeError(f"physics audit failed at peak energy {peak_energy:g}, seed {seed}")
        values.append(float(audit[OBSERVABLES[observable]]))
    return float(np.mean(values))


def calibrate(
    base_args: list[str],
    seeds: list[int],
    observable: str,
    target: float,
    initial_peak: float = 6.0,
    tolerance: float = 0.01,
    max_iterations: int = 8,
    max_peak: float | None = None,
) -> dict[str, object]:
    """Iterate on a power law value ~ A peak^b fitted (least squares in log-log) to every evaluation.

    Fitting all points instead of the last two (secant) keeps Monte Carlo noise in the observable from
    being amplified near the solution.
    """
    history: list[tuple[float, float]] = []
    peak = initial_peak
    exponent = 0.75  # entropy ~ e^(3/4) for a nearly conformal fluid; refined from the history
    for _ in range(max_iterations):
        value = evaluate(base_args, peak, seeds, observable)
        history.append((peak, value))
        print(f"  peak energy {peak:9.4f} GeV/fm^3 -> {observable} = {value:.4g} (target {target:.4g})", flush=True)
        if abs(value / target - 1.0) <= tolerance:
            break
        log_peak = np.log([p for p, _ in history])
        log_value = np.log([v for _, v in history])
        if len(history) >= 2 and np.ptp(log_peak) > 1.0e-6:
            exponent, intercept = np.polyfit(log_peak, log_value, 1)
            exponent = max(float(exponent), 0.1)
            peak = math.exp((math.log(target) - intercept) / exponent)
        else:
            peak = peak * (target / value) ** (1.0 / exponent)
        if max_peak is not None:
            peak = min(peak, max_peak)
    else:
        raise RuntimeError(f"calibration did not converge within {max_iterations} iterations: {history}")
    return {
        "observable": observable,
        "target": target,
        "seeds": seeds,
        "calibrated_initial_peak_energy_gev_fm3": history[-1][0],
        "calibrated_value": history[-1][1],
        "local_power_law_exponent": exponent,
        "history": [{"initial_peak_energy": p, "value": v} for p, v in history],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate --initial-peak-energy to a multiplicity target.")
    parser.add_argument("--observable", choices=sorted(OBSERVABLES), required=True)
    parser.add_argument("--target", type=float, required=True, help="Target value per unit space-time rapidity.")
    parser.add_argument("--impact-parameter", type=float, default=2.2, help="fm; ~0-5%% central Au+Au by default.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--grid-size", type=int, default=150)
    parser.add_argument("--extent", type=float, default=15.0, help="Transverse half-width (fm); central events need room.")
    parser.add_argument("--initial-peak", type=float, default=6.0)
    parser.add_argument("--candidate-pool", type=int, default=160000, help="Cooper-Frye pool size (Monte Carlo noise).")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration.json"))
    args = parser.parse_args()

    base = [
        "--impact-parameter", str(args.impact_parameter),
        "--grid-size", str(args.grid_size),
        "--extent", str(args.extent),
        "--particles", "0",
        "--candidate-pool", str(args.candidate_pool),
    ]
    if args.cpu:
        base.append("--cpu")
    max_peak = pipeline.read_eos_table(pipeline.parse_args([]).eos_table)[0][-1]
    print(f"Calibrating {args.observable} to {args.target:g} at b={args.impact_parameter} fm, seeds {args.seeds}")
    result = calibrate(base, args.seeds, args.observable, args.target, args.initial_peak, args.tolerance, max_peak=max_peak)
    result["impact_parameter"] = args.impact_parameter
    result["grid_size"] = args.grid_size
    result["extent"] = args.extent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(
        f"Calibrated --initial-peak-energy = {result['calibrated_initial_peak_energy_gev_fm3']:.4f} GeV/fm^3 "
        f"({args.observable} = {result['calibrated_value']:.4g}); saved {args.output}"
    )


if __name__ == "__main__":
    main()
