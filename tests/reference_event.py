"""Reference configuration and tolerances for the Blender physics event regression test.

Regenerate the stored reference after an intentional physics change with

    python -m tests.reference_event

and commit tests/reference/blender_event_audit.json together with the change.
"""
import json
import platform
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "tests" / "reference" / "blender_event_audit.json"

# The README reference event (Au+Au, b=5.7 fm, seed 17, 160x160 grid, 900 particles), forced onto CPU.
EVENT_ARGS = (
    "--cpu",
    "--impact-parameter", "5.7",
    "--seed", "17",
    "--grid-size", "160",
    "--particles", "900",
)

# Tolerances per audit key. "exact" keys must match bit-for-bit; "rel"/"abs" absorb roundoff
# differences across PyTorch versions and CPU instruction sets. Sampled observables (mean pT,
# vn) are given tolerances comparable to their Monte Carlo spread.
TOLERANCES: dict[str, dict[str, float | bool]] = {
    "passed": {"exact": True},
    "n_part": {"exact": True},
    "n_coll": {"exact": True},
    "preequilibrium_snapshots": {"exact": True},
    "preequilibrium_tau_start_fm_c": {"rel": 1.0e-6},
    "preequilibrium_tau_end_fm_c": {"rel": 1.0e-6},
    "initial_max_landau_effective_temperature_gev": {"rel": 1.0e-3},
    "handoff_max_landau_effective_temperature_gev": {"rel": 1.0e-3},
    "max_preequilibrium_flow": {"rel": 1.0e-3},
    "energy_weighted_mean_preequilibrium_flow": {"rel": 1.0e-3},
    "qgp_cells_at_hydrodynamization": {"rel": 0.01},
    "initial_max_temperature_gev": {"rel": 1.0e-3},
    "final_max_temperature_gev": {"rel": 0.01},
    "max_transverse_velocity": {"rel": 0.01},
    "hydro_snapshots": {"rel": 0.03},
    "freezeout_cells": {"rel": 0.02},
    "freezeout_heating_elements": {"rel": 0.15},
    "entropy_at_tau0_above_tsw": {"rel": 1.0e-3},
    "entropy_flux_ratio": {"abs": 0.01},
    "dnch_deta": {"rel": 0.03},
    "dn_dy_pion": {"rel": 0.03},
    "dn_dy_kaon": {"rel": 0.04},
    "dn_dy_baryon": {"rel": 0.06},
    "hrg_to_eos_entropy_ratio_at_tsw": {"rel": 1.0e-6},
    "mean_pt_charged_pion_gev": {"rel": 0.02},
    "mean_pt_charged_kaon_gev": {"rel": 0.03},
    "mean_pt_proton_antiproton_gev": {"rel": 0.04},
    "sampled_primaries": {"exact": True},
    "mean_pt_gev": {"rel": 0.02},
    "v2_reaction_plane": {"abs": 0.004},
    "v3_reaction_plane": {"abs": 0.004},
}


def compare(actual: dict, reference: dict) -> list[str]:
    problems = []
    for key, spec in reference["observables"].items():
        if key not in actual:
            problems.append(f"{key}: missing from audit")
            continue
        expected, value = spec["value"], actual[key]
        if spec.get("exact"):
            ok = value == expected
            bound = "exact"
        elif "rel" in spec:
            ok = abs(value - expected) <= spec["rel"] * abs(expected)
            bound = f"rel {spec['rel']}"
        else:
            ok = abs(value - expected) <= spec["abs"]
            bound = f"abs {spec['abs']}"
        if not ok:
            problems.append(f"{key}: got {value!r}, reference {expected!r} ({bound})")
    return problems


def generate(output_dir: Path) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from tests.conftest import assert_ok, run_module

    cache = output_dir / "event.npz"
    assert_ok(run_module("hic.blender_event_pipeline", *EVENT_ARGS, "--output", cache))
    return json.loads(cache.with_suffix(".audit.json").read_text())


def main() -> None:
    import numpy
    import torch

    with tempfile.TemporaryDirectory() as directory:
        audit = generate(Path(directory))
    reference = {
        "description": "CPU regression reference for hic.blender_event_pipeline; regenerate with python -m tests.reference_event",
        "event_args": list(EVENT_ARGS),
        "generated_with": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "observables": {key: {"value": audit[key], **spec} for key, spec in TOLERANCES.items()},
    }
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
