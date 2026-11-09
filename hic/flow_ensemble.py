"""Centrality-binned event ensemble: multiplicity, identified mean pT, v_n{2}, v_n{2}(pT) and eccentricities.

1. Centrality is defined from a Monte Carlo Glauber scan with the same nuclear geometry as the
   hydro events: b is drawn from p(b) ~ b, each configuration is identified by its random seed, and
   events are ranked by the initial-entropy proxy 0.5 (1 - alpha) N_part + alpha N_coll (the integral
   of the deposited source), the quantity the final multiplicity follows.
2. For each centrality class, the first ``--events-per-class`` configurations of that class are run
   through ``hic.blender_event_pipeline.simulate`` with one global ``--energy-scale``.
3. Flow is measured with two-particle cumulants on the Cooper-Frye pool of each event, excluding
   self-pairs and decay siblings, and with events weighted by their number of pairs:

       v_n{2}^2 = sum_events (pairs . cos n dphi) / sum_events pairs,
       v_n{2}(pT) = <<cos n(phi_pT - phi_ref)>> / v_n{2},

   with jackknife (leave-one-event-out) uncertainties. eps_n{2} = sqrt(<eps_n^2>) uses the
   energy density at tau0 about its centroid.

Usage:
    python -m hic.flow_ensemble --energy-scale 80 --classes 0-5,10-20,20-30,30-40,40-50 \\
        --events-per-class 8 --grid-size 250 --extent 15 --output outputs/flow_ensemble.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from hic import blender_event_pipeline as pipeline
from hic.nuclear_collision_2d import tag_glauber_collisions

DEFAULT_CLASSES = "0-5,10-20,20-30,30-40,40-50"


def parse_classes(text: str) -> list[tuple[float, float]]:
    classes = []
    for part in text.split(","):
        low, high = (float(v) for v in part.split("-"))
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(f"bad centrality class {part!r}")
        classes.append((low, high))
    return classes


def glauber_scan(event_args: argparse.Namespace, events: int, b_max: float, seed: int) -> list[dict[str, float]]:
    """Glauber configurations keyed by seed, reproducing ``build_event_geometry`` for that seed."""
    b_rng = np.random.default_rng(seed)
    records = []
    configuration_seed = 1000 * seed
    while len(records) < events:
        configuration_seed += 1
        b = float(b_max * math.sqrt(b_rng.random()))  # p(b) ~ b
        rng = np.random.default_rng(configuration_seed)
        a = pipeline.sample_woods_saxon_3d(event_args.mass_number, event_args.radius, event_args.skin_depth, rng)
        c = pipeline.sample_woods_saxon_3d(event_args.mass_number, event_args.radius, event_args.skin_depth, rng)
        part_a, part_b, collision_x, _ = tag_glauber_collisions(
            a[:, 0], a[:, 1], c[:, 0], c[:, 1], b, event_args.inelastic_cross_section
        )
        n_coll = int(collision_x.size)
        if n_coll == 0:
            continue  # not an inelastic event
        n_part = int(part_a.sum() + part_b.sum())
        alpha = event_args.binary_fraction
        records.append({
            "seed": configuration_seed,
            "impact_parameter": b,
            "n_part": n_part,
            "n_coll": n_coll,
            "source_proxy": 0.5 * (1.0 - alpha) * n_part + alpha * n_coll,
        })
    order = np.argsort([-r["source_proxy"] for r in records], kind="stable")
    for rank, index in enumerate(order):
        records[index]["centrality_percent"] = 100.0 * (rank + 0.5) / len(records)
    return records


def jackknife(values_num: np.ndarray, values_den: np.ndarray, statistic) -> tuple[float, float]:
    """Statistic of sum(num)/sum(den) and its jackknife error over events (axis 0)."""
    full = statistic(values_num.sum(axis=0) / values_den.sum(axis=0))
    count = values_num.shape[0]
    if count < 2:
        return full, float("nan")
    leave_one_out = np.asarray([
        statistic((values_num.sum(axis=0) - values_num[i]) / (values_den.sum(axis=0) - values_den[i]))
        for i in range(count)
    ])
    error = np.sqrt((count - 1) / count * np.sum((leave_one_out - leave_one_out.mean(axis=0)) ** 2, axis=0))
    return full, error


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def summarize_class(events: list[dict]) -> dict[str, object]:
    def mean_and_error(key):
        values = np.asarray([e[key] for e in events], dtype=np.float64)
        error = values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else float("nan")
        return float(values.mean()), float(error)

    summary: dict[str, object] = {"events": len(events)}
    for key in ("n_part", "n_coll", "impact_parameter", "dnch_deta", "dn_dy_pion", "dn_dy_kaon", "dn_dy_baryon",
                "mean_pt_gev", "mean_pt_charged_pion_gev", "mean_pt_charged_kaon_gev",
                "mean_pt_proton_antiproton_gev", "entropy_flux_through_freezeout"):
        summary[key], summary[f"{key}_error"] = mean_and_error(key)
    flows = [e["flow"] for e in events]
    edges = np.asarray(flows[0]["pt_edges"])
    summary["pt_edges"] = edges.tolist()
    for n in pipeline.FLOW_HARMONICS:
        num = np.asarray([f[f"v{n}_ref_num"] for f in flows], dtype=np.float64)
        den = np.asarray([f[f"v{n}_ref_den"] for f in flows], dtype=np.float64)
        vn, vn_error = jackknife(num, den, signed_sqrt)
        summary[f"v{n}_2"], summary[f"v{n}_2_error"] = float(vn), float(vn_error)
        eps = np.asarray([e[f"eps{n}_participant"] for e in events])
        summary[f"eps{n}_2"] = float(math.sqrt(np.mean(eps**2)))
        summary[f"v{n}_pp"], summary[f"v{n}_pp_error"] = mean_and_error(f"v{n}_participant_plane")
        reference = vn if vn > 0 else float("nan")
        for label in ("charged", "pion", "kaon", "proton"):
            key = f"v{n}_{label}_pt"
            if f"{key}_num" not in flows[0]:
                continue
            bin_num = np.asarray([f[f"{key}_num"] for f in flows], dtype=np.float64)
            bin_den = np.asarray([f[f"{key}_den"] for f in flows], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio, ratio_error = jackknife(bin_num, bin_den, lambda x: x)
            summary[key] = (np.asarray(ratio) / reference).tolist()
            summary[f"{key}_error"] = (np.asarray(ratio_error) / reference).tolist()
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)
    for label in ("charged", "pion", "kaon", "proton"):
        spectra = np.asarray([f[f"dn_dy_{label}_pt"] for f in flows], dtype=np.float64) / (2.0 * math.pi * centers * widths)
        summary[f"spectrum_{label}"] = spectra.mean(axis=0).tolist()
        summary[f"spectrum_{label}_error"] = (
            spectra.std(axis=0, ddof=1) / math.sqrt(len(flows)) if len(flows) > 1 else np.full(centers.size, np.nan)
        ).tolist()
    return summary


def run_ensemble(args: argparse.Namespace) -> dict[str, object]:
    base = [
        "--grid-size", str(args.grid_size),
        "--extent", str(args.extent),
        "--energy-scale", f"{args.energy_scale:.10g}",
        "--particles", "1",
        "--candidate-pool", str(args.candidate_pool),
    ]
    if args.cpu:
        base.append("--cpu")
    event_args = pipeline.parse_args(base)
    classes = parse_classes(args.classes)
    scan = glauber_scan(event_args, args.glauber_events, args.b_max, args.seed)
    result: dict[str, object] = {
        "description": "Centrality-binned ensemble from hic.flow_ensemble",
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "classes": [],
    }
    started = time.time()
    for low, high in classes:
        members = [r for r in scan if low <= r["centrality_percent"] < high][: args.events_per_class]
        if len(members) < args.events_per_class:
            raise RuntimeError(f"only {len(members)} Glauber events in {low}-{high}%; increase --glauber-events")
        events = []
        for record in members:
            event = pipeline.parse_args(base + ["--seed", str(record["seed"]), "--impact-parameter", f"{record['impact_parameter']:.10g}"])
            outcome = pipeline.simulate(event, verbose=False)
            audit = outcome.audit
            if outcome.geometry.n_part != record["n_part"]:
                raise RuntimeError("hydro event does not reproduce the Glauber scan geometry")
            entry = {**record, **{k: v for k, v in audit.items() if isinstance(v, (int, float, bool))}}
            entry["flow"] = {k: np.asarray(v).tolist() for k, v in outcome.particles.flow.items()}
            events.append(entry)
            print(
                f"  {low:g}-{high:g}%: seed {record['seed']} b={record['impact_parameter']:.2f} Npart={record['n_part']} "
                f"dNch/deta={audit['dnch_deta']:.1f} v2{{2}}^ev={audit['v2_two_particle']:.4f} "
                f"audit={'ok' if audit['passed'] else 'FAILED'} ({time.time() - started:.0f}s)",
                flush=True,
            )
        result["classes"].append({"summary": summarize_events(events, low, high), "events": events})
    return result


def summarize_events(events: list[dict], low: float, high: float) -> dict[str, object]:
    """Class summary from the events that passed the physics audit; failures are listed, not averaged."""
    passed = [e for e in events if e["passed"]]
    if not passed:
        raise RuntimeError(f"no event in {low:g}-{high:g}% passed the physics audit")
    summary = summarize_class(passed)
    summary.update({
        "centrality_low": low,
        "centrality_high": high,
        "failed_audits": [e["seed"] for e in events if not e["passed"]],
        "failed_entropy_flux_ratio": [e["entropy_flux_ratio"] for e in events if not e["passed"]],
    })
    return summary


def resummarize(result: dict[str, object]) -> dict[str, object]:
    for entry in result["classes"]:
        old = entry["summary"]
        entry["summary"] = summarize_events(entry["events"], old["centrality_low"], old["centrality_high"])
    return result


def write_summary_csv(result: dict[str, object], path: Path) -> None:
    keys = ["centrality_low", "centrality_high", "events", "n_part", "dnch_deta", "dnch_deta_error",
            "mean_pt_charged_pion_gev", "mean_pt_charged_kaon_gev", "mean_pt_proton_antiproton_gev",
            "eps2_2", "v2_2", "v2_2_error", "eps3_2", "v3_2", "v3_2_error", "v2_pp", "v3_pp"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for entry in result["classes"]:
            writer.writerow([entry["summary"][k] for k in keys])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energy-scale", type=float, default=None, help="Global normalization (see calibrate_normalization).")
    parser.add_argument("--from-json", type=Path, default=None, help="Recompute class summaries of an existing run.")
    parser.add_argument("--classes", default=DEFAULT_CLASSES, help="Comma-separated centrality classes in percent.")
    parser.add_argument("--events-per-class", type=int, default=8)
    parser.add_argument("--glauber-events", type=int, default=4000, help="Glauber configurations for the centrality scan.")
    parser.add_argument("--b-max", type=float, default=20.0, help="fm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-size", type=int, default=250, help="Transverse cells (dx <= 0.12 fm is needed for converged v2).")
    parser.add_argument("--extent", type=float, default=15.0, help="Transverse half-width in fm.")
    parser.add_argument("--candidate-pool", type=int, default=80000, help="Cooper-Frye pool per event.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/flow_ensemble.json"))
    args = parser.parse_args()

    if args.from_json is not None:
        result = resummarize(json.loads(args.from_json.read_text()))
    else:
        if args.energy_scale is None:
            parser.error("--energy-scale is required (unless --from-json)")
        print(f"Glauber scan: {args.glauber_events} events; classes {args.classes}; {args.events_per_class} hydro events each")
        result = run_ensemble(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="ascii")
    write_summary_csv(result, args.output.with_suffix(".csv"))
    for entry in result["classes"]:
        s = entry["summary"]
        print(
            f"{s['centrality_low']:g}-{s['centrality_high']:g}%: Npart={s['n_part']:.0f} dNch/deta={s['dnch_deta']:.0f} "
            f"v2{{2}}={s['v2_2']:.4f}+-{s['v2_2_error']:.4f} v3{{2}}={s['v3_2']:.4f}+-{s['v3_2_error']:.4f} "
            f"eps2{{2}}={s['eps2_2']:.3f} eps3{{2}}={s['eps3_2']:.3f} "
            f"({s['events']} events; failed audits {s['failed_audits']})"
        )
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
