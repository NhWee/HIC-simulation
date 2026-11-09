"""Flow estimators, participant planes, centrality scan and the ensemble driver (ROADMAP P1-2)."""
import math

import numpy as np
import pytest

import hic.blender_event_pipeline as pipeline
from hic import flow_ensemble


def synthetic_decaying_pool(v2: float, parents: int, opening: float, rng):
    """Parents with dN/dphi ~ 1 + 2 v2 cos 2phi, each splitting into two strongly correlated daughters."""
    phi = rng.uniform(0.0, 2.0 * math.pi, 4 * parents)
    keep = rng.uniform(0.0, 1.0 + 2.0 * v2, phi.size) < 1.0 + 2.0 * v2 * np.cos(2.0 * phi)
    phi = phi[keep][:parents]
    half = rng.uniform(0.0, opening, phi.size)
    daughters = np.concatenate((phi + half, phi - half))
    parent = np.concatenate((np.arange(phi.size), np.arange(phi.size)))
    weight = rng.uniform(0.5, 1.5, phi.size)[parent]  # importance weight shared by siblings
    # Single-daughter flow: v2 <cos 2a> with a uniform in [0, opening].
    expected = v2 * math.sin(2.0 * opening) / (2.0 * opening)
    return weight, daughters, parent, phi.size, expected


def test_pair_sum_is_unbiased_with_flow_and_decays():
    rng = np.random.default_rng(4)
    weight, phi, parent, groups, expected = synthetic_decaying_pool(0.10, 60000, 0.6, rng)
    everything = np.ones(phi.size, dtype=bool)
    num, den = pipeline.pair_sum(weight, np.exp(2j * phi), parent, groups, everything, everything)
    assert math.sqrt(num / den) == pytest.approx(expected, rel=0.03)


def test_pair_sum_removes_sibling_nonflow():
    # No flow at all, but each parent yields ten nearly collinear daughters (pure nonflow). The naive
    # estimator (self-pairs removed only) is biased by the sibling pairs, ~ (d - 1)/(d N) <cos 2 dphi>;
    # the parent-grouped estimator averages to zero.
    rng = np.random.default_rng(8)
    parents, daughters, spread, repeats = 2000, 10, 0.1, 40
    corrected, naive = [], []
    for _ in range(repeats):
        phi_parent = rng.uniform(0.0, 2.0 * math.pi, parents)
        parent = np.repeat(np.arange(parents), daughters)
        phi = phi_parent[parent] + rng.normal(0.0, spread, parent.size)
        weight = np.ones(parent.size)
        everything = np.ones(phi.size, dtype=bool)
        num, den = pipeline.pair_sum(weight, np.exp(2j * phi), parent, parents, everything, everything)
        naive_num, naive_den = pipeline.pair_sum(
            weight, np.exp(2j * phi), np.arange(phi.size), phi.size, everything, everything
        )
        corrected.append(num / den)
        naive.append(naive_num / naive_den)
    corrected, naive = np.asarray(corrected), np.asarray(naive)
    bias = (daughters - 1) / (daughters * parents) * math.exp(-4.0 * spread**2)  # <cos 2(d1 - d2)>, Gaussian
    assert abs(corrected.mean()) < 4.0 * corrected.std(ddof=1) / math.sqrt(repeats)
    assert (naive - corrected).mean() == pytest.approx(bias, rel=0.1)


def test_participant_eccentricity_of_an_elliptic_gaussian():
    axis = np.linspace(-25.0, 25.0, 501)  # wide enough that r^2 moments are not truncated
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sx, sy = 3.0, 2.0
    density = np.exp(-((xx - 0.7) ** 2) / (2 * sx**2) - ((yy + 0.4) ** 2) / (2 * sy**2))  # off-centre
    planes = pipeline.participant_eccentricities(density, axis)
    eps2, psi2 = planes[2]
    assert eps2 == pytest.approx((sx**2 - sy**2) / (sx**2 + sy**2), rel=1.0e-3)
    # The short axis (y) is the participant plane: flow develops along it.
    assert math.cos(2.0 * (psi2 - math.pi / 2)) == pytest.approx(1.0, abs=1.0e-6)
    assert planes[3][0] < 1.0e-6  # no triangularity


def test_glauber_scan_orders_centrality_and_reproduces_event_geometry():
    event_args = pipeline.parse_args(["--cpu", "--grid-size", "40"])
    scan = flow_ensemble.glauber_scan(event_args, 300, 20.0, seed=3)
    by_centrality = sorted(scan, key=lambda r: r["centrality_percent"])
    proxies = [r["source_proxy"] for r in by_centrality]
    assert proxies == sorted(proxies, reverse=True)
    central = np.mean([r["impact_parameter"] for r in by_centrality[:30]])
    peripheral = np.mean([r["impact_parameter"] for r in by_centrality[-30:]])
    assert central < 4.0 < 10.0 < peripheral
    record = by_centrality[40]
    args = pipeline.parse_args(["--cpu", "--grid-size", "40", "--seed", str(record["seed"]),
                                "--impact-parameter", f"{record['impact_parameter']:.10g}"])
    import torch
    xx, yy, _ = pipeline.make_grid(args.grid_size, args.extent, torch.device("cpu"))
    geometry = pipeline.build_event_geometry(args, xx, yy, np.random.default_rng(args.seed))
    assert (geometry.n_part, geometry.n_coll) == (record["n_part"], record["n_coll"])


def test_jackknife_of_a_ratio_of_sums():
    num = np.asarray([1.0, 2.0, 3.0, 4.0])
    den = np.ones(4)
    value, error = flow_ensemble.jackknife(num, den, lambda x: x)
    assert value == pytest.approx(2.5)
    assert error == pytest.approx(np.std(num, ddof=1) / 2.0)


def test_global_energy_scale_ties_multiplicity_to_geometry():
    common = ["--cpu", "--grid-size", "60", "--extent", "14", "--energy-scale", "80", "--particles", "1",
              "--candidate-pool", "40000", "--seed", "11"]
    central = pipeline.simulate(pipeline.parse_args(common + ["--impact-parameter", "2.0"]), verbose=False).audit
    peripheral = pipeline.simulate(pipeline.parse_args(common + ["--impact-parameter", "9.0"]), verbose=False).audit
    assert central["passed"] and peripheral["passed"]
    assert central["dnch_deta"] > 3.0 * peripheral["dnch_deta"]
    assert central["eps2_participant"] < peripheral["eps2_participant"]


def test_ensemble_driver_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["flow_ensemble", "--energy-scale", "80", "--classes", "0-10,30-50", "--events-per-class", "2",
         "--glauber-events", "200", "--grid-size", "50", "--extent", "14", "--candidate-pool", "30000",
         "--cpu", "--output", str(tmp_path / "ensemble.json")],
    )
    flow_ensemble.main()
    import json
    result = json.loads((tmp_path / "ensemble.json").read_text())
    central, peripheral = (c["summary"] for c in result["classes"])
    assert central["dnch_deta"] > peripheral["dnch_deta"]
    assert central["n_part"] > peripheral["n_part"]
    assert peripheral["eps2_2"] > central["eps2_2"]
    for summary in (central, peripheral):
        assert np.isfinite(summary["v2_2"]) and summary["v2_2"] > 0.0
        assert len(summary["v2_charged_pt"]) == len(summary["pt_edges"]) - 1
    assert (tmp_path / "ensemble.csv").exists()
