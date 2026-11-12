import csv
import math

import numpy as np
import pytest

from hic import blender_event_pipeline as pipeline
from hic import centrality_analysis, glauber_ensemble
from hic.nuclear_collision_2d import sample_woods_saxon_nucleons, tag_glauber_collisions


def sample_pair(seed: int):
    rng = np.random.default_rng(seed)
    a = sample_woods_saxon_nucleons(197, 6.38, 0.535, rng)
    b = sample_woods_saxon_nucleons(197, 6.38, 0.535, rng)
    return a, b


def test_no_overlap_means_no_participants():
    (ax, ay), (bx, by) = sample_pair(3)
    participant_a, participant_b, collision_x, collision_y = tag_glauber_collisions(ax, ay, bx, by, 30.0, 4.2)
    assert not participant_a.any() and not participant_b.any()
    assert collision_x.size == 0 and collision_y.size == 0


def test_participants_have_a_partner_within_the_cross_section_radius():
    (ax, ay), (bx, by) = sample_pair(5)
    impact = 6.0
    participant_a, participant_b, collision_x, _ = tag_glauber_collisions(ax, ay, bx, by, impact, 4.2)
    distance2 = (ax[:, None] - bx[None, :]) ** 2 + ((ay[:, None] - 0.5 * impact) - (by[None, :] + 0.5 * impact)) ** 2
    touching = distance2 <= 4.2 / math.pi
    assert np.array_equal(participant_a, touching.any(axis=1))
    assert np.array_equal(participant_b, touching.any(axis=0))
    assert collision_x.size == int(touching.sum())


def test_central_events_have_more_participants():
    (ax, ay), (bx, by) = sample_pair(9)
    counts = []
    for impact in (0.0, 6.0, 12.0):
        participant_a, participant_b, collision_x, _ = tag_glauber_collisions(ax, ay, bx, by, impact, 4.2)
        counts.append((int(participant_a.sum() + participant_b.sum()), collision_x.size))
    assert counts[0][0] > counts[1][0] > counts[2][0]
    assert counts[0][1] > counts[1][1] > counts[2][1]
    assert counts[0][0] <= 2 * 197


def test_woods_saxon_sampling_matches_the_density():
    radius, diffuseness = 6.38, 0.535
    rng = np.random.default_rng(2)
    nuclei = np.concatenate([pipeline.sample_woods_saxon_3d(197, radius, diffuseness, rng) for _ in range(40)])
    assert nuclei.shape == (40 * 197, 3)
    r = np.linspace(0.0, radius + 12.0 * diffuseness, 200_001)
    density = r * r / (1.0 + np.exp((r - radius) / diffuseness))
    expected_r2 = np.trapezoid(density * r * r, r) / np.trapezoid(density, r)
    measured_r2 = float(np.mean(np.sum(nuclei.astype(np.float64) ** 2, axis=1)))
    # Re-centering each nucleus removes about <r^2>/A; sampling noise is ~0.5%.
    assert measured_r2 == pytest.approx(expected_r2 * (1.0 - 1.0 / 197), rel=0.02)


def test_glauber_ensemble_and_centrality_bins(tmp_path):
    ensemble = tmp_path / "glauber.csv"
    glauber_ensemble.main(["--cpu", "--events", "60", "--grid-size", "48", "--seed", "4", "--output", str(ensemble)])
    rows = centrality_analysis.read_rows(ensemble)
    assert len(rows) == 60
    assert all(0.0 < row["centrality_percentile"] < 100.0 for row in rows)
    analysis = tmp_path / "centrality.csv"
    centrality_analysis.main(["--input", str(ensemble), "--bins", "0-20,20-50,50-100", "--output", str(analysis)])
    with analysis.open(newline="") as handle:
        n_part = [float(row["n_part_mean"]) for row in csv.DictReader(handle)]
    assert n_part[0] > n_part[1] > n_part[2]
