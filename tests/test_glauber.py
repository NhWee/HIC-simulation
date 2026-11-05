"""Physical sanity checks of the Monte Carlo Glauber geometry used by the reference event."""
import numpy as np
import pytest

from hic.blender_event_pipeline import sample_woods_saxon_3d
from hic.nuclear_collision_2d import tag_glauber_collisions

AU_A, AU_RADIUS, AU_SKIN, SIGMA_NN_FM2 = 197, 6.38, 0.535, 4.2  # 42 mb


def sample_event(impact_parameter: float, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    a = sample_woods_saxon_3d(AU_A, AU_RADIUS, AU_SKIN, rng)
    b = sample_woods_saxon_3d(AU_A, AU_RADIUS, AU_SKIN, rng)
    part_a, part_b, collision_x, _ = tag_glauber_collisions(
        a[:, 0], a[:, 1], b[:, 0], b[:, 1], impact_parameter, SIGMA_NN_FM2
    )
    return int(part_a.sum() + part_b.sum()), int(collision_x.size)


def test_woods_saxon_sample_is_centered_with_expected_radius():
    nucleus = sample_woods_saxon_3d(AU_A, AU_RADIUS, AU_SKIN, np.random.default_rng(3))
    assert nucleus.shape == (AU_A, 3)
    np.testing.assert_allclose(nucleus.mean(axis=0), 0.0, atol=1.0e-5)
    rms = float(np.sqrt(np.mean(np.sum(nucleus.astype(np.float64) ** 2, axis=1))))
    # Woods-Saxon with R=6.38, a=0.535 has <r^2>^(1/2) ~ 5.3 fm; allow sampling noise.
    assert 4.9 < rms < 5.7


def test_reference_event_geometry_is_reproducible():
    # README / REPRODUCIBILITY reference: seed 17, b = 5.7 fm.
    rng = np.random.default_rng(17)
    a = sample_woods_saxon_3d(AU_A, AU_RADIUS, AU_SKIN, rng)
    b = sample_woods_saxon_3d(AU_A, AU_RADIUS, AU_SKIN, rng)
    part_a, part_b, collision_x, _ = tag_glauber_collisions(
        a[:, 0], a[:, 1], b[:, 0], b[:, 1], 5.7, SIGMA_NN_FM2
    )
    assert int(part_a.sum() + part_b.sum()) == 236
    assert int(collision_x.size) == 574


def test_central_collisions_wound_most_nucleons():
    n_part = [sample_event(0.0, seed)[0] for seed in range(5)]
    assert 360 <= np.mean(n_part) <= 2 * AU_A


@pytest.mark.parametrize("seed", range(3))
def test_participants_and_collisions_decrease_with_impact_parameter(seed):
    values = [sample_event(b, seed) for b in (0.0, 5.0, 10.0, 15.0)]
    n_part = [v[0] for v in values]
    n_coll = [v[1] for v in values]
    assert n_part == sorted(n_part, reverse=True)
    assert n_coll == sorted(n_coll, reverse=True)
    assert all(c >= p / 2 for p, c in values if p > 0)  # each participant pair implies >= 1 collision
