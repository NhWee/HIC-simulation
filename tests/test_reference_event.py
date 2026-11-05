"""End-to-end regression test of the README reference event on CPU."""
import json

import numpy as np
import pytest

from tests.reference_event import REFERENCE_PATH, compare, generate


@pytest.fixture(scope="module")
def event(tmp_path_factory):
    directory = tmp_path_factory.mktemp("event")
    audit = generate(directory)
    return audit, directory / "event.npz"


def test_physics_audit_passes(event):
    audit, _ = event
    failed = {key: value for key, value in audit.items() if value is False}
    assert audit["passed"], f"audit failed; false flags: {failed}\n{json.dumps(audit, indent=2)}"


def test_matches_stored_reference(event):
    audit, _ = event
    reference = json.loads(REFERENCE_PATH.read_text())
    problems = compare(audit, reference)
    assert not problems, "\n".join(problems) + "\nIf the change is intentional, run: python -m tests.reference_event"


def test_cache_contents_are_consistent(event):
    audit, cache = event
    with np.load(cache) as data:
        assert str(data["format_version"]) == "hic_blender_event_v4"
        assert int(data["n_part"]) == audit["n_part"]
        assert int(data["particle_species"].size) == audit["sampled_particles"]
        assert data["energy_density"].shape[0] == audit["hydro_snapshots"]
        assert data["surface_tau"].size == audit["freezeout_cells"]
        assert np.all(np.diff(data["hydro_tau"]) > 0.0)
        energy = data["particle_energy"].astype(np.float64)
        p2 = (
            data["particle_momentum_x"].astype(np.float64) ** 2
            + data["particle_momentum_y"].astype(np.float64) ** 2
            + data["particle_momentum_longitudinal"].astype(np.float64) ** 2
        )
        mass = data["particle_mass"].astype(np.float64)
        np.testing.assert_allclose(energy**2 - p2, mass**2, atol=1.0e-4)
