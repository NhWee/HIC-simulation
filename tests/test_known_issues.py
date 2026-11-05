"""Physics checks that the current code is known to FAIL (see docs/ROADMAP.md).

Mark them ``xfail(strict=True, raises=AssertionError)``: the suite stays green while the defect
exists and turns red (XPASS) once a fix lands, so the marker is removed together with the fix.
Any other exception is a real failure.

Resolved so far: P0-1 EoS (tests/test_eos.py), P0-3 Milne sources (tests/test_hydro_sources.py),
P0-4 freeze-out surface (tests/test_freezeout.py). No known-issue checks are open at the moment.
"""
