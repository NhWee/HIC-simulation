# Reproducibility guide

This document reproduces the reference Au+Au Blender result from source. No
generated cache, `.blend`, PNG, CSV, or MP4 file is required from Git.

## Tested environment

The reference values below were regenerated on a CPU with Python 3.13.18,
NumPy 2.4.4, and PyTorch 2.15.0 on Linux. The Linux jobs in
`.github/workflows/tests.yml` check them against the pinned PyTorch 2.11.0
CPU build through `tests/test_reference_event.py`.

The README images were re-rendered from this revision's CPU reference cache
with the Blender 5.1.2 Python module (`bpy` from PyPI) in background mode on
Linux, using Mesa software OpenGL, and scaled from 1280 x 720 to 1024 x 576.
The 13-second animation was last rendered, before the corrections below, with:

- Windows, PowerShell
- Python 3.13.3
- NumPy 2.4.4
- PyTorch 2.11.0+cu128
- Plotly 6.8.0
- PySide6 6.10.1
- pyqtgraph 0.14.0
- NVIDIA GeForce RTX 4060 Ti
- Blender 5.1.2
- FFmpeg 8.1.1

Dependency versions are pinned in `requirements.txt`,
`requirements-cuda.txt`, and `requirements-cpu.txt`.

## External prerequisites

Install Python 3.11 or newer. Blender and FFmpeg are only required for the
Blender deliverables. Typical Windows installations are:

```powershell
winget install --id BlenderFoundation.Blender -e
winget install --id Gyan.FFmpeg -e
```

Restart PowerShell after installation so updated `PATH` entries are visible.
The Blender runner also searches common Blender installation directories.

## Clean-clone procedure

```powershell
git clone https://github.com/NhWee/HIC-simulation.git
cd HIC-simulation
.\setup.ps1
.\run_reproduce_blender.ps1
```

Use `-Cpu` for both setup and reproduction when CUDA 12.8 is unavailable:

```powershell
.\setup.ps1 -Cpu
.\run_reproduce_blender.ps1 -Cpu
```

The CPU path implements the same equations. Floating-point roundoff and runtime
differ between devices; the tolerances below absorb them.

On Linux or macOS, run the same stages with `python -m`, as described in the
README:

```bash
python -m hic.hydro_validation --cpu --output outputs/hydro_validation.csv
python -m hic.blender_event_pipeline --cpu
python -m pytest
```

## Manual staged procedure

Run the validation suite:

```powershell
.\run_hydro_validation.ps1 --output outputs\hydro_validation.csv
```

Generate the fixed reference event and Blender previews:

```powershell
.\run_blender_hic_scene.ps1 -ImpactParameter 5.7 -Seed 17 -GridSize 160 -Particles 900
```

Render the full animation without recalculating the event:

```powershell
.\run_blender_hic_scene.ps1 -ReuseCache -RenderAnimation -NoPreviews
```

Open the generated scene:

```powershell
.\run_blender_hic_scene.ps1 -ReuseCache -Open
```

## Expected artifacts

The run creates these primary files:

```text
outputs/blender_physics_event.npz
outputs/blender_physics_event.audit.json
outputs/hic_blender_physics_scene.blend
outputs/hic_blender_physics_simulation.mp4
outputs/blender_physics_contact.png
outputs/blender_physics_qgp_formation.png
outputs/blender_physics_landau_handoff.png
outputs/blender_physics_qgp_onset.png
outputs/blender_physics_hydro.png
outputs/blender_physics_particlization.png
outputs/blender_physics_final_hadrons.png
```

The reference video is 1280 x 720, 24 fps, 312 frames, and 13 seconds long.

## Reference audit

`outputs/blender_physics_event.audit.json` must contain `"passed": true`.
For seed 17 and `b=5.7 fm`, the CPU reference run produced:

```text
Npart                                      236
Ncoll                                      574
pre-equilibrium tau range                  0.1 -> 0.6 fm/c
initial max Landau effective temperature   0.47479 GeV
handoff max effective temperature          0.28024 GeV
energy-weighted handoff pre-flow            0.14869 c
Landau eigen-equation residual              3.54e-7
handoff temperature L-infinity error        6.0e-8 GeV
QGP-core cells at hydrodynamization         3349
hydrodynamic evolution                      0.6 -> 4.657 fm/c (52 snapshots)
freeze-out cells                            4039
displayed particles                         900
Cooper-Frye candidates for observables      1000000
mean pT                                     0.4629 +- 0.0004 GeV
v2 relative to the reaction plane           0.0142 +- 0.0008
```

The integer Glauber geometry must match exactly for the fixed NumPy seed.
`tests/test_reference_event.py` accepts 0.2% in the temperatures, 0.001 c in
the pre-flow, 1% in the cell counts, 0.05 fm/c in the hydrodynamic lifetime,
and the larger of four statistical errors and 0.0015 GeV (pT) or 0.002 (v2)
in the observables. Small differences across PyTorch devices, GPU
architectures, and Blender versions are expected; these tolerances, rather
than byte-identical generated files, define a successful reproduction.

## Corrections in this revision

Earlier revisions produced the values in the first column. Each correction
below was checked separately on the same event.

| Quantity | Before | After |
| --- | ---: | ---: |
| initial max Landau effective temperature | 0.612 GeV | 0.475 GeV |
| handoff max effective temperature | 0.370 GeV | 0.280 GeV |
| QGP-core cells at hydrodynamization | 3524 | 3349 |
| hydrodynamic evolution until `T<145 MeV` | 4.55 fm/c | 4.66 fm/c |
| mean pT, average of 4 x 10^6 candidates | 0.4477 GeV | 0.4631 GeV |
| v2, average of 4 x 10^6 candidates | 0.0132 +- 0.0003 | 0.0147 +- 0.0003 |

1. **Boost-invariant source terms.** The Bjorken step cooled `e` at fixed
   velocity. With transverse flow the Milne-frame equations require
   `d(tau T^taui)/dtau = 0` from the geometry, so the old step damped the
   transverse momentum as `tau^-(1+cs2)` instead of `tau^-1`. The analytic
   Gubser flow exposes this: the old solver stalled at a relative L1 error of
   0.31-0.45 under grid refinement, while the corrected one converges at first
   order (0.20, 0.10, 0.05 at `dx=0.2, 0.1, 0.05 fm`). This correction raises
   the mean pT by 3.5% and v2 by about 10%.
2. **Equation-of-state temperatures.** The temperature column of
   `data/eos_table.csv` was not consistent with its pressure column; it
   overstated T by up to 32%. Temperatures now follow from `p(e)` with
   `T=145 MeV` kept at `e=0.35 GeV/fm^3`, so the switching surface stays where
   it was. The tabulated pressure no longer has a constant 0.006 GeV/fm^3
   vacuum value below the first node, and the solver's signal speed uses the
   largest `dp/de` (0.367) instead of 1/3. See [data/README.md](data/README.md).
3. **Observable statistics.** Mean pT and v2 used 28800 candidates, which
   gives a single-run v2 spread of 0.005-0.006, about half of v2 itself. The
   pipeline now uses 10^6 candidates (spread 0.0005) and reports the errors.
4. **Primitive recovery.** For the tabulated and crossover EoS the damped
   fixed-point iteration left errors of up to 3% in `e` at high velocity. The
   undamped iteration reaches float32 roundoff; the reference observables did
   not change beyond their errors.

The PySide hydro cache, the event pipeline, and the standalone tools use the
same solver and EoS code and change accordingly.

## Numerical traceability

The event cache records:

- sampled nuclei, participant masks, and binary-collision centers
- all pre-equilibrium tensor components and Landau-frame diagnostics
- hydrodynamic energy, temperature, velocity, eccentricity, and time arrays
- the freeze-out surface and oriented surface elements
- sampled particle species, positions, four-momenta, and Cooper-Frye weights
- the impact parameter, seed, model description, device, and cache format

The Blender file stores the source cache path and event metadata as scene custom
properties. The scene builder audits snapshot counts, visibility transitions,
fixed-camera behavior, and ballistic two-key particle tracks before saving.

## Artifact policy

`outputs/` is ignored by Git because a complete run can regenerate every file
and because numerical caches, Blender files, PNG sequences, and videos are
large. Keep source code, scripts, the EoS table, and audit logic under version
control; publish selected rendered media through a release when needed.

## Known model limitations

The pre-equilibrium stage is a transverse collisionless approximation rather
than IP-Glasma or KoMPoST. Hydrodynamics is ideal and boost invariant rather
than event-by-event 3+1D second-order viscous hydrodynamics, and the
first-order Rusanov scheme is diffusive (about 10% L1 error against the Gubser
flow at `dx=0.1 fm`). The included EoS is a compact demonstration table: it is
thermodynamically consistent, but its `dp/de` exceeds 1/3 above about
2.5 GeV/fm^3, unlike lattice QCD. With the default initial peak energy of
6 GeV/fm^3 the hydrodynamic stage starts at 280 MeV and lasts only about
4 fm/c, shorter than in realistic RHIC conditions. Sampled hadrons free-stream
without resonance decays or a hadronic transport afterburner. The output is
suitable for numerical development and education, not precision extraction of
QCD transport coefficients.
