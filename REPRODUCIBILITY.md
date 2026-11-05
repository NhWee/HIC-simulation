# Reproducibility guide

This document reproduces the reference Au+Au Blender result from source. No
generated cache, `.blend`, PNG, CSV, or MP4 file is required from Git.

## Tested environment

The reference result was last verified with:

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
will differ from the reference GPU run.

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
For seed 17 and `b=5.7 fm`, the reference run produced the values below. The
same values, with per-observable tolerances, are stored in
`tests/reference/blender_event_audit.json` and checked by `python -m pytest`
(CPU). The hydrodynamic and particle values reflect the exact Milne
source terms introduced for ROADMAP item P0-3.

```text
Npart                                      236
Ncoll                                      574
pre-equilibrium tau range                  0.1 -> 0.6 fm/c
initial max Landau effective temperature   0.61184 GeV
handoff max effective temperature          0.37000 GeV
energy-weighted handoff pre-flow            0.14869 c
Landau eigen-equation residual              3.54e-7
handoff temperature L-infinity error        7.38e-7 GeV
QGP-core cells at hydrodynamization         3524
freeze-out cells                            4039
sampled particles                           900
mean pT                                     0.4663 GeV
v2 relative to the reaction plane           0.0111
```

The integer Glauber geometry should be stable for the fixed NumPy seed. Small
floating-point differences in hydro, surface extraction, and sampled
observables are acceptable across PyTorch devices, GPU architectures, and
Blender versions. The audit tolerances, rather than byte-identical generated
files, define a successful reproduction.

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
than event-by-event 3+1D second-order viscous hydrodynamics. The included EoS is
a compact demonstration table, and sampled hadrons free-stream without
resonance decays or a hadronic transport afterburner. The output is suitable
for numerical development and education, not precision extraction of QCD
transport coefficients.
