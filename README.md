# HIC-simulation

GPU-accelerated numerical and visual prototypes for relativistic heavy-ion
collisions, with a reproducible Blender pipeline for an Au+Au event.

The repository contains three levels of tooling:

- a physics-driven Blender pipeline built from a numerical event cache
- a native PySide field simulator for interactive exploration
- smaller Glauber, hydrodynamics, particlization, validation, and educational tools

The Blender result is generated from calculated fields and sampled particles;
the plasma expansion and particle trajectories are not hand animated.

## Representative result

![Landau-matched QGP hydrodynamization for the reference Au+Au event](docs/assets/qgp-hydrodynamization.webp)

The reference event is Au+Au at `sqrt(sNN)=200 GeV`, `b=5.7 fm`, seed 17.
The bright interior is the hydrodynamic temperature field, the highlighted
core is `T>=165 MeV`, and the outer contour marks `Tsw=145 MeV`.

The images show the corrected event described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#corrections-in-this-revision). The
on-screen diagnostics match the reference audit; for example, the field peaks
at 280 MeV right after hydrodynamization.

| Collisionless pre-equilibrium | Sampled hadrons after particlization |
| --- | --- |
| ![Landau-frame effective-temperature field during pre-equilibrium](docs/assets/qgp-formation.webp) | ![Pion, kaon, and proton classes after Cooper-Frye sampling](docs/assets/sampled-hadrons.webp) |

Values from the reference run on a CPU:

| Observable | Result |
| --- | ---: |
| Glauber geometry | `Npart=236`, `Ncoll=574` |
| Pre-equilibrium interval | `0.1 -> 0.6 fm/c` |
| Maximum Landau effective temperature | `475 -> 280 MeV` |
| Energy-weighted handoff pre-flow | `0.149 c` |
| QGP-core cells at hydrodynamization | `3349` |
| Hydrodynamic evolution until `T<145 MeV` | `0.6 -> 4.66 fm/c` |
| Displayed hadrons | `900` |
| Mean transverse momentum | `0.4629 +- 0.0004 GeV` |
| Reaction-plane `v2` | `0.0142 +- 0.0008` |

Mean pT and `v2` come from 10^6 importance-weighted Cooper-Frye candidates;
the uncertainties are their statistical errors. These values are regression
references rather than experimental predictions, and `tests/test_reference_event.py`
checks them. The complete run writes a 13-second, 1280 x 720, 24 fps Blender
animation and a machine-readable physics audit.

## Physics scope

The reference Blender event uses:

1. Woods-Saxon Au nuclei and event-by-event Monte Carlo Glauber geometry
2. wounded-nucleon plus binary-collision energy deposition
3. transverse collisionless free streaming from `tau=0.1` to `0.6 fm/c`
4. transport of `T00`, `T0i`, and `Tij`, with timelike-eigenvector Landau matching
5. a tabulated crossover equation of state whose temperature and sound speed
   follow thermodynamically from its pressure (see [data/README.md](data/README.md))
6. boost-invariant 2+1D ideal relativistic hydrodynamics with the exact
   Milne-frame source terms `-(T^tautau + p)/tau` and `-T^taui/tau`
7. a `Tsw=145 MeV` isothermal surface
8. positive-contribution Cooper-Frye sampling of pion, kaon, and proton classes
9. ballistic free streaming of sampled hadrons

The visualization marks a `T>=165 MeV` QGP core separately from the 145 MeV
particlization contour. The pre-equilibrium color field uses a Landau-frame
effective temperature as a diagnostic; it is not a claim of local thermal
equilibrium before hydrodynamization.

This is a physically constrained prototype, not a precision event generator.
It does not include second-order viscous hydrodynamics, a full 3+1D kinetic
pre-equilibrium solver, jet-medium interactions, resonance decays, or a
SMASH/UrQMD hadronic afterburner. The finite-volume solver is first order and
diffusive: it reproduces the analytic Gubser flow to about 10% (L1) at
`dx=0.1 fm` and converges at first order.

## Quick start

Requirements:

- Windows PowerShell
- Python 3.11 or newer
- Blender 5.1 or compatible for the Blender scene
- FFmpeg on `PATH` for MP4 encoding
- an NVIDIA GPU for the default CUDA 12.8 setup, or use CPU mode

Create the local environment:

```powershell
git clone https://github.com/NhWee/HIC-simulation.git
cd HIC-simulation
.\setup.ps1
```

For a machine without a compatible NVIDIA GPU:

```powershell
.\setup.ps1 -Cpu
```

To select a particular Python installation:

```powershell
.\setup.ps1 -PythonExe "C:\Path\To\python.exe"
```

### Linux and macOS

The `run_*.ps1` launchers are thin wrappers around `python -m hic.<module>`,
so every tool also runs directly from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-cpu.txt
python -m hic.hydro_validation --cpu
python -m hic.blender_event_pipeline --cpu
```

Use `requirements-cuda.txt` instead of `requirements-cpu.txt` on a Linux
machine with CUDA 12.8. On macOS, install `torch==2.11.0` from PyPI instead of
either file. Build the Blender scene from the event cache with:

```bash
blender --background --python-exit-code 1 --python tools/blender_hic_scene.py -- \
  --cache outputs/blender_physics_event.npz --output outputs/hic_blender_physics_scene.blend
```

## Reproduce the Blender result

Run validation, calculate the reference event, build the scene and previews,
and render the 13-second H.264 animation:

```powershell
.\run_reproduce_blender.ps1
```

CPU-only reproduction is supported but slower:

```powershell
.\run_reproduce_blender.ps1 -Cpu
```

Skip the full animation while checking the numerical cache, audit, scene, and
representative PNG previews:

```powershell
.\run_reproduce_blender.ps1 -SkipAnimation
```

The reference configuration is fixed by the reproduction script:

- Au+Au at `sqrt(sNN)=200 GeV`
- impact parameter `b=5.7 fm`
- random seed `17`
- `160 x 160` transverse grid
- 48 pre-equilibrium angular streams
- 900 sampled particles

Generated files are written to `outputs/` and intentionally ignored by Git.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for tested versions, expected
audit values, manual commands, and determinism notes.

## Blender controls

Calculate a new event and build the scene:

```powershell
.\run_blender_hic_scene.ps1
```

Change collision geometry or open the finished scene:

```powershell
.\run_blender_hic_scene.ps1 -ImpactParameter 9.0 -Seed 31 -Open
```

Reuse an existing event cache and render the animation:

```powershell
.\run_blender_hic_scene.ps1 -ReuseCache -RenderAnimation -NoPreviews
```

If Blender is not on `PATH`, pass its executable explicitly:

```powershell
.\run_blender_hic_scene.ps1 -BlenderPath "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
```

## Validation

Run the hydrodynamic validation suite:

```powershell
.\run_hydro_validation.ps1 --output outputs\hydro_validation.csv
```

It checks uniform rest and boosted states, energy conservation of a Gaussian
fireball including boundary outflow, analytic Bjorken cooling, crossover-EoS
substep convergence, and the analytic Gubser flow at two resolutions. The
Gubser test requires first-order convergence, which a wrong coupling between
transverse flow and longitudinal expansion cannot reach. Additional
benchmarks are available through:

```powershell
.\run_hydro_convergence.ps1 --grid-sizes 40,60,80 --cfl-values 0.15,0.25,0.35
.\run_hydro_shock_tube.ps1
```

The Blender event generator also writes
`outputs\blender_physics_event.audit.json` and exits unsuccessfully if its
causality, monotonicity, Landau-handoff, freeze-out, or particle checks fail.

Check the equation-of-state table, or a table of your own, for thermodynamic
consistency:

```powershell
.\.venv\Scripts\python.exe -m hic.eos --eos-table data\eos_table.csv
```

The automated tests cover the EoS, the hydrodynamic solver, Landau matching,
the Glauber and Cooper-Frye samplers, every command-line tool, the desktop
apps in headless mode, and the reference event. GitHub Actions runs them on
Linux (Python 3.11 and 3.13) and runs `setup.ps1` and the launchers on
Windows. Run them locally with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

## Other entry points

| Command | Purpose |
| --- | --- |
| `.\run_pyside_simulator_app.ps1` | Native interactive field and hadronization simulator |
| `.\run_glauber_ensemble.ps1 --events 200` | Monte Carlo Glauber event ensemble |
| `.\run_centrality_analysis.ps1` | Centrality-binned geometry analysis |
| `.\run_ideal_hydro_2d.ps1 --bjorken-expansion` | Standalone ideal-hydro prototype |
| `.\run_particlization.ps1` | Standalone freeze-out particle sampling |
| `.\run_event_pipeline.ps1 --events 3` | Small multi-event numerical pipeline |
| `.\run_visualize_results.ps1` | Plotly report from generated CSV files |
| `.\run_visual_simulation.ps1` | Browser-playable qualitative simulation |
| `.\run_educational_visual.ps1` | Classroom-style educational board |
| `.\run_educational_simulator_app.ps1` | Lightweight native educational simulator |

## Repository layout

```text
data/                       compact crossover EoS table and its description
docs/                       model notes
hic/                        numerical models and applications
tests/                      pytest suite, including the reference-event regression
tools/blender_hic_scene.py  cache-to-Blender scene builder
run_*.ps1                   reproducible Windows entry points
.github/workflows/          continuous integration on Linux and Windows
outputs/                    generated caches, reports, scenes, images, and video
```

## Model references

- [Free-streaming approximation and Landau matching](https://arxiv.org/abs/0812.3393)
- [Gubser flow, the analytic solution used for validation](https://arxiv.org/abs/1006.0006)
- [Universal early transverse flow](https://arxiv.org/abs/0810.4325)
- [KoMPoST kinetic pre-equilibrium framework](https://arxiv.org/abs/1805.00961)
- [Cooper-Frye particlization](https://doi.org/10.1103/PhysRevD.10.186)
