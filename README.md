# HIC Simulation

GPU-accelerated toy simulations for exploring heavy-ion collision initial
conditions and early-time evolution.

## Current prototype

The first prototype is a 2D qualitative heavy-ion collision viewer:

- moving Woods-Saxon-like nuclear thickness profiles
- wounded-nucleon and binary-collision mixed initial energy density
- simple diffusion, radial expansion, and damping evolution
- interactive Plotly HTML output

This is an exploratory model, not a validated relativistic hydrodynamics solver.

## Run

From this directory:

```powershell
& ..\gpu\.venv\Scripts\python.exe -m hic.nuclear_collision_2d
```

The default output is:

```text
outputs\nuclear_collision_2d.html
```

For a quick smoke test:

```powershell
& ..\gpu\.venv\Scripts\python.exe -m hic.nuclear_collision_2d --grid-size 80 --steps 80 --frame-every 8 --output outputs\nuclear_collision_2d_smoke.html
```

