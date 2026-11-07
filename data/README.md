# Data files

## `eos_table.csv`

Compact demonstration equation of state at zero baryon chemical potential.
It is not a lattice-QCD table.

| Column | Unit | Meaning |
| --- | --- | --- |
| `energy_density` | GeV/fm^3 | node energy density e |
| `pressure` | GeV/fm^3 | node pressure p(e) |
| `sound_speed2` | 1 | mean of the slopes dp/de on both sides of the node |
| `temperature` | GeV | temperature implied by p(e) |

The hydrodynamics uses only `energy_density` and `pressure`. The code
interpolates p(e) linearly between nodes, continues it linearly to p = 0 at
e = 0 below the first node, and continues it with the last slope above the
last node. The squared sound speed seen by the solver is therefore the
piecewise-constant slope dp/de.

The temperature column follows from the pressure column through

```text
e + p = T s,    d ln s = de / (e + p),    cs2 = dp/de
```

which, for a linear segment with slope k, gives

```text
T_{i+1} / T_i = ((e + p)_{i+1} / (e + p)_i) ** (k / (1 + k)).
```

The integration constant is fixed by `T = 0.145 GeV` at `e = 0.35 GeV/fm^3`,
the particlization temperature used by the reference event.

Check a table, or write a consistent copy of it, with:

```bash
python -m hic.eos --eos-table data/eos_table.csv
python -m hic.eos --eos-table my_table.csv --write-consistent my_table_consistent.csv --anchor-energy 0.35
```

Tables that violate these relations still load, with a warning. Tables with
dp/de outside [0, 1] are rejected because they are mechanically unstable or
acausal.

Earlier versions of this file had a hand-set temperature column that was
not consistent with the pressure column. It overstated T by up to 32% at
6 GeV/fm^3 (0.370 GeV instead of 0.280 GeV), understated it by up to 18% at
0.05 GeV/fm^3, and implied dp/de up to 0.58.
The pressure column is unchanged. The code changes that came with the fix
(p going to zero below the first node instead of a constant 0.006 GeV/fm^3
vacuum pressure, and the Rusanov signal speed taken from the largest slope)
change the hydrodynamic evolution only slightly; temperatures and
temperature-based diagnostics change more.

Above about 2.5 GeV/fm^3 the tabulated pressure has dp/de of 0.35 to 0.37,
slightly above the conformal value 1/3. Lattice QCD stays below 1/3 at zero
chemical potential, so replacing this table with a lattice-based one is the
natural next step for quantitative work.
