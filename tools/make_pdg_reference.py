"""Write tests/data/pdg_hadrons.json: PDG masses, widths and spins for every state in hic.hadrons.

Requires the PDG Python API (``pip install pdg``), which ships the Review of Particle Physics
database. Run from the repository root:

    python tools/make_pdg_reference.py

Mesons are looked up by Monte Carlo id, baryons by name. Broad baryons without a single "mass"
entry take the central value of PDG's quoted Breit-Wigner range ("1410 to 1470 (~1440)").
"""
import json
import re
import sys
import warnings
from pathlib import Path

import pdg

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from hic.hadrons import MULTIPLETS  # noqa: E402

# Multiplet -> {charge: Monte Carlo id} for mesons and for baryons with per-charge PDG entries.
MCID = {
    "pi": {1: 211, 0: 111}, "K": {1: 321, 0: 311}, "eta": {0: 221}, "rho": {0: 113, 1: 213}, "omega": {0: 223},
    "K*(892)": {1: 323, 0: 313}, "eta'": {0: 331}, "f0(980)": {0: 9010221}, "a0(980)": {0: 9000111},
    "phi": {0: 333}, "h1(1170)": {0: 10223}, "b1(1235)": {0: 10113}, "a1(1260)": {0: 20113},
    "K1(1270)": {1: 10323}, "f2(1270)": {0: 225}, "f1(1285)": {0: 20223}, "a2(1320)": {0: 115},
    "K1(1400)": {1: 20323}, "K0*(1430)": {1: 10321}, "K2*(1430)": {1: 325}, "eta(1295)": {0: 100221},
    "pi(1300)": {0: 100111}, "f0(1370)": {0: 10221}, "eta(1405)": {0: 9020221}, "omega(1420)": {0: 100223},
    "K*(1410)": {1: 100323}, "f1(1420)": {0: 20333}, "rho(1450)": {0: 100113}, "f0(1500)": {0: 9030221},
    "f2'(1525)": {0: 335}, "omega(1650)": {0: 30223}, "omega3(1670)": {0: 227}, "pi2(1670)": {0: 10115},
    "rho3(1690)": {0: 117}, "rho(1700)": {0: 30113}, "K*(1680)": {1: 30323}, "K2(1770)": {1: 10325},
    "K3*(1780)": {1: 327},
    "N": {1: 2212, 0: 2112}, "Lambda": {0: 3122}, "Sigma": {1: 3222, 0: 3212, -1: 3112},
    "Xi": {0: 3322, -1: 3312}, "Omega": {-1: 3334},
    "Sigma(1385)": {1: 3224, 0: 3214, -1: 3114}, "Xi(1530)": {0: 3324, -1: 3314},
}


def central(text: str | None) -> float | None:
    match = re.search(r"\(~\s*([\d.]+)\)", text or "") or re.search(r"^\s*([\d.]+)", text or "")
    return float(match.group(1)) if match else None


def main() -> None:
    warnings.filterwarnings("ignore")
    api = pdg.connect()
    out = {"source": str(api), "states": []}
    for mult in MULTIPLETS:
        if mult.name in MCID:
            for charge, mcid in MCID[mult.name].items():
                particle = api.get_particle_by_mcid(mcid)
                out["states"].append({
                    "multiplet": mult.name, "charge": charge, "pdg_name": particle.name,
                    "mass": particle.mass, "width": particle.width, "J": str(particle.quantum_J),
                })
            continue
        particle = api.get_particles_by_name(mult.name)[0]
        mass, width = particle.mass, particle.width
        if mass is None:
            masses = list(particle.masses())
            mass = central(masses[0].display_value_text) / 1000.0 if masses else None
        if width is None:
            widths = list(particle.widths())
            value = central(widths[0].display_value_text) if widths else None
            width = value / 1000.0 if value is not None else None
        out["states"].append({
            "multiplet": mult.name, "charge": None, "pdg_name": particle.name,
            "mass": mass, "width": width, "J": str(particle.quantum_J),
        })
    path = REPO_ROOT / "tests" / "data" / "pdg_hadrons.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"Wrote {len(out['states'])} states to {path}")


if __name__ == "__main__":
    main()
