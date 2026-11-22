"""Download Au+Au sqrt(sNN) = 200 GeV measurements from HEPData into data/measurements/auau200.csv.

Tables (HEPData record, table):
  * PHENIX, PRC 71, 034908 (2005), ins659749 "ppg019fig13": dN_ch/deta and N_part per 5% class.
  * STAR, PRC 88, 014904 (2013), ins1315466 "Table22": sqrt(<v2^2>) after removing nonflow, i.e.
    flow including event-by-event fluctuations -- the quantity the model's v2{2} estimates.
  * STAR, PRL 92, 112301 (2004), ins630160 "fig3-1": <pT> of pi-, K-, pbar at |y| < 0.1 versus
    dN_ch/deta (|eta| < 0.5).

Errors are statistical and systematic added in quadrature (the larger side of an asymmetric
systematic). The CSV feeds ``python -m hic.plot_flow_ensemble ... --data``. Requires PyYAML.

    python tools/fetch_measurements.py
"""
import csv
import io
import math
import subprocess
import tarfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "measurements" / "auau200.csv"


def table(record: str, name: str) -> dict:
    # HEPData answers 403 to most non-browser clients (including Python's urllib) but serves curl, so the
    # archive is fetched with the curl binary.
    url = f"https://www.hepdata.net/download/submission/{record}/1/yaml"
    payload = subprocess.run(["curl", "-sSLf", "--max-time", "120", url], check=True, capture_output=True).stdout
    archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")
    member = next(m for m in archive.getmembers() if m.name.endswith("/" + name))
    return yaml.safe_load(archive.extractfile(member))


def total_error(value: dict, scale: float = 1.0) -> float:
    total = 0.0
    for error in value.get("errors", []):
        if "symerror" in error:
            size = abs(float(str(error["symerror"]).rstrip("%")))
        else:
            asym = error["asymerror"]
            size = max(abs(float(asym["plus"])), abs(float(asym["minus"])))
        total += size * size
    return math.sqrt(total) * scale


def main() -> None:
    rows = []
    phenix = table("ins659749", "ppg019fig13.yaml")
    bins = phenix["independent_variables"][0]["values"]
    columns = {dv["header"]["name"]: dv["values"] for dv in phenix["dependent_variables"]}
    npart = columns["$N_p$"]
    for observable, key in (("dnch_deta", "$dN_{ch}$/$d\\eta$"), ("dnch_deta_per_pair", "$dN_{ch}$/$d\\eta$/($0.5N_p$)")):
        for b, n, v in zip(bins, npart, columns[key]):
            rows.append({"observable": observable, "centrality_low": b["low"], "centrality_high": b["high"],
                         "n_part": n["value"], "dnch_deta": "", "value": v["value"], "error": f"{total_error(v):.4g}",
                         "label": "PHENIX", "source": "PRC 71 034908 (ins659749)"})
    star_v2 = table("ins1315466", "Table22.yaml")
    for b, v in zip(star_v2["independent_variables"][0]["values"], star_v2["dependent_variables"][0]["values"]):
        rows.append({"observable": "v2_2", "centrality_low": b["low"], "centrality_high": b["high"], "n_part": "",
                     "dnch_deta": "", "value": f"{float(v['value']) / 100.0:.5g}", "error": f"{total_error(v, 0.01):.3g}",
                     "label": "STAR sqrt<v2^2>, nonflow removed", "source": "PRC 88 014904 (ins1315466)"})
    star_pt = table("ins630160", "fig3-1.yaml")
    multiplicity = [float(v["value"]) for v in star_pt["independent_variables"][0]["values"]]
    for dv, observable in zip(star_pt["dependent_variables"], ("mean_pt_pion", "mean_pt_kaon", "mean_pt_proton")):
        for dnch, v in zip(multiplicity, dv["values"]):
            if dnch < 10.0:  # the first point is p+p
                continue
            rows.append({"observable": observable, "centrality_low": "", "centrality_high": "", "n_part": "",
                         "dnch_deta": dnch, "value": v["value"], "error": f"{total_error(v):.3g}",
                         "label": "STAR (pi-, K-, pbar)", "source": "PRL 92 112301 (ins630160)"})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
