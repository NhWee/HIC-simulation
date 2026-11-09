"""Render a flow-ensemble JSON (hic.flow_ensemble) as a self-contained HTML report.

    python -m hic.plot_flow_ensemble outputs/flow_ensemble.json --output outputs/flow_ensemble.html \\
        [--data measurements.csv]

The optional data CSV overlays measurements on the centrality panels; columns:
observable (v2_2 | v3_2 | dnch_deta_per_pair | mean_pt_pion | mean_pt_kaon | mean_pt_proton),
centrality_low, centrality_high, n_part (optional), value, error, label.
Colors follow the dataviz reference palette (fixed categorical order, separate dark-mode steps).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SURFACE = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "data": "#0b0b0b"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f", "data": "#f0efec"},
}


def class_label(summary: dict) -> str:
    return f"{summary['centrality_low']:g}–{summary['centrality_high']:g}%"


def load_data(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_panels(result: dict, data: list[dict]) -> list[dict]:
    """Panel specs: each is {title, x_title, y_title, series: [{name, x, y, error, kind, slot}]}."""
    summaries = [c["summary"] for c in result["classes"]]
    centrality = [0.5 * (s["centrality_low"] + s["centrality_high"]) for s in summaries]
    labels = [class_label(s) for s in summaries]
    n_part = [s["n_part"] for s in summaries]
    panels = []

    def measured(observable):
        rows = [r for r in data if r["observable"] == observable]
        if not rows:
            return []
        return [{
            "name": rows[0].get("label") or "data",
            "x": [0.5 * (float(r["centrality_low"]) + float(r["centrality_high"])) for r in rows],
            "y": [float(r["value"]) for r in rows],
            "error": [float(r.get("error") or 0.0) for r in rows],
            "kind": "data",
        }]

    per_pair = [s["dnch_deta"] / (0.5 * s["n_part"]) for s in summaries]
    per_pair_error = [s["dnch_deta_error"] / (0.5 * s["n_part"]) for s in summaries]
    data_rows = [r for r in data if r["observable"] == "dnch_deta_per_pair" and r.get("n_part")]
    panels.append({
        "title": "Charged multiplicity per participant pair",
        "x_title": "⟨N_part⟩", "y_title": "(dN_ch/dη) / (⟨N_part⟩/2)",
        "series": [{"name": "model", "x": n_part, "y": per_pair, "error": per_pair_error, "kind": "model", "slot": 0,
                    "text": labels}]
        + ([{"name": data_rows[0].get("label") or "data", "x": [float(r["n_part"]) for r in data_rows],
             "y": [float(r["value"]) for r in data_rows], "error": [float(r.get("error") or 0) for r in data_rows],
             "kind": "data"}] if data_rows else []),
    })
    panels.append({
        "title": "Two-particle cumulant flow",
        "x_title": "centrality (%)", "y_title": "v_n{2}",
        "series": [
            {"name": "v₂{2}", "x": centrality, "y": [s["v2_2"] for s in summaries],
             "error": [s["v2_2_error"] for s in summaries], "kind": "model", "slot": 0, "text": labels},
            {"name": "v₃{2}", "x": centrality, "y": [s["v3_2"] for s in summaries],
             "error": [s["v3_2_error"] for s in summaries], "kind": "model", "slot": 1, "text": labels},
        ] + measured("v2_2") + measured("v3_2"),
    })
    panels.append({
        "title": "Hydrodynamic response",
        "x_title": "centrality (%)", "y_title": "v_n{2} / ε_n{2}",
        "series": [
            {"name": "v₂{2}/ε₂{2}", "x": centrality, "y": [s["v2_2"] / s["eps2_2"] for s in summaries],
             "error": [s["v2_2_error"] / s["eps2_2"] for s in summaries], "kind": "model", "slot": 0, "text": labels},
            {"name": "v₃{2}/ε₃{2}", "x": centrality, "y": [s["v3_2"] / s["eps3_2"] for s in summaries],
             "error": [s["v3_2_error"] / s["eps3_2"] for s in summaries], "kind": "model", "slot": 1, "text": labels},
        ],
    })
    identified = []
    for slot, (key, name) in enumerate((("mean_pt_charged_pion_gev", "π±"), ("mean_pt_charged_kaon_gev", "K±"),
                                         ("mean_pt_proton_antiproton_gev", "p + p̄"))):
        identified.append({"name": name, "x": n_part, "y": [s[key] for s in summaries],
                           "error": [s[f"{key}_error"] for s in summaries], "kind": "model", "slot": slot, "text": labels})
    panels.append({"title": "Identified mean transverse momentum", "x_title": "⟨N_part⟩", "y_title": "⟨p_T⟩ (GeV)",
                   "series": identified})

    edges = summaries[0]["pt_edges"]
    centers = [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]
    panels.append({
        "title": "Charged-hadron v₂{2}(p_T)",
        "x_title": "p_T (GeV)", "y_title": "v₂{2}",
        "series": [{"name": class_label(s), "x": centers, "y": s["v2_charged_pt"], "error": s["v2_charged_pt_error"],
                    "kind": "model", "slot": slot} for slot, s in enumerate(summaries[:5])],
    })
    middle = summaries[len(summaries) // 2]
    panels.append({
        "title": f"Identified v₂{{2}}(p_T), {class_label(middle)}",
        "x_title": "p_T (GeV)", "y_title": "v₂{2}",
        "series": [{"name": name, "x": centers, "y": middle[f"v2_{key}_pt"], "error": middle[f"v2_{key}_pt_error"],
                    "kind": "model", "slot": slot}
                   for slot, (key, name) in enumerate((("pion", "π±"), ("kaon", "K±"), ("proton", "p + p̄")))],
    })
    return panels


def render(result: dict, data: list[dict], title: str) -> str:
    panels = build_panels(result, data)
    settings = result["settings"]
    events = sum(c["summary"]["events"] for c in result["classes"])
    failed = sum(len(c["summary"]["failed_audits"]) for c in result["classes"])
    total = sum(len(c["events"]) for c in result["classes"])
    rows = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in (
            class_label(s), s["events"], f"{s['n_part']:.0f}", f"{s['dnch_deta']:.0f} ± {s['dnch_deta_error']:.0f}",
            f"{s['eps2_2']:.3f}", f"{s['v2_2']:.4f} ± {s['v2_2_error']:.4f}", f"{s['eps3_2']:.3f}",
            f"{s['v3_2']:.4f} ± {s['v3_2_error']:.4f}",
            f"{s['mean_pt_charged_pion_gev']:.3f} / {s['mean_pt_charged_kaon_gev']:.3f} / {s['mean_pt_proton_antiproton_gev']:.3f}",
        )) + "</tr>"
        for s in (c["summary"] for c in result["classes"])
    )
    payload = json.dumps({"panels": panels, "palette": PALETTE, "surface": SURFACE})
    note = (
        f"{events} of {total} hydro events (those passing the physics audit; {failed} excluded), "
        f"global energy scale {settings['energy_scale']:g}, {settings['grid_size']}² grid over ±{settings['extent']:g} fm. "
        "Ideal hydrodynamics (no viscosity, no hadronic rescattering): v_n is expected to exceed measurements."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.0.1/plotly.min.js"></script>
<style>
:root {{ color-scheme: light; --surface: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --rule: #e4e3df; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --rule: #33322f; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --rule: #33322f; }}
body {{ margin: 0; background: var(--surface); color: var(--text); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} p.note {{ color: var(--muted); margin: 0 0 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
.panel {{ border: 1px solid var(--rule); border-radius: 8px; padding: 8px; min-width: 0; }}
.panel h2 {{ font-size: 14px; font-weight: 600; margin: 4px 8px; }}
.plot {{ height: 300px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 13px; font-variant-numeric: tabular-nums; }}
.table-wrap {{ overflow-x: auto; }}
th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ color: var(--muted); font-weight: 600; }}
</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="note">{html.escape(note)}</p>
<div class="grid" id="grid"></div>
<div class="table-wrap"><table>
<thead><tr><th>centrality</th><th>events</th><th>⟨N_part⟩</th><th>dN_ch/dη</th><th>ε₂{{2}}</th><th>v₂{{2}}</th><th>ε₃{{2}}</th><th>v₃{{2}}</th><th>⟨p_T⟩ π / K / p (GeV)</th></tr></thead>
<tbody>{rows}</tbody></table></div>
</main>
<script>
const spec = {payload};
function mode() {{
  const forced = document.documentElement.dataset.theme;
  if (forced === "dark" || forced === "light") return forced;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}}
function draw() {{
  const m = mode(), c = spec.surface[m], colors = spec.palette[m];
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  // Lay out every panel first, so each plot is sized from its final grid column.
  spec.panels.forEach((panel, i) => {{
    const box = document.createElement("section"); box.className = "panel";
    box.innerHTML = `<h2>${{panel.title}}</h2><div class="plot" id="p${{i}}"></div>`;
    grid.appendChild(box);
  }});
  spec.panels.forEach((panel, i) => {{
    const traces = panel.series.map(s => {{
      const isData = s.kind === "data";
      const color = isData ? c.data : colors[s.slot];
      return {{
        name: s.name, x: s.x, y: s.y, text: s.text, type: "scatter",
        mode: isData ? "markers" : "lines+markers",
        line: {{ color, width: 2 }},
        marker: {{ color: isData ? "rgba(0,0,0,0)" : color, size: 8, symbol: isData ? "square" : "circle",
                   line: {{ color: isData ? c.data : c.surface, width: 2 }} }},
        error_y: {{ type: "data", array: s.error, visible: true, color, thickness: 1.5, width: 3 }},
        hovertemplate: (s.text ? "%{{text}}<br>" : "") + "%{{x:.3g}}, %{{y:.4g}}<extra>" + s.name + "</extra>",
      }};
    }});
    const axis = t => ({{ title: {{ text: t, font: {{ color: c.muted, size: 12 }} }}, gridcolor: c.grid, zerolinecolor: c.grid,
                          linecolor: c.grid, tickfont: {{ color: c.muted, size: 11 }} }});
    Plotly.newPlot("p" + i, traces, {{
      paper_bgcolor: c.surface, plot_bgcolor: c.surface, font: {{ color: c.text, family: "system-ui, sans-serif" }},
      margin: {{ l: 56, r: 12, t: 8, b: 44 }}, xaxis: axis(panel.x_title), yaxis: axis(panel.y_title),
      showlegend: panel.series.length > 1, legend: {{ orientation: "h", y: 1.12, x: 0, font: {{ size: 11, color: c.text }} }},
      hovermode: "closest", hoverlabel: {{ bgcolor: c.surface, bordercolor: c.grid, font: {{ color: c.text }} }},
    }}, {{ displayModeBar: false, responsive: true }});
  }});
}}
draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ensemble", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Optional measurements CSV to overlay.")
    parser.add_argument("--title", default="Au+Au √s_NN = 200 GeV centrality ensemble")
    args = parser.parse_args()
    result = json.loads(args.ensemble.read_text())
    output = args.output or args.ensemble.with_suffix(".html")
    output.write_text(render(result, load_data(args.data), args.title), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
