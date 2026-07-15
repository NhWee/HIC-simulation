import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_SUMMARY = {
    "nPart": 268,
    "nColl": 638,
    "spectators": 148,
    "centralityProxy": 18.9,
    "peakTemperature": 0.33,
    "peakEps2": 0.21,
    "peakFlow": 0.58,
    "finalV2": 0.36,
    "stageCounts": {
        "incoming nuclei": 12,
        "pre-equilibrium": 4,
        "QGP hydrodynamics": 27,
        "mixed phase": 35,
        "hadron gas": 43,
    },
}


def read_visual_simulation_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return DEFAULT_SUMMARY

    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"const data = (.*?);\nconst frames", text)
    if not match:
        return DEFAULT_SUMMARY

    data = json.loads(match.group(1))
    frames = data.get("frames", [])
    settings = data.get("settings", {})
    if not frames:
        return DEFAULT_SUMMARY

    stage_counts = Counter(frame.get("stage", "unknown") for frame in frames)
    return {
        "nPart": settings.get("nPart", DEFAULT_SUMMARY["nPart"]),
        "nColl": settings.get("nColl", DEFAULT_SUMMARY["nColl"]),
        "spectators": settings.get("spectators", DEFAULT_SUMMARY["spectators"]),
        "centralityProxy": round(float(settings.get("centralityProxy", DEFAULT_SUMMARY["centralityProxy"])), 1),
        "peakTemperature": round(max(float(frame.get("maxTemperature", 0.0)) for frame in frames), 3),
        "peakEps2": round(max(float(frame.get("eps2", 0.0)) for frame in frames), 3),
        "peakFlow": round(max(float(frame.get("meanRadialFlow", 0.0)) for frame in frames), 3),
        "finalV2": round(float(frames[-1].get("emittedV2", 0.0)), 3),
        "stageCounts": dict(stage_counts),
    }


def build_html(summary: dict[str, object]) -> str:
    payload = json.dumps(summary, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heavy-Ion Collision Educational Visual</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #07100f;
  --panel: #101817;
  --line: #2a3a37;
  --text: #eef7f4;
  --muted: #a6b8b3;
  --cyan: #68d4e7;
  --red: #f2533f;
  --gold: #f6c453;
  --green: #9ee7c3;
  --violet: #b9a7ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  background: #07100f;
  color: var(--text);
  font-family: Inter, Segoe UI, Malgun Gothic, system-ui, sans-serif;
}}
.board {{
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.75fr);
  gap: 14px;
  padding: 14px;
}}
.panel {{
  border: 1px solid var(--line);
  background: var(--panel);
  padding: 14px;
}}
.title {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
  margin-bottom: 12px;
}}
h1 {{
  margin: 0;
  font-size: clamp(24px, 3vw, 42px);
  letter-spacing: 0;
}}
h2 {{
  margin: 0 0 10px;
  font-size: 18px;
  letter-spacing: 0;
}}
.subtitle {{
  color: var(--muted);
  font-size: 14px;
  margin-top: 6px;
}}
.stage-grid {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
}}
.stage {{
  min-height: 126px;
  border: 1px solid var(--line);
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.stage strong {{
  font-size: 14px;
}}
.stage span {{
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}}
.swatch {{
  width: 100%;
  height: 6px;
  background: var(--cyan);
}}
.main-visual {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 270px;
  gap: 12px;
  margin-top: 12px;
}}
canvas {{
  display: block;
  width: 100%;
  background: #020504;
  border: 1px solid var(--line);
}}
#collision {{
  height: 520px;
}}
.metrics {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}
.metric {{
  border: 1px solid var(--line);
  padding: 10px;
  min-height: 78px;
}}
.metric .label {{
  color: var(--muted);
  font-size: 12px;
}}
.metric .value {{
  margin-top: 8px;
  font-size: 24px;
  font-variant-numeric: tabular-nums;
}}
.explain {{
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
}}
.concepts {{
  display: grid;
  gap: 10px;
}}
.concept {{
  border-left: 3px solid var(--cyan);
  padding: 8px 10px;
  background: #0b1211;
}}
.concept strong {{
  display: block;
  margin-bottom: 4px;
}}
.concept p {{
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}}
.lab {{
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 12px;
  margin-top: 14px;
}}
.simulation {{
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 12px;
  margin-top: 14px;
}}
.control {{
  border: 1px solid var(--line);
  padding: 12px;
}}
input[type="range"] {{
  width: 100%;
}}
.reading {{
  display: grid;
  gap: 8px;
  margin-top: 10px;
}}
.row {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
}}
.row b {{
  color: var(--text);
  font-variant-numeric: tabular-nums;
}}
#labCanvas {{
  height: 230px;
}}
#simCanvas {{
  height: 300px;
}}
.button-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 12px;
}}
button {{
  border: 1px solid var(--line);
  background: #15211f;
  color: var(--text);
  min-height: 34px;
  cursor: pointer;
}}
button:hover {{
  border-color: var(--cyan);
}}
.phase-pill {{
  margin-top: 10px;
  border: 1px solid var(--line);
  padding: 9px;
  color: var(--text);
  font-size: 13px;
  text-align: center;
}}
.footer-note {{
  margin-top: 12px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}}
@media (max-width: 980px) {{
  .board, .main-visual, .lab, .simulation {{ grid-template-columns: 1fr; }}
  .stage-grid {{ grid-template-columns: 1fr; }}
  #collision {{ height: 420px; }}
}}
</style>
</head>
<body>
<main class="board">
  <section class="panel">
    <div class="title">
      <div>
        <h1>상대론적 중이온 충돌 한 장 요약</h1>
        <div class="subtitle">두 원자핵이 충돌해 QGP를 만들고, 팽창/냉각 후 입자로 freeze-out되는 흐름</div>
      </div>
    </div>
    <div class="stage-grid">
      <div class="stage"><div class="swatch" style="background:var(--cyan)"></div><strong>1. Incoming nuclei</strong><span>로런츠 수축된 두 핵이 impact parameter b를 가지고 접근한다.</span></div>
      <div class="stage"><div class="swatch" style="background:var(--violet)"></div><strong>2. Pre-equilibrium</strong><span>초기 에너지 밀도와 hot spot이 만들어지고 아직 완전한 유체 평형은 아니다.</span></div>
      <div class="stage"><div class="swatch" style="background:var(--red)"></div><strong>3. QGP hydrodynamics</strong><span>고온 QGP가 압력 구배 때문에 바깥으로 흐르며 공간 비대칭을 운동량 비대칭으로 바꾼다.</span></div>
      <div class="stage"><div class="swatch" style="background:var(--gold)"></div><strong>4. Mixed / freeze-out</strong><span>온도가 내려가며 hadronization과 kinetic freeze-out 표면이 생긴다.</span></div>
      <div class="stage"><div class="swatch" style="background:var(--green)"></div><strong>5. Hadron gas</strong><span>검출기는 최종 hadron의 분포, 평균 pT, v2 등을 본다.</span></div>
    </div>
    <div class="main-visual">
      <canvas id="collision"></canvas>
      <div>
        <h2>현재 예시 이벤트</h2>
        <div class="metrics">
          <div class="metric"><div class="label">Npart</div><div class="value" id="nPart">0</div></div>
          <div class="metric"><div class="label">Ncoll</div><div class="value" id="nColl">0</div></div>
          <div class="metric"><div class="label">centrality proxy</div><div class="value" id="centrality">0%</div></div>
          <div class="metric"><div class="label">spectators</div><div class="value" id="spectators">0</div></div>
          <div class="metric"><div class="label">peak T</div><div class="value" id="peakT">0</div></div>
          <div class="metric"><div class="label">final v2</div><div class="value" id="finalV2">0</div></div>
        </div>
        <p class="footer-note">숫자는 `outputs/visual_simulation.html`에서 읽은 최근 GPU 시뮬레이션 결과를 사용한다.</p>
      </div>
    </div>
    <div class="lab">
      <div class="control">
        <h2>Impact parameter 실험</h2>
        <input id="impact" type="range" min="0" max="12" step="0.1" value="5.7">
        <div class="reading">
          <div class="row"><span>b</span><b id="bRead">5.7 fm</b></div>
          <div class="row"><span>overlap</span><b id="overlapRead">0%</b></div>
          <div class="row"><span>eccentricity</span><b id="eccRead">0.00</b></div>
          <div class="row"><span>expected multiplicity</span><b id="multRead">0</b></div>
        </div>
      </div>
      <canvas id="labCanvas"></canvas>
    </div>
    <div class="simulation">
      <div class="control">
        <h2>단계 시뮬레이션</h2>
        <div class="button-row">
          <button id="simPlay">pause</button>
          <button id="simReset">reset</button>
        </div>
        <input id="simTime" type="range" min="0" max="1000" step="1" value="0">
        <div class="reading">
          <div class="row"><span>proper time</span><b id="simTau">0.0 fm/c</b></div>
          <div class="row"><span>temperature</span><b id="simTemp">0.00</b></div>
          <div class="row"><span>radial flow</span><b id="simFlow">0.00</b></div>
          <div class="row"><span>emitted hadrons</span><b id="simHadrons">0</b></div>
        </div>
        <div class="phase-pill" id="simPhase">incoming nuclei</div>
      </div>
      <canvas id="simCanvas"></canvas>
    </div>
  </section>
  <aside class="panel">
    <h2>핵심 개념 해석</h2>
    <div class="concepts">
      <div class="concept"><strong>Participants vs spectators</strong><p>겹친 영역에서 상호작용한 핵자는 participants, 지나쳐 나간 핵자는 spectators다. Participants가 많을수록 더 central한 충돌이다.</p></div>
      <div class="concept"><strong>Ncoll</strong><p>핵자-핵자 충돌 수 proxy다. hard process나 binary scaling을 설명할 때 자주 쓰인다.</p></div>
      <div class="concept"><strong>eps2 -> v2</strong><p>초기 겹침 영역이 타원형이면 압력 구배가 방향마다 달라진다. 이 공간 비대칭 eps2가 최종 입자의 elliptic flow v2로 이어진다.</p></div>
      <div class="concept"><strong>Temperature contours</strong><p>T가 critical temperature 아래로 내려가면 QGP 지배 그림에서 hadronic matter와 freeze-out 표면으로 넘어간다.</p></div>
      <div class="concept"><strong>Freeze-out</strong><p>입자들이 더 이상 충분히 산란하지 않고 검출 가능한 최종 상태로 날아가는 경계다.</p></div>
    </div>
    <h2 style="margin-top:16px">단계별 frame 비중</h2>
    <canvas id="stageChart" style="height:210px"></canvas>
    <p class="explain">이 자료는 정량 예측용이 아니라 수업용 개념 시각화다. 실제 연구에서는 calibrated initial condition, viscous relativistic hydrodynamics, EoS table, Cooper-Frye particlization이 필요하다.</p>
  </aside>
</main>
<script>
const summary = {payload};

function resizeCanvas(canvas) {{
  const box = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(box.width * ratio));
  canvas.height = Math.max(1, Math.floor(box.height * ratio));
}}

function drawCollision() {{
  const canvas = document.getElementById("collision");
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#020504";
  ctx.fillRect(0, 0, w, h);
  const cx = w * 0.50;
  const cy = h * 0.53;
  const r = Math.min(w, h) * 0.25;
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.35);
  grad.addColorStop(0, "rgba(242,83,63,0.92)");
  grad.addColorStop(0.45, "rgba(246,196,83,0.78)");
  grad.addColorStop(0.72, "rgba(158,231,195,0.42)");
  grad.addColorStop(1, "rgba(20,40,38,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(cx, cy, r * 1.32, r * 0.92, -0.18, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,246,214,0.70)";
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  ctx.ellipse(cx, cy, r * 0.88, r * 0.50, -0.18, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(246,196,83,0.70)";
  ctx.beginPath();
  ctx.ellipse(cx, cy, r * 1.12, r * 0.72, -0.18, 0, Math.PI * 2);
  ctx.stroke();
  for (let i = 0; i < 120; i++) {{
    const a = i * 2.399;
    const rr = r * (0.2 + (i % 23) / 19);
    const x = cx + Math.cos(a) * rr * 1.45;
    const y = cy + Math.sin(a) * rr * 0.95;
    const color = i % 5 === 0 ? "rgba(246,196,83,0.70)" : "rgba(158,231,195,0.35)";
    ctx.fillStyle = color;
    ctx.fillRect(x, y, 3 * (window.devicePixelRatio || 1), 3 * (window.devicePixelRatio || 1));
  }}
  ctx.strokeStyle = "rgba(238,247,244,0.55)";
  ctx.setLineDash([10 * (window.devicePixelRatio || 1), 8 * (window.devicePixelRatio || 1)]);
  ctx.beginPath();
  ctx.moveTo(w * 0.17, h * 0.78);
  ctx.lineTo(w * 0.82, h * 0.28);
  ctx.stroke();
  ctx.setLineDash([]);
  drawNucleus(ctx, w * 0.20, h * 0.42, r * 0.70, "rgba(104,212,231,0.70)");
  drawNucleus(ctx, w * 0.82, h * 0.58, r * 0.70, "rgba(104,212,231,0.58)");
}}

function drawNucleus(ctx, x, y, r, color) {{
  const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
  grad.addColorStop(0, color);
  grad.addColorStop(1, "rgba(104,212,231,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(x, y, r * 0.42, r, 0, 0, Math.PI * 2);
  ctx.fill();
}}

function drawStageChart() {{
  const canvas = document.getElementById("stageChart");
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const counts = summary.stageCounts;
  const keys = ["incoming nuclei", "pre-equilibrium", "QGP hydrodynamics", "mixed phase", "hadron gas"];
  const colors = ["#68d4e7", "#b9a7ff", "#f2533f", "#f6c453", "#9ee7c3"];
  const max = Math.max(...keys.map(k => counts[k] || 0), 1);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  keys.forEach((key, i) => {{
    const x = 12 * devicePixelRatio;
    const y = (18 + i * 35) * devicePixelRatio;
    const width = ((counts[key] || 0) / max) * (canvas.width - 125 * devicePixelRatio);
    ctx.fillStyle = colors[i];
    ctx.fillRect(x, y, width, 18 * devicePixelRatio);
    ctx.fillStyle = "#eef7f4";
    ctx.font = `${{12 * devicePixelRatio}}px Segoe UI`;
    ctx.fillText(`${{key}} (${{counts[key] || 0}})`, x + width + 8 * devicePixelRatio, y + 14 * devicePixelRatio);
  }});
}}

function drawLab() {{
  const b = Number(document.getElementById("impact").value);
  const canvas = document.getElementById("labCanvas");
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const r = Math.min(w, h) * 0.28;
  const offset = (b / 12) * r * 1.45;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#020504";
  ctx.fillRect(0, 0, w, h);
  drawDisc(ctx, w * 0.5, h * 0.5 - offset, r, "rgba(104,212,231,0.48)");
  drawDisc(ctx, w * 0.5, h * 0.5 + offset, r, "rgba(246,196,83,0.40)");
  const overlap = Math.max(0, 1 - b / 12);
  const ecc = Math.min(0.85, b / 12 * 1.05);
  const mult = Math.round(900 * overlap * overlap + 70);
  document.getElementById("bRead").textContent = `${{b.toFixed(1)}} fm`;
  document.getElementById("overlapRead").textContent = `${{Math.round(overlap * 100)}}%`;
  document.getElementById("eccRead").textContent = ecc.toFixed(2);
  document.getElementById("multRead").textContent = String(mult);
}}

function drawDisc(ctx, x, y, r, color) {{
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}}

let simProgress = 0;
let simRunning = true;
let lastSimStamp = 0;
const simParticles = Array.from({{ length: 170 }}, (_, i) => {{
  const angle = i * 2.399963;
  const radius = 0.18 + ((i * 37) % 100) / 100;
  return {{
    a: angle,
    r: radius,
    speed: 0.35 + ((i * 19) % 100) / 210,
    size: 1.1 + ((i * 13) % 100) / 70
  }};
}});

function simStage(p) {{
  if (p < 0.20) return "incoming nuclei";
  if (p < 0.32) return "pre-equilibrium";
  if (p < 0.56) return "QGP hydrodynamics";
  if (p < 0.76) return "mixed / freeze-out";
  return "hadron gas";
}}

function drawEducationalSimulation() {{
  const canvas = document.getElementById("simCanvas");
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const p = simProgress;
  const stage = simStage(p);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#020504";
  ctx.fillRect(0, 0, w, h);
  const cx = w * 0.5;
  const cy = h * 0.52;
  const unit = Math.min(w, h);
  const nuclearR = unit * 0.22;
  const approach = Math.max(0, 1 - p / 0.28);
  const pass = Math.max(0, (p - 0.28) / 0.72);
  const sep = unit * (0.50 * approach + 0.35 * pass);
  const impact = unit * 0.12;
  const leftX = cx - sep;
  const rightX = cx + sep;
  const opacity = Math.max(0.08, 1 - p * 1.4);
  drawLorentzNucleus(ctx, leftX, cy - impact, nuclearR, opacity);
  drawLorentzNucleus(ctx, rightX, cy + impact, nuclearR, opacity);

  const fire = Math.max(0, Math.min(1, (p - 0.18) / 0.24)) * Math.max(0, 1 - Math.max(0, p - 0.72) / 0.45);
  const mixed = Math.max(0, 1 - Math.abs(p - 0.66) / 0.18);
  const hadron = Math.max(0, Math.min(1, (p - 0.62) / 0.26));
  if (fire > 0.01) {{
    const r = unit * (0.10 + 0.28 * Math.min(1, p * 1.35));
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.35);
    grad.addColorStop(0, `rgba(242, 83, 63, ${{0.78 * fire}})`);
    grad.addColorStop(0.48, `rgba(246, 196, 83, ${{0.55 * Math.max(fire, mixed)}})`);
    grad.addColorStop(0.82, `rgba(158, 231, 195, ${{0.34 * Math.max(mixed, hadron)}})`);
    grad.addColorStop(1, "rgba(12, 24, 22, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(cx, cy, r * (1.10 + 0.35 * p), r * (0.72 + 0.18 * p), -0.20, 0, Math.PI * 2);
    ctx.fill();
    drawContour(ctx, cx, cy, r * 0.72, -0.20, "rgba(255,246,214,0.70)");
    drawContour(ctx, cx, cy, r * 1.02, -0.20, "rgba(246,196,83,0.68)");
  }}

  if (p > 0.34 && p < 0.84) {{
    ctx.strokeStyle = "rgba(238,247,244,0.42)";
    ctx.lineWidth = 1.2 * devicePixelRatio;
    for (let i = 0; i < 32; i++) {{
      const a = i * 2.399963;
      const rr = unit * (0.08 + ((i * 23) % 100) / 330);
      const x = cx + Math.cos(a) * rr * (1.25 + p * 0.3);
      const y = cy + Math.sin(a) * rr * 0.85;
      const len = unit * (0.025 + 0.030 * p);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len);
      ctx.stroke();
    }}
  }}

  if (hadron > 0.02) {{
    for (const particle of simParticles) {{
      const radius = unit * (0.05 + particle.r * 0.34 * hadron);
      const x = cx + Math.cos(particle.a) * radius * particle.speed * 1.35;
      const y = cy + Math.sin(particle.a) * radius * particle.speed * 0.92;
      ctx.fillStyle = `rgba(158, 231, 195, ${{0.18 + 0.72 * hadron}})`;
      ctx.beginPath();
      ctx.arc(x, y, particle.size * devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}

  const tau = Math.max(0, (p - 0.20) * 14.0);
  const heatUp = Math.max(0, Math.min(1, (p - 0.18) / 0.18));
  const temp = Math.max(0.03, 0.55 * heatUp * Math.exp(-Math.max(0, p - 0.34) * 2.2));
  const flow = Math.max(0, Math.min(0.75, (p - 0.30) * 1.25));
  document.getElementById("simTau").textContent = `${{tau.toFixed(1)}} fm/c`;
  document.getElementById("simTemp").textContent = temp.toFixed(2);
  document.getElementById("simFlow").textContent = flow.toFixed(2);
  document.getElementById("simHadrons").textContent = String(Math.round(hadron * 170));
  document.getElementById("simPhase").textContent = stage;
  document.getElementById("simTime").value = String(Math.round(p * 1000));
}}

function drawLorentzNucleus(ctx, x, y, r, opacity) {{
  const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
  grad.addColorStop(0, `rgba(104,212,231,${{0.70 * opacity}})`);
  grad.addColorStop(1, "rgba(104,212,231,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(x, y, r * 0.34, r, 0, 0, Math.PI * 2);
  ctx.fill();
}}

function drawContour(ctx, x, y, r, angle, color) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4 * devicePixelRatio;
  ctx.beginPath();
  ctx.ellipse(x, y, r * 1.32, r * 0.72, angle, 0, Math.PI * 2);
  ctx.stroke();
}}

function tickSimulation(stamp) {{
  if (simRunning && stamp - lastSimStamp > 28) {{
    simProgress = (simProgress + 0.0038) % 1;
    lastSimStamp = stamp;
    drawEducationalSimulation();
  }}
  requestAnimationFrame(tickSimulation);
}}

document.getElementById("nPart").textContent = summary.nPart;
document.getElementById("nColl").textContent = summary.nColl;
document.getElementById("spectators").textContent = summary.spectators;
document.getElementById("centrality").textContent = `${{summary.centralityProxy}}%`;
document.getElementById("peakT").textContent = summary.peakTemperature;
document.getElementById("finalV2").textContent = summary.finalV2;
document.getElementById("impact").addEventListener("input", drawLab);
document.getElementById("simPlay").addEventListener("click", () => {{
  simRunning = !simRunning;
  document.getElementById("simPlay").textContent = simRunning ? "pause" : "play";
}});
document.getElementById("simReset").addEventListener("click", () => {{
  simProgress = 0;
  simRunning = true;
  document.getElementById("simPlay").textContent = "pause";
  drawEducationalSimulation();
}});
document.getElementById("simTime").addEventListener("input", event => {{
  simProgress = Number(event.target.value) / 1000;
  simRunning = false;
  document.getElementById("simPlay").textContent = "play";
  drawEducationalSimulation();
}});
window.addEventListener("resize", () => {{
  drawCollision();
  drawStageChart();
  drawLab();
  drawEducationalSimulation();
}});
drawCollision();
drawStageChart();
drawLab();
drawEducationalSimulation();
requestAnimationFrame(tickSimulation);
</script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an educational heavy-ion collision visual board.")
    parser.add_argument("--simulation", type=Path, default=Path("outputs/visual_simulation.html"), help="Optional visual simulation HTML to summarize.")
    parser.add_argument("--output", type=Path, default=Path("outputs/hic_educational_visual.html"), help="Output educational HTML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_visual_simulation_summary(args.simulation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(summary), encoding="utf-8")
    print(f"Saved educational visual: {args.output}")


if __name__ == "__main__":
    main()
