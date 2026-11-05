# 보완점과 Next Step (2026-11-05)

현재 코드(`85718db`)를 읽고, 기준 이벤트(Au+Au, b = 5.7 fm, seed 17)를 CPU에서
다시 돌려 확인한 결과입니다. 각 항목에 **[확인]**(코드/수치로 검증),
**[추정]**(근거는 있으나 미검증) 표시를 붙였습니다.

## 핵심 진단: v₂가 왜 0.012인가

README의 기준값 v₂ = 0.0119는 20–30% 중심도 Au+Au에 대한 ideal hydro 기대치
(대략 0.05 내외)보다 크게 작습니다. 원인을 하나씩 분리해 측정했습니다.

| 가설 | 시험 방법 | 결과 | 판정 |
| --- | --- | --- | --- |
| 1차 Rusanov 수치 확산 | 격자 80/120/160/220 | v₂ ≈ 0.010–0.014, 격자 120 이상 통계오차 내 수렴 | 주원인 아님 |
| Bjorken 소스항 분리 오류 (아래 P0-3) | 정확한 Milne 소스로 교체 | v₂ 0.0113 → 0.0138, ⟨pT⟩ +3.5% (격자 160) | 부분 원인 (~20%) |
| 동결면 `initially_hot` 마스크 (P0-4) | 마스크 제거 | 1567개 중 6개 셀 추가, 관측량 변화 없음 | 무시 가능 |
| **초기 에너지 밀도 상한 6 GeV/fm³** | EoS 표에 conformal tail 추가 후 peak 변경 | 아래 표 | **주원인** |

| peak ε (GeV/fm³) | 수명 τ_end (fm/c) | ε₂ 최종 | ⟨pT⟩ (GeV) | v₂ |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2.6 | 0.170 | 0.415 | 0.004 ± 0.002 |
| 6 (현재) | 4.5 | 0.111 | 0.461 | 0.014 ± 0.002 |
| 12 | 5.4 | 0.083 | 0.514 | 0.019 ± 0.001 |
| 24 | 6.4 | 0.058 | 0.586 | 0.032 ± 0.002 |

(격자 120, Milne 소스 수정 적용, CPU. v₂ 오차는 Cooper-Frye 재샘플 8회의 표준오차.)

결론적으로 fireball이 너무 묽어서 약 4.5 fm/c 만에 동결되고, 흐름이 자랄
시간이 부족합니다. 이 상한은 `data/eos_table.csv`가 6 GeV/fm³에서 끝나기 때문에
`blender_event_pipeline.py:966`에 강제로 걸려 있습니다.

## P0 — 물리적 정확성 (먼저 고칠 것)

### P0-1. EoS 표 교체 [확인]
`data/eos_table.csv`는 열역학적으로 일관되지 않습니다.

- 표의 `sound_speed2`가 `dp/de`(유한차분)와 10–20% 다릅니다. 예를 들어
  0.6–1.0 구간에서 0.205와 0.250, 2.5–4.0 구간에서 0.31과 0.367입니다.
  `pressure`와 `sound_speed2`를 따로 보간하기 때문에 solver 안에서 서로 모순됩니다.
- s = (e+p)/T라 놓으면 `ds/de`와 `1/T`가 일치하지 않습니다(예: 0.05–0.10 구간 9.1 vs 10.3).
- e/T⁴가 고온으로 갈수록 5.9에서 2.5로 **감소**합니다. 격자 QCD(HotQCD,
  arXiv:1407.6387)에서는 T ≈ 300–400 MeV에서 e/T⁴ ≈ 12–13으로 증가하고,
  Stefan–Boltzmann 극한(2+1 flavor)은 15.6입니다(3단계에서 파라미터화로 확인: 300 MeV에서 12.2, 400 MeV에서 13.4).
  따라서 "최대 온도 612 → 370 MeV" 같은 온도값이 실제 e에 비해 과대평가되어 있습니다.

조치: HotQCD 격자 EoS와 HRG를 매끄럽게 접합한 표(s95p 계열이나 HotQCD+HRG)로
교체합니다. p(T)에서 e, s, c_s²를 **미분으로 일관되게** 생성하고, ε ≳ 100 GeV/fm³까지
확장합니다. `hydro_validation`에는 열역학 일관성 검사(`e + p = Ts`,
`c_s² = dp/de`)를 추가합니다.

### P0-2. 초기 조건 정규화를 데이터에 맞추기 [확인]
`--initial-peak-energy`가 고정값(6 GeV/fm³)이고, 출력 입자 수도 고정(900)입니다.

조치:
- Cooper-Frye 적분 N_i = ∫ n_i(T) u·dσ으로 평균 다중도를 계산하고, 샘플 수를
  Poisson(N)으로 정합니다.
- 정규화 상수 하나를 0–5% 중심도의 측정 dN_ch/dη(PHOBOS/STAR)에 맞춰 튜닝합니다.
  나머지 중심도는 **예측**으로 남겨 튜닝과 예측을 구분합니다.
- P0-1이 끝나야 상한을 풀 수 있습니다.

### P0-3. Bjorken 소스항을 정확한 Milne 형태로 교체 [확인]
`ideal_hydro_2d.apply_bjorken_cooling`은 유체 속도 v를 고정한 채
de/dτ = −(e+p)/τ로 국소 에너지 밀도만 식힌 뒤 T^τμ를 다시 조립합니다.
boost-invariant ideal hydro의 정확한 소스항은 다음과 같습니다.

```
∂τ T^ττ + ∂i T^iτ = −(T^ττ + p)/τ
∂τ T^τi + ∂j T^ij = −T^τi/τ
```

현재 방식에서 T^τi의 감쇠율은 −(1 + c_s²)·T^τi/τ가 되어, 움직이는 셀의 횡운동량을
약 33%(c_s² = 1/3 기준) 더 빨리 잃습니다. 현재 검증은 정지한 균일 Bjorken만
시험하므로 이 오류를 잡지 못합니다.

조치: 위 소스항을 RK2로 직접 적분합니다(진단에 쓴 구현은 약 10줄입니다).
**Gubser flow** 해석해(conformal ideal hydro, 횡팽창과 종팽창을 동시에 포함)를
검증 세트에 추가합니다. 이것이 2+1D 코드의 표준 시험이며 이번 오류도 잡았을 것입니다.

### P0-4. 동결면 마스크 정리 [확인, 저우선]
`evolve_hydrodynamics`의 `crossed = initially_hot & ...` 조건 때문에, τ₀ 이후에
T_sw를 넘었다가 식는 셀이 빠집니다. 영향은 기준 이벤트에서 무시할 수준(셀 0.4%)이지만,
초기 에너지를 키우면 늘어나므로 P0-2와 함께 정리합니다. 장기적으로는
CORNELIUS 방식의 3D(τ, x, y) 등온면 추출로 바꿉니다.

## P1 — 수치 기법과 관측량

### P1-1. 2차 정확도 solver [추정]
`rusanov_step`은 공간 1차(재구성 없음)와 시간 1차(forward Euler)입니다. 이번 시험에서
v₂는 수렴했지만, 점성 hydro를 넣으면 수치 점성이 물리 점성(η/s)과 섞여 구분할 수 없게
됩니다. **P2-1 전에** MUSCL(minmod) 재구성과 SSP-RK2로 Kurganov–Tadmor 방식을 구현하고,
Gubser 해에 대해 수렴 차수(≈2)를 측정해 `hydro_convergence`에 기록합니다.

### P1-2. 흐름 관측량 정의 [확인]
- `v3`를 반응평면(고정축) 기준으로 계산하므로, 대칭상 0에 가까운 의미 없는 값이
  나옵니다. participant plane Ψₙ 기준이나 2-입자 cumulant vₙ{2}로 바꿉니다.
- v₂{2}, pT-미분 vₙ(pT), 종별 스펙트럼(π/K/p)을 이벤트 앙상블(중심도당 ≥ 100 이벤트)로
  계산합니다. `event_pipeline` / `analyze_pipeline`을 기준 파이프라인과 통합합니다.
- 응답계수 v₂/ε₂를 기록하는 audit 항목을 추가합니다(현재 기준 약 0.06).

## P2 — 물리 확장 (P0/P1 이후)

1. **2차 점성 hydro**(MIS/DNMR, η/s ≈ 0.08–0.2, ζ/s)
2. **공명 붕괴**: 현재는 직접 π/K/p만 있어, 붕괴 π가 ⟨pT⟩와 v₂(pT)에 주는 효과가 빠져 있습니다.
   PDG 공명 목록을 넣은 thermal sampler와 붕괴를 추가합니다.
3. **hadronic afterburner**: SMASH 또는 UrQMD 연동(입출력 포맷만 맞추면 됩니다).
4. **pre-equilibrium**: 무충돌 자유흐름을 KøMPøST(선형응답)로 확장하거나 T_R ENTo 초기조건 옵션을 추가합니다.
5. 장기적으로는 Bayesian 보정으로 ε 정규화, η/s, T_sw를 데이터에 맞춥니다.

## E — 엔지니어링

| 항목 | 현황 | 조치 |
| --- | --- | --- |
| 자동 테스트 | 없음 (검증은 수동 실행 스크립트) | `pytest`로 `hydro_validation`, 새 Gubser/EoS 검사, 이벤트 audit를 래핑 |
| CI | 없음 | GitHub Actions (Linux, CPU torch, 격자 80–100). 기준 이벤트가 CPU에서 수 초면 끝나므로 PR마다 실행 가능 |
| 플랫폼 | `run_*.ps1` Windows 전용 | `pyproject.toml`과 `python -m hic.<module>` 진입점(또는 console scripts)을 추가하고, ps1은 얇은 래퍼로 유지 |
| 회귀 기준값 | README/REPRODUCIBILITY에 수기 기록 | `tests/reference_audit.json`과 허용오차로 자동 비교. P0 수정 후 기준값을 갱신 |
| 모듈 크기 | `pyside_simulator_app.py` 1231줄, `blender_event_pipeline.py` 999줄 | 물리 코어(`hic/physics/`)와 앱/시각화를 분리하고, 시각화 앱이 같은 코어를 쓰도록 정리 |
| 중복 코드 | Woods–Saxon 핵자 샘플러가 `nuclear_collision_2d`, `visual_simulation`, `blender_event_pipeline` 세 곳에 따로 구현됨 [확인] | 공통 모듈로 통합해 Glauber 기하가 도구마다 달라지지 않게 함 |

## 실행 순서 (제안)

| 단계 | 작업 | 완료 기준 |
| --- | --- | --- |
| 1 ✅ | pytest + CI 골격, 현재 동작을 기준값으로 고정 | CI 녹색. 이후 변경의 효과를 숫자로 볼 수 있음 (아래 "1단계 결과") |
| 2 ✅ | P0-3 Milne 소스 + Gubser 검증 | Gubser 1차 수렴 확인, 격자 160에서 L∞ 15% (1% 목표는 6단계로 이동; 아래 "2단계 결과") |
| 3 ✅ | P0-1 격자 QCD EoS | 일관성 검사 통과, 표 범위 ≥ 100 GeV/fm³ (181까지; 아래 "3단계 결과") |
| 4 | **P0-4 먼저**, 그다음 P0-2 정규화 | P0-4 xfail 테스트 통과(동결면 누락 셀 0). 0–5% dN_ch/dη 재현, 입자 수가 Cooper-Frye 적분에서 결정 |
| 5 | P1-2 관측량 + 중심도 앙상블 | v₂{2}(centrality)를 STAR/PHENIX 데이터와 비교한 그림 (ideal hydro이므로 과대 예측이 정상) |
| 6 | P1-1 KT solver | Gubser 수렴 차수 ≈ 2, 격자 160에서 L∞ < 1~2%. `--gubser-tolerance` 기본값 축소 |
| 7 | P2-1 점성 hydro | η/s 스캔 시 v₂가 단조 감소, Gubser 점성 해 검증 |
| 8 | P2-2/3 공명 붕괴, afterburner | 종별 ⟨pT⟩, p/π 비 비교 |

1–4단계가 끝나면 README의 기준값(온도, v₂, ⟨pT⟩)을 새로 생성하고, Blender 장면도 그
캐시로 다시 렌더링해야 합니다.

## 1단계 결과 (완료)

- `python -m pytest`: 16 passed, 4 xfailed, CPU 약 25초. CI는 `.github/workflows/tests.yml`에서
  Ubuntu(Python 3.11/3.13)와 Windows(Python 3.13)로 돌립니다.
- 기준 이벤트 회귀: `tests/reference/blender_event_audit.json`에 관측량별 허용오차와 함께
  고정했습니다. CPU 값이 README의 CUDA 값과 일치합니다(QGP 셀 3524, 동결 셀 4039,
  ⟨pT⟩ 0.4491, v₂ 0.0119). binary fraction을 0.14에서 0.16으로 바꾸는 뮤테이션을 4개 항목에서 검출했습니다.
- 알려진 버그 고정(`tests/test_known_issues.py`, `xfail(strict=True, raises=AssertionError)`):
  - P0-3 Bjorken 소스: τ·T^τx 보존 시험. 현재 0.794배로, 2^(−1/3)과 같아 (1+c_s²) 초과 감쇠를 확인했습니다.
  - P0-1 EoS: c_s² ≠ dp/de(최대 18%), dp/dT ≠ s(최대 45%), 표 범위가 6 GeV/fm³까지뿐임.
- 이 과정에서 발견해 고친 버그: `hydro_convergence`가 `crossover_bjorken_tolerance`를
  전달하지 않아 README의 `run_hydro_convergence.ps1` 명령이 `AttributeError`로 죽었습니다.
- 남은 공백: shock tube에 해석해(Riemann) 비교가 없어 에너지 보존만 검사합니다.
  Blender 장면 빌더(`tools/blender_hic_scene.py`)와 GUI 앱은 bpy/Qt가 필요해 테스트 대상에서 뺐습니다.

## 2단계 결과 (완료)

- `ideal_hydro_2d.apply_bjorken_cooling`을 정확한 Milne 소스항으로 교체했습니다.
  τ·T^τi는 해석적으로 보존하고, T^ττ는 s = ln τ에 대해 RK4(Δs ≤ 0.02)로 적분하며,
  매 stage마다 (T^ττ, T^τi)에서 압력을 복원합니다. 이 함수를 쓰는 기준 이벤트,
  `ideal_hydro_2d`, PySide 앱에 모두 적용됩니다.
- Gubser flow 검증(`hydro_validation`의 `gubser_flow`, q = 1 fm⁻¹, τ 1 → 2 fm/c, r ≤ 3 fm)을 추가했습니다.
  해석해 공식 자체도 유한차분으로 Milne 방정식 잔차 약 10⁻⁸임을 테스트에서 확인합니다.

| 격자 (±8 fm) | 새 소스: e L∞ 상대오차 | v L∞ | 기존 소스: e | v |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 0.284 | 0.045 | 0.545 | 0.117 |
| 160 | 0.154 | 0.024 | 0.432 | 0.099 |
| 320 | 0.081 | 0.012 | 0.372 | 0.096 |

  기존 소스는 격자를 늘려도 약 37%에서 멈추는 **수렴하지 않는 체계적 오류**였습니다.
  새 소스는 1차로 수렴합니다. τ 1 → 1.2 fm/c만 보면 격자 160에서 2.4%입니다.
  남은 오차는 1차 Rusanov의 수치 확산이므로, **1% 목표는 6단계(KT solver)의 완료 기준으로 옮겼습니다.**
- 기준 이벤트 변화: ⟨pT⟩ 0.449 → 0.466 GeV (+3.8%), hydro 스냅샷 50 → 52(수명 증가).
  v₂는 단일 샘플 기준 0.0119 → 0.0111로, Monte Carlo 오차(약 0.002) 안입니다.
  진단 때 8회 평균으로 본 v₂ 증가(약 +20%)와 모순되지 않습니다.
  `tests/reference/blender_event_audit.json`, README, REPRODUCIBILITY를 갱신했습니다.
- 부수 효과: `bjorken_crossover_convergence` 오차가 6.7×10⁻⁴에서 1.2×10⁻⁷로 줄었습니다.
  `hydro_convergence`는 검증 parser의 기본값을 물려받도록 바꿔, 1단계에서 본 "인자 누락" 버그가 재발하지 않게 했습니다.
- 남은 일: `docs/assets`의 렌더 이미지는 수정 전 물리로 만든 것입니다(Blender가 필요해 여기서 재렌더링하지 못함).

## 3단계 결과 (완료)

- `data/eos_table.csv`를 HotQCD (2+1)-flavor 격자 EoS로 교체했습니다(Bazavov et al., PRD 90, 094503 (2014),
  arXiv:1407.6387의 p/T⁴ 파라미터화). 생성기는 `hic/lattice_eos.py`이며, s = dp/dT, e = Ts − p,
  c_s² = s/(T ds/dT)로 모든 열을 만들어 **구성상 열역학적으로 일관**됩니다.
  T = 70–560 MeV, e = 0.0019–181 GeV/fm³ 범위의 491행이고, `entropy_density` 열을 추가했습니다(4단계 다중도 정규화용).
- **주의 — 파라미터 출처:** 이 세션에서는 arXiv 접근이 네트워크 정책으로 막혀 있어서, 계수를 원문과 대조하지 못하고
  기억에서 옮겨 적었습니다. 대신 아래의 독립 기준을 테스트(`tests/test_eos.py`)로 고정했고, 모두 통과합니다.
  그래도 **원문 표와 한 번은 대조해야 합니다.**

| 검사 | 기준 | 생성된 표 |
| --- | --- | --- |
| ε(145 MeV), ε(163 MeV) | 논문 초록: 0.18–0.5 GeV/fm³ | 0.204, 0.512 |
| trace anomaly 최대 | 약 4, T ≈ 200 MeV | 4.05, 201 MeV |
| c_s² 최소 | 약 0.14–0.15, T_c 부근 | 0.144, 147 MeV |
| p_id | SB 2+1 flavor 47.5π²/90 | 5.209 |
| e/T⁴ (300 MeV) | 증가 추세, 약 12 | 12.2 (기존 표는 2.5로 감소) |

- 표 밖 외삽 버그 수정(`EquationOfState.pressure`): 기존에는 표 범위 밖 압력을 끝값으로 고정해서,
  진공 셀(e = 10⁻⁸)이 p = 0.006 GeV/fm³(p > e)을 가졌습니다. 이제 아래쪽은 p = (p₀/e₀)e로 0에 수렴하고,
  위쪽은 마지막 c_s²로 연장합니다. 6 GeV/fm³ 하드코딩 상한도 표 최대값으로 바꿨습니다.
- 기준 이벤트(peak 6 GeV/fm³ 그대로): hydro 시작 T 370 → 254 MeV, 수명 4.5 → 6.2 fm/c,
  ⟨pT⟩ 0.466 → 0.474 GeV, 동결 셀 4039 → 4481.
  v₂는 단일 샘플 0.0076으로, 8회 평균(격자 120)으로 비교하면 기존 표 0.0131 ± 0.0010, 새 표 0.0127 ± 0.0015로 차이가 없습니다.
- **새로 드러난 문제 — P0-4가 이제 v₂의 주된 억제 요인:**

| peak ε (GeV/fm³) | T₀ (MeV) | 수명 (fm/c) | 유체 ε_p 최종 | v₂ (8회 평균) | τ₀ 이후 뜨거워진 셀 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 254 | 6.1 | 0.063 | 0.013 ± 0.002 | 37 / 2559 (1.4%) |
| 12 | 295 | 7.5 | — | 0.017 ± 0.001 | — |
| 24 | 346 | 8.7 | 0.083 | 0.018 ± 0.002 | 351 / 3347 (10%) |
| 48 | 407 | 9.9 | 0.092 | −0.003 ± 0.002 | 916 / 4091 (22%) |

  유체의 운동량 비등방성 ε_p(= ⟨T^yy − T^xx⟩/⟨T^yy + T^xx⟩)는 에너지와 함께 정상적으로 커지므로 hydro는 문제가 없습니다.
  그런데 v₂는 peak 48에서 0으로 무너집니다. 격자 경계는 원인이 아닙니다(±20 fm로 넓혀도 동일, 동결면 r_max ≈ 8 fm).
  원인은 흐름이 가장 강한 바깥 셀이 `initially_hot` 마스크로 동결면에서 빠지는 것입니다.
  `tests/test_known_issues.py`에 xfail로 고정했고(peak 24, 격자 60에서 840개 중 89개 누락), **4단계에서 정규화보다 먼저 고칩니다.**
- **원인 미확인:** peak 6에서도 v₂/ε_p ≈ 0.2로, 통상의 ideal hydro(약 0.5)보다 작습니다. 수치 확산(2단계 격자 수렴),
  경계, P0-4(이 에너지에서 셀 1.4%)는 배제했습니다. 다음 후보는 동결면 법선(τ_f(x,y) 그래프의 유한차분)과
  Cooper-Frye 가중치이며, 4–5단계에서 pT-미분 v₂와 해석적 blast-wave 비교로 판별할 계획입니다.

## 진단 재현 메모

진단은 `hic.blender_event_pipeline`의 함수를 그대로 import하고, `evolve_hydrodynamics`의
두 줄(마스크, 소스항)만 바꾼 변형과 비교했습니다. torch 2.15 CPU, NumPy 2.4.4,
Linux에서 실행했습니다. 기준 이벤트의 N_part = 236, N_coll = 574는 README와 일치했고,
격자 160에서 v₂ = 0.0113 ± 0.0013으로 README 값 0.0119(CUDA)와 통계오차 안에서 일치했습니다.
