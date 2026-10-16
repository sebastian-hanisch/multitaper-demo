"""Multitaper: orthogonale Fenster auf der ganzen Aufzeichnung - kein Zauber, aber ein gültiger Linientest - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal (Hintergrund mit bekannter Dichte, bis zu zwei Linien) wird mit K Slepian-Tapern
analysiert; gezeigt werden Streuung, Leckage, Auflösung und der harmonische F-Test gegen Welch bei gleicher Bandbreite. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import mtp_constants as C
import mtp_evaluation as ev
from mtp_evaluation import Settings, analyse, de, k_options, sci
from mtp_pdf_export import build_pdf
from mtp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from mtp_visualization import (
    build_best,
    build_concentration,
    build_equal_bandwidth,
    build_f_spectrum,
    build_line_test,
    build_resolution,
    build_spectrum,
    build_tapers,
    build_valley,
    build_window,
)

st.set_page_config(page_title="Multitaper – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🎚️ Multitaper: orthogonale Fenster auf der ganzen Aufzeichnung")
st.markdown(
    """
Welch schneidet die Aufzeichnung in Segmente und mittelt ihre Spektren. Das **Multitaper** lässt die Aufzeichnung ganz und legt **K orthogonale Fenster** (Slepian-Taper) darüber: Jedes
Taper liefert ein **Eigenspektrum**, ihr Mittel ist der Schätzer. Die Taper sind so gebaut, dass fast keine Energie außerhalb des Bandes der Breite 2·NW/n liegt; die K Eigenspektren
sind bei weißem Rauschen **unabhängig**. Dazu gibt es einen **Linientest** (harmonischer F-Test), dessen Falschalarm auch bei gefärbtem Hintergrund stimmt.

Dies ist die vierte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf **Periodogramm**, **Fenstern** und **Bartlett/Welch** aufbauend; mit „Multi-…“-Demos anderer Reihen hat sie nichts zu tun.
Alles ist **simuliert**: Hintergrund, Linien und die wahre Dichte sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Multitaper", expanded=True):
    st.markdown(
        """
1. **Taper:** NW legt die halbe Bandbreite W = NW/n fest. Die ersten 2NW − 1 Taper haben mindestens 92 % ihrer Energie in [−W, W] (das erste mehr als 99,99 %); Taper Nr. 2NW hat nur noch etwa 70 %.
2. **Eigenspektren:** Jedes Taper wird mit der ganzen Aufzeichnung multipliziert und per FFT transformiert; der Schätzer ist das Mittel der Betragsquadrate.
3. **Streuung:** Orthogonale Taper machen die Eigenspektren bei weißem Rauschen unabhängig: Die Streuung sinkt mit 1/√K, Überlappung gibt es nicht.
4. **Gewichte:** Die höheren Taper lassen mehr Energie von außerhalb ein. **Adaptive Gewichte** (Thomson) dämpfen sie, wo das Spektrum steil ist.
5. **F-Test:** Alle Taper sehen eine Linie mit demselben Vorzeichenmuster ihrer Summe. Das Verhältnis aus diesem Anteil und dem Rest ist F-verteilt – eine Schwelle mit gültigem Falschalarm.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Multitaper**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Anzahl der Messwerte (Abtastrate 1000 Hz).")
    nw = st.select_slider("Zeit-Bandbreite-Produkt NW", options=list(C.NW_CHOICES), key="nw_select",
                          help="Halbe Bandbreite W = NW/n. Größer: breiteres Band, mehr brauchbare Taper, gröbere Auflösung.")
    if st.session_state["k_select"] > int(2 * nw):
        st.session_state["k_select"] = int(2 * nw)
    k = st.select_slider("Zahl der Taper K", options=k_options(nw), key="k_select",
                         help="Höchstens 2 NW; die ersten 2 NW − 1 sind gut konzentriert. Weniger Taper: mehr Streuung, weniger Leckage.")
    weights = st.radio("Gewichte der Eigenspektren", list(C.WEIGHTS), key="weights_select", horizontal=True,
                       help="Gleich: einfaches Mittel. Adaptiv: Thomsons Gewichte dämpfen die höheren Taper, wo das Spektrum steil ist.")
    st.markdown("**Signal**")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select",
                              help="Weiß: flach. Resonanz: ein Gipfel bei 200 Hz. Steil: 57 dB Spannweite. Zwei scharfe Resonanzen bei 100 und 300 Hz.")
    lines = st.radio("Linien", list(C.LINES), key="lines_select", horizontal=True, help="Reine Töne auf dem Hintergrund (Lage und Stärke bekannt).")
    if lines != "keine":
        line = st.slider("Linie 1 (Bruchteil der Abtastrate)", C.LINE_MIN, C.LINE_MAX, float(st.session_state["line_kept"]), step=C.LINE_STEP, format="%.4f",
                         help="Lage der ersten Linie; mal 1000 Hz ergibt die Frequenz.")
        st.session_state["line_kept"] = float(line)
        snr = st.slider("Linie 1 gegen Rauschen (dB je Abtastwert)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]),
                        help="Linienleistung gegen die Rauschvarianz 1. Bei −10 dB steckt die Linie 10 dB unter dem Rauschen eines Abtastwerts.")
        st.session_state["snr_kept"] = int(snr)
    else:
        line, snr = float(st.session_state["line_kept"]), int(st.session_state["snr_kept"])
    if lines == "zwei":
        gap = st.slider("Abstand der Linie 2 (Bins von f_s/n)", C.GAP_MIN, C.GAP_MAX, float(st.session_state["gap_kept"]), step=C.GAP_STEP,
                        help="Abstand der zweiten Linie in Vielfachen von f_s/n. Die Bandbreite des Multitapers ist 2·NW Bins.")
        st.session_state["gap_kept"] = float(gap)
        rel = st.slider("Linie 2 gegen Linie 1 (dB)", C.REL_MIN, C.REL_MAX, int(st.session_state["rel_kept"]), help="0 dB: gleich stark.")
        st.session_state["rel_kept"] = int(rel)
    else:
        gap, rel = float(st.session_state["gap_kept"]), int(st.session_state["rel_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "nw_select": int(nw), "k_select": int(k), "weights_select": weights, "background_select": background, "lines_select": lines,
                   "line_kept": float(line), "gap_kept": float(gap), "rel_kept": int(rel), "snr_kept": int(snr), "seed_input": int(seed)})

settings = Settings(int(n), int(nw), int(k), weights, background, lines, float(line), float(gap), int(rel), int(snr), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Unabhängige Mittel", f"{a.k}", delta=f"Welch bei gleicher Bandbreite: {de(a.welch_k_eff, 1)}", delta_color="off")
m2.metric("Streuung je Frequenz", f"±{de(a.std_db, 1)} dB", delta=f"Welch ±{de(a.welch_std_db, 1)} dB, ein Periodogramm ±{de(a.full_std_db, 1)} dB", delta_color="off")
m3.metric("Rauschbandbreite", f"{de(a.bandwidth_hz, 1)} Hz", delta=f"= 2·NW = {2 * int(nw)} Bins von f_s/n", delta_color="off")
m4.metric("Rechenaufwand", f"{a.ops:,}".replace(",", "."), delta=f"{de(a.ops / a.ops_welch, 1)}-fach Welch, {de(a.ops / a.ops_full, 1)}-fach ein Periodogramm", delta_color="off",
          help="Komplexe Multiplikationen der K FFTs der Länge n (Radix 2).")
st.plotly_chart(build_spectrum(a, zoom=False), width="stretch", key="spectrum_full")
st.caption("Grau: Welch (Hann, 50 % Überlappung) mit der Segmentlänge, die dieselbe Rauschbandbreite ergibt. Blau: Multitaper. Orange: exakter Erwartungswert des Multitapers "
           "(nur bei gleichen Gewichten), grün gepunktet: wahre Dichte, rot gestrichelt: wahre Lage der Linien.")
if lines != "keine":
    st.plotly_chart(build_spectrum(a, zoom=True), width="stretch", key="spectrum_zoom")
    st.caption("Ausschnitt um die Linien.")
st.markdown("#### F-Test: Welche Frequenzen sind Linien?")
if a.f_stat is None:
    st.info("Mit einem Taper ist kein F-Test möglich (Zähler und Nenner brauchen mindestens zwei Taper).")
else:
    st.plotly_chart(build_f_spectrum(a), width="stretch", key="f_spectrum")
    found = ", ".join(f"{de(f, 1)} Hz" for f in a.found) or "keine"
    st.caption(f"F(2, {2 * a.k - 2}), Schwelle {de(a.f_thr, 1)} für {de(100 * C.ALPHA, 0)} % Falschalarm je Aufzeichnung (Bonferroni über {C.PADF * a.settings.n // 2} Punkte des {C.PADF}-fach aufgefüllten "
               f"Rasters). Gefundene Linien: {found}. Fisher-g (nur weißes Rauschen): {'Alarm' if a.fisher_alarm else 'kein Alarm'}.")
with st.expander("Taper und Spektralfenster"):
    st.plotly_chart(build_tapers(a), width="stretch", key="tapers")
    st.plotly_chart(build_window(a), width="stretch", key="window")
    st.caption("Das mittlere Betragsquadrat der Taperantworten ist das Spektralfenster des Schätzers: kastenförmig bis ±NW Bins (rot gepunktet), danach steil abfallend. Das Hann-Fenster "
               "der ganzen Länge hat eine glockenförmige Hauptkeule von 1,5 Bins Rauschbandbreite.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="multitaper.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kann das Multitaper, was Welch nicht kann?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme (Seed {C.SWEEP_SEED} und folgende), unabhängig von den Reglern. Aufzeichnung n = {C.EXP_N}, Abtastrate {C.FS:g} Hz, außer wo angegeben.")

st.markdown("#### Wie gut sind die Taper konzentriert?")
conc = _experiment("concentration")
rows = []
for r in conc["tapers"]:
    k_good = 2 * r["nw"] - 1
    rows.append({"NW": r["nw"], "Taper 1": f"{sci(r['lost'][0])}", f"Taper {k_good} (letzter gut konzentrierter)": f"{sci(r['lost'][k_good - 1])}",
                 f"Taper {k_good + 1}": f"{de(100 * r['lost'][k_good], 1)} %", f"Taper {k_good + 2}": f"{de(100 * r['lost'][k_good + 1], 1)} %"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Anteil der Taper-Energie außerhalb des Bandes [−W, W] (1 − λ). Zum Vergleich außerhalb von ±" + f"{conc['band_nw']} Bins: "
           + ", ".join(f"{c['window']} {sci(c['outside'])}" for c in conc["classic"]) + ".")
st.plotly_chart(build_concentration(conc), width="stretch", key="exp_concentration")
kind, text = ev.verdict_concentration(conc)
getattr(st, kind)(text)

st.markdown("#### Gleiche Bandbreite: Multitaper gegen Welch")
bw = _experiment("equal_bandwidth")
st.caption(f"Weißes Rauschen, {C.EXP_BW_RUNS} Läufe. Welch mit Hann und 50 % Überlappung, Segmentlänge so, dass die Rauschbandbreite 2·NW Bins beträgt.")
st.dataframe(pd.DataFrame([{"NW": r["nw"], "Bandbreite (Bins)": r["bandwidth_bins"], "Taper K": r["k"], "Mittel Multitaper": de(r["k_eff_mt"], 1),
                            "Streuung Multitaper (dB)": de(r["std_mt"], 2), "Segmentlänge Welch": r["seg"], "Segmente": r["k_welch"],
                            "Mittel Welch gemessen": de(r["k_eff_welch"], 1), "Mittel Welch Formel": de(r["k_eff_welch_formula"], 1),
                            "Streuung Welch (dB)": de(r["std_welch"], 2)} for r in bw]), hide_index=True, width="stretch")
st.plotly_chart(build_equal_bandwidth(bw), width="stretch", key="exp_bandwidth")
kind, text = ev.verdict_equal_bandwidth(bw)
getattr(st, kind)(text)

st.markdown("#### Beste gegen beste")
best = _experiment("best_vs_best")
st.caption(f"Gesamtfehler (RMSE in dB gegen die wahre Dichte), {C.EXP_BEST_RUNS} Läufe; je Verfahren der beste Parameter (Welch: Segmentlänge 16 bis 512, Multitaper: NW 2 bis 12).")
st.dataframe(pd.DataFrame([{"n": r["n"], "Hintergrund": r["background"], "Spanne (dB)": de(r["span_db"], 0), "Welch (dB)": f"{de(r['welch'], 2)} (L = {r['welch_seg']})",
                            "Multitaper gleich (dB)": f"{de(r['equal'], 2)} (NW = {r['equal_nw']})", "Multitaper adaptiv (dB)": f"{de(r['adaptive'], 2)} (NW = {r['adaptive_nw']})"}
                           for r in best]), hide_index=True, width="stretch")
st.plotly_chart(build_best(best), width="stretch", key="exp_best")
kind, text = ev.verdict_best(best)
getattr(st, kind)(text)
valley = _experiment("valley")
st.caption(f"Leckage im Tal: steiler Hintergrund, n = {C.EXP_N}, {C.EXP_VALLEY_RUNS} Läufe; Mittelwert des Schätzers gegen die wahre Dichte (Gipfel 150–250 Hz, Flanke 300–350 Hz, Tal 450–500 Hz).")
st.plotly_chart(build_valley(valley), width="stretch", key="exp_valley")
kind, text = ev.verdict_valley(valley)
getattr(st, kind)(text)

st.markdown("#### Wie dicht dürfen zwei Linien liegen?")
res = _experiment("resolution")
st.caption(f"Zwei gleich starke Linien (10 dB je Linie) in weißem Rauschen, {C.EXP_RES_RUNS} Läufe; aufgelöst heißt zwei Maxima mit einem Einbruch von mindestens 3 dB dazwischen. "
           "Die Rauschbandbreite ist beim Multitaper 2·NW Bins, bei Welch mit Hann 1,5·n/L Bins.")
st.plotly_chart(build_resolution(res), width="stretch", key="exp_resolution")
kind, text = ev.verdict_resolution(res)
getattr(st, kind)(text)

st.markdown("#### Linientest mit gültigem Falschalarm")
lt = _experiment("line_test")
st.caption(f"Eine Linie zwischen zwei Bins, n = {C.EXP_N}, NW = {lt['nw']:g}, K = {lt['k']}, {lt['runs']} Läufe; Schwellen für {de(100 * lt['alpha'], 0)} % Falschalarm je Aufzeichnung "
           f"(F > {de(lt['f_thr'], 1)}, g > {de(lt['g_thr'], 4)}).")
st.dataframe(pd.DataFrame([{"Hintergrund": r["background"], "Falschalarm F-Test": de(100 * r["f"], 1) + " %", "Falschalarm Fisher-g": de(100 * r["g"], 1) + " %"}
                           for r in lt["false_alarm"]]), hide_index=True, width="stretch")
st.plotly_chart(build_line_test(lt), width="stretch", key="exp_line_test")
kind, text = ev.verdict_line_test(lt)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **K ≈ 2NW − 1 Taper** | Mehr Taper lassen Energie von außerhalb ein (Leckage); weniger streuen stärker. | – |
| **Eine Linie im Band** (F-Test) | Zwei Linien im selben Band lassen F klein werden; der Test sieht sie nur einzeln. | Unterraumverfahren |
| **Linie nicht am Rand** | Nahe 0 und der Nyquist-Frequenz ist die Spiegelung nicht zu trennen; die ersten und letzten Bins zählen nicht. | – |
| **Stationär** | Ändert sich das Spektrum, mittelt das Multitaper darüber hinweg. | Spektrogramm |
| **Bonferroni über das Raster** | Konservativ: der Falschalarm je Aufzeichnung liegt unter dem Nennwert, die Empfindlichkeit leidet. | Konfidenz und Falschalarm |
| **Adaptive Gewichte** | Eine Näherung mit eigener Verzerrung; der Erwartungswert ist nur durch Simulation zu bekommen. | – |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Slepian-Taper.** Die Folgen $h_k$, $k = 0, 1, \dots$, mit Energie 1, die im Band $[-W, W]$, $W = NW/n$, die größte Energiekonzentration $\lambda_k$ haben; sie sind die Eigenvektoren
von $A_{mm'} = \sin(2\pi W(m-m'))/(\pi(m-m'))$ und der Tridiagonalmatrix mit Diagonale $\big(\tfrac{n-1-2t}{2}\big)^2\cos 2\pi W$ und Nebendiagonale $\tfrac{t(n-t)}{2}$.
Gut konzentriert sind etwa die ersten $2NW - 1$.

**Eigenspektren und Schätzer.** $Y_k(f) = \sum_t h_k[t]\,x[t]\,e^{-2\pi i f t/f_s}$; $\hat S(f) = \dfrac{1}{f_s}\sum_k d_k^2(f)\,|Y_k(f)|^2 \Big/ \sum_k d_k^2(f)$ mit $d_k \equiv 1$ (gleiche Gewichte)
oder $d_k^2 = \lambda_k\big(\hat S/(\lambda_k \hat S + (1-\lambda_k)\sigma^2)\big)^2$ (adaptiv, Fixpunkt).

**Erwartungswert (gleiche Gewichte).** $E\hat S(f) = \dfrac{1}{K f_s}\sum_k\sum_{|\tau|<n} r(\tau)\,c_k(\tau)\,e^{-i\omega\tau} + $ Anteil der Linien, mit $c_k(\tau) = \sum_t h_k[t]h_k[t+\tau]$.
Bei weißem Rauschen sind die $Y_k$ unkorreliert: $\hat S \sim S\,\chi^2_{2K}/(2K)$, also $K_\mathrm{eff} = K$.

**Harmonischer F-Test.** Mit $U_k(0) = \sum_t h_k[t]$: $\hat\mu(f) = \sum_k U_k(0) Y_k(f)\big/\sum_k U_k(0)^2$ und
$F(f) = (K-1)\,|\hat\mu|^2 \sum_k U_k(0)^2 \Big/ \sum_k |Y_k(f) - \hat\mu U_k(0)|^2$ ist unter $H_0$ $F(2, 2K-2)$-verteilt; $P(F > x) = (1 + x/(K-1))^{-(K-1)}$.

**Fisher-g.** $g = \max_j I_j / \sum_j I_j$ über die Periodogrammordinaten; für weißes Gauß-Rauschen $P(g > x) = \sum_{j \ge 1} (-1)^{j-1}\binom{m}{j}(1 - jx)_+^{m-1}$.

**Quellen.** Lehrbuchwissen (Slepian 1978; Thomson 1982; Percival und Walden 1993). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
