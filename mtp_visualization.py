"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import mtp_constants as C
from mtp_evaluation import Analysis

EPS = 1e-30


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=float), EPS))


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _line_positions(a: Analysis) -> list[float]:
    s = a.settings
    out = []
    if s.lines in ("eine", "zwei"):
        out.append(s.f1_hz)
    if s.lines == "zwei":
        out.append(s.f2_hz)
    return out


def build_spectrum(a: Analysis, zoom: bool = False) -> go.Figure:
    """Spektren in dB über der Frequenz in Hz; zoom: Ausschnitt um die Linien, sonst bis zur Nyquist-Frequenz."""
    s = a.settings
    lines = _line_positions(a)
    if zoom and lines:
        pad = max(3.0 * a.bandwidth_hz, 0.5 * (max(lines) - min(lines)))
        lo, hi = max(0.0, min(lines) - pad), min(C.FS / 2, max(lines) + pad)
    else:
        lo, hi = 0.0, C.FS / 2
    sel = (a.f_hz >= lo) & (a.f_hz <= hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.welch[sel]), mode="lines", line=dict(color=C.COLOR_WELCH, width=1.5), name="Welch, gleiche Bandbreite",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.est[sel]), mode="lines", line=dict(color=C.COLOR_MT, width=2), name=f"Multitaper, {s.weights}e Gewichte",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    if a.expected is not None:
        fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.expected[sel]), mode="lines", line=dict(color=C.COLOR_EXPECTED, width=2.5), name="Erwartungswert (exakt)",
                                 hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.true_density[sel]), mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dot"), name="wahre Dichte des Hintergrunds",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    for pos in lines:
        if lo <= pos <= hi:
            fig.add_vline(x=pos, line=dict(color=C.COLOR_LINE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Hz)", range=[lo, hi])
    fig.update_yaxes(title_text="Leistungsdichte (dB)")
    return _fixed(fig, height=400)


def build_f_spectrum(a: Analysis) -> go.Figure:
    """F-Statistik über der Frequenz (logarithmische Achse) mit der Schwelle für 1 % Falschalarm je Aufzeichnung."""
    fig = go.Figure()
    if a.f_stat is not None:
        fig.add_trace(go.Scatter(x=a.f_hz_f, y=np.maximum(a.f_stat, 1e-3), mode="lines", line=dict(color=C.COLOR_MT, width=1.5), name="F(f)",
                                 hovertemplate="%{x:.1f} Hz: F = %{y:.1f}<extra></extra>"))
        fig.add_hline(y=a.f_thr, line=dict(color=C.COLOR_LINE, width=1.5, dash="dash"), annotation_text="Schwelle")
        for pos in _line_positions(a):
            fig.add_vline(x=pos, line=dict(color=C.COLOR_LINE, width=1, dash="dot"))
    fig.update_xaxes(title_text="Frequenz (Hz)", range=[0, C.FS / 2])
    fig.update_yaxes(title_text="F-Statistik", type="log")
    return _fixed(fig, height=300, showlegend=False)


def build_tapers(a: Analysis) -> go.Figure:
    fig = go.Figure()
    t = np.arange(a.h.shape[1])
    for k in range(a.h.shape[0]):
        fig.add_trace(go.Scatter(x=t, y=a.h[k], mode="lines", name=f"Taper {k + 1} (1 − λ = {1 - a.lam[k]:.1e})", line=dict(width=1.6),
                                 hovertemplate=f"Taper {k + 1}, Messwert %{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title_text="Messwert Nr.")
    fig.update_yaxes(title_text="Gewicht (Energie 1)")
    return _fixed(fig, height=360)


def build_window(a: Analysis) -> go.Figure:
    """Spektralfenster des Multitapers (Mittel der Betragsquadrate der Taperantworten) gegen das Hann-Fenster über dem Abstand in Bins."""
    lim = max(3.0 * a.settings.nw, 12.0)
    sel = a.window_f <= lim
    xs = np.r_[-a.window_f[sel][:0:-1], a.window_f[sel]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=_db(np.r_[a.window_mt[sel][:0:-1], a.window_mt[sel]]), mode="lines", name="Multitaper (Mittel der Taper)", line=dict(color=C.COLOR_MT, width=2),
                             hovertemplate="%{x:.2f} Bins: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=_db(np.r_[a.window_hann[sel][:0:-1], a.window_hann[sel]]), mode="lines", name="Hann, ganze Länge", line=dict(color=C.COLOR_WELCH, width=2),
                             hovertemplate="%{x:.2f} Bins: %{y:.1f} dB<extra></extra>"))
    for edge in (-a.settings.nw, a.settings.nw):
        fig.add_vline(x=edge, line=dict(color=C.COLOR_LINE, width=1, dash="dot"))
    fig.update_xaxes(title_text="Abstand von der Linie (Bins von f_s / n)", range=[-lim, lim])
    fig.update_yaxes(title_text="Antwort (dB)", range=[-130, 3])
    return _fixed(fig, height=320)


def build_concentration(res: dict) -> go.Figure:
    fig = go.Figure()
    for r in res["tapers"]:
        fig.add_trace(go.Scatter(x=list(range(1, len(r["lost"]) + 1)), y=np.maximum(r["lost"], 1e-16), mode="lines+markers", name=f"NW = {r['nw']}",
                                 hovertemplate="Taper %{x}: 1 − λ = %{y:.1e}<extra></extra>"))
    fig.update_xaxes(title_text="Nummer des Tapers k")
    fig.update_yaxes(title_text="Energie außerhalb des Bandes (1 − λ)", type="log", range=[-15.5, 0.2], tickformat=".0e")
    return _fixed(fig, height=340)


def build_equal_bandwidth(rows) -> go.Figure:
    xs = [r["bandwidth_bins"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["k_eff_mt"] for r in rows], mode="lines+markers", name="Multitaper (K = 2 NW − 1)",
                             hovertemplate="Bandbreite %{x} Bins: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["k_eff_welch"] for r in rows], mode="lines+markers", name="Welch, Hann, 50 %",
                             hovertemplate="Bandbreite %{x} Bins: %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text="Rauschbandbreite (Bins von f_s / n)", tickvals=xs)
    fig.update_yaxes(title_text="unabhängige Mittel", rangemode="tozero")
    return _fixed(fig, height=320)


def build_best(rows) -> go.Figure:
    labels = [f"n = {r['n']}, {r['background']}" for r in rows]
    fig = go.Figure()
    for key, name in (("welch", "Welch (bestes L)"), ("equal", "Multitaper, gleiche Gewichte (bestes NW)"), ("adaptive", "Multitaper, adaptiv (bestes NW)")):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in rows], name=name, hovertemplate="%{x}: %{y:.2f} dB<extra></extra>"))
    fig.update_yaxes(title_text="Gesamtfehler RMSE (dB)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=420)


def build_valley(rows) -> go.Figure:
    names = ["Gipfel", "Flanke", "Tal"]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Bar(x=names, y=r["bands"], name=r["label"], hovertemplate="%{x}: %{y:.1f} dB<extra></extra>"))
    fig.update_yaxes(title_text="Mittelwert über wahrer Dichte (dB)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=400)


def build_resolution(rows) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=r["deltas"], y=[100 * v for v in r["shares"]], mode="lines+markers", name=r["label"],
                                 hovertemplate=r["label"] + ", %{x:.0f} Bins: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Abstand der Linien (Bins von f_s / n)")
    fig.update_yaxes(title_text="Anteil aufgelöst (%)", range=[-3, 103])
    return _fixed(fig, height=380)


def build_line_test(res: dict) -> go.Figure:
    snrs = [d["snr"] for d in res["detect"]]
    fig = go.Figure()
    for key, name in (("f", "harmonischer F-Test"), ("g", "Fisher-g (nur weißes Rauschen)"), ("welch", "Welch, Schwelle aus Rauschläufen (Pegel bekannt)")):
        fig.add_trace(go.Scatter(x=snrs, y=[100 * d[key] for d in res["detect"]], mode="lines+markers", name=name, hovertemplate=name + ", %{x} dB: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Linie gegen Rauschen (dB je Abtastwert)")
    fig.update_yaxes(title_text="Linie gefunden (%)", range=[-3, 103])
    return _fixed(fig, height=360)
