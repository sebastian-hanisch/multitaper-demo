"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from mtp_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "β": "beta", "λ": "lambda", "Σ": "Summe", "±": "+-", "√": "Wurzel aus "}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Multitaper: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = {"keine": "keine Linie", "eine": f"eine Linie bei {de(s.f1_hz, 1)} Hz ({s.snr} dB je Abtastwert)",
             "zwei": f"zwei Linien bei {de(s.f1_hz, 1)} und {de(s.f2_hz, 1)} Hz ({s.snr} dB, zweite {s.rel} dB)"}[s.lines]
    pdf.multi_cell(0, 5, _s(f"Folge der Länge n = {s.n}, NW = {s.nw:g}, {a.k} Taper, {s.weights}e Gewichte, Hintergrund: {s.background}, {lines}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [
        f"Unabhängige Mittel: {de(a.k_eff, 1)} (Multitaper) gegen {de(a.welch_k_eff, 1)} (Welch bei gleicher Rauschbandbreite, Segmentlänge {min(s.welch_seg, s.n)}).",
        f"Streuung des Logarithmus je Frequenz: {de(a.std_db, 2)} dB (Multitaper), {de(a.welch_std_db, 2)} dB (Welch), {de(a.full_std_db, 2)} dB (ein Periodogramm).",
        f"Rauschbandbreite: {de(a.bandwidth_hz, 2)} Hz (Multitaper), {de(a.welch_enbw_hz, 2)} Hz (Welch).",
        f"Rechenaufwand (komplexe Multiplikationen): {a.ops} (Multitaper), {a.ops_welch} (Welch), {a.ops_full} (ein Periodogramm).",
        f"Konzentration der Taper (1 - lambda): " + ", ".join(f"{1 - v:.1e}" for v in a.lam) + ".",
    ]
    if a.f_stat is not None:
        found = ", ".join(f"{de(f, 1)} Hz" for f in a.found) or "keine"
        out.append(f"F-Test (Falschalarm 1 % je Aufzeichnung, Schwelle {de(a.f_thr, 1)}): gefundene Linien: {found}. Fisher-g: {'Alarm' if a.fisher_alarm else 'kein Alarm'}.")
    if a.resolved is not None:
        out.append(f"Zwei Linien getrennt: Multitaper {'ja' if a.resolved else 'nein'}, Welch {'ja' if a.welch_resolved else 'nein'}.")
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: multitaper-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
