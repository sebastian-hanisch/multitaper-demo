"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import mtp_constants as C
import mtp_spectrum as S
import mtp_taper as T
import mtp_welch as H
import signal_generator as G

DB = H.DB


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def k_options(nw: float) -> list[int]:
    """Zahl der Taper, die zu NW passen: 1 bis 2 NW (die ersten 2 NW - 1 sind gut konzentriert)."""
    return list(range(1, int(2 * nw) + 1))


def sci(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma (2,9e-10)."""
    return f"{x:.1e}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    nw: float = C.DEFAULT_NW
    k: int = C.DEFAULT_K
    weights: str = C.DEFAULT_WEIGHTS
    background: str = C.DEFAULT_BACKGROUND
    lines: str = C.DEFAULT_LINES
    line: float = C.DEFAULT_LINE
    gap: float = C.DEFAULT_GAP
    rel: int = C.DEFAULT_REL
    snr: int = C.DEFAULT_SNR
    seed: int = C.DEFAULT_SEED

    @property
    def k_used(self) -> int:
        """Zahl der Taper, höchstens 2 NW."""
        return max(1, min(self.k, int(2 * self.nw)))

    @property
    def f1_hz(self) -> float:
        return self.line * C.FS

    @property
    def f2_hz(self) -> float:
        """Lage der Linie 2: oberhalb von Linie 1; läge sie jenseits der Nyquist-Grenze (zwei Bins davor), dann unterhalb im selben Abstand."""
        step = self.gap * C.FS / self.n
        up = self.f1_hz + step
        return up if up <= C.FS / 2 - 2.0 * C.FS / self.n else self.f1_hz - step

    @property
    def welch_seg(self) -> int:
        """Segmentlänge des Welch-Vergleichs: gleiche Rauschbandbreite (Hann: 1,5 n / L = 2 NW Bins), nicht kürzer als 8."""
        return max(8, int(round(1.5 * self.n / (2.0 * self.nw))))


def noise_spec(background: str):
    """(Pole, Innovations-Sigma, weißes Sigma) eines Hintergrunds, auf Varianz 1 normiert."""
    poles = C.BACKGROUNDS[background]
    if poles:
        return poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0
    return (), 0.0, 1.0


def amplitude(snr_db: float) -> float:
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    k: int
    h: np.ndarray                 # Taper k x n
    lam: np.ndarray               # Konzentration je Taper
    f_hz: np.ndarray              # Frequenzachse der Spektren (aufgefüllt)
    x: np.ndarray
    est: np.ndarray               # Multitaper-Spektrum
    welch: np.ndarray             # Welch bei gleicher Rauschbandbreite
    welch_k: int
    welch_k_eff: float
    expected: np.ndarray | None   # exakter Erwartungswert (nur gleiche Gewichte)
    true_density: np.ndarray
    f_hz_f: np.ndarray            # Frequenzachse des F-Spektrums
    f_stat: np.ndarray | None     # None bei einem Taper
    f_thr: float | None
    found: list                   # Frequenzen (Hz) der Linien, die der F-Test findet
    fisher_g: float
    fisher_thr: float
    fisher_alarm: bool
    window_f: np.ndarray          # Spektralfenster: Abstand in Bins
    window_mt: np.ndarray         # mittleres Betragsquadrat der Taper (normiert)
    window_hann: np.ndarray
    bandwidth_hz: float
    welch_enbw_hz: float
    k_eff: float                  # äquivalente Zahl unabhängiger Mittel (gleiche Gewichte: k)
    std_db: float
    welch_std_db: float
    full_std_db: float
    resolved: bool | None
    welch_resolved: bool | None
    ops: int
    ops_welch: int
    ops_full: int


def build_signal(s: Settings):
    rng = G.SplitMix64(s.seed)
    ph = 2.0 * math.pi * rng.uniform(2)
    poles, ar_sigma, white = noise_spec(s.background)
    a1 = amplitude(s.snr)
    a2 = a1 * 10.0 ** (s.rel / 20.0)
    lines = []
    if s.lines in ("eine", "zwei"):
        lines.append((s.f1_hz, a1, float(ph[0])))
    if s.lines == "zwei":
        lines.append((s.f2_hz, a2, float(ph[1])))
    sig = G.make_signal(s.n, C.FS, lines, poles, ar_sigma, white, rng)
    return sig, (poles, ar_sigma, white)


def _peaks_above(values, threshold: float, min_distance: int) -> list[int]:
    """Indizes lokaler Maxima über der Schwelle, in absteigender Höhe; Nachbarn näher als min_distance werden verworfen."""
    v = np.asarray(values, dtype=float)
    idx = [i for i in range(1, len(v) - 1) if v[i] > threshold and v[i] >= v[i - 1] and v[i] > v[i + 1]]
    idx.sort(key=lambda i: -v[i])
    kept: list[int] = []
    for i in idx:
        if all(abs(i - j) >= min_distance for j in kept):
            kept.append(i)
    return sorted(kept)


def analyse(s: Settings) -> Analysis:
    sig, (poles, ar_sigma, white) = build_signal(s)
    n, k = s.n, s.k_used
    h = T.dpss(n, s.nw, k)
    lam = T.concentration(h, s.nw)
    nfft = C.PAD * n
    f_hz = np.arange(nfft // 2 + 1) * C.FS / nfft
    if s.weights == "adaptiv":
        est = S.multitaper_adaptive(sig.x, h, lam, C.FS, nfft)
        expected = None
    else:
        est = S.multitaper_equal(sig.x, h, C.FS, nfft)
        r = G.noise_autocov(poles, ar_sigma, white, n)
        expected = S.expected_multitaper(h, r, C.FS, nfft)
        if s.lines != "keine":
            expected = expected + S.multitaper_equal(sig.clean, h, C.FS, nfft)
    seg = min(s.welch_seg, n)
    hop = H.hop_for(seg, 0.5)
    w = H.window("Hann", seg)
    welch, wk = H.welch(sig.x, seg, hop, w, C.FS, nfft)
    welch_k_eff = H.k_effective(w, hop, wk)
    true_density = G.true_psd(f_hz, C.FS, poles, ar_sigma, white)
    nfft_f = C.PADF * n
    f_hz_f = np.arange(nfft_f // 2 + 1) * C.FS / nfft_f
    f_stat = f_thr = None
    found: list = []
    if k >= 2:
        f_stat = S.f_statistic(sig.x, h, nfft_f)
        f_thr = S.f_threshold(k, C.ALPHA / (nfft_f // 2))
        guard = C.F_GUARD
        inner = np.zeros(len(f_stat), dtype=bool)
        inner[guard:len(f_stat) - guard] = True
        masked = np.where(inner, f_stat, 0.0)
        found = [float(f_hz_f[i]) for i in _peaks_above(masked, f_thr, max(1, int(s.nw * C.PADF)))]
    p0 = np.abs(np.fft.rfft(sig.x)) ** 2
    fisher_thr = S.fisher_threshold(len(p0) - 2, C.ALPHA)
    g = float(S.fisher_g(p0[1:-1]))
    win_nfft = 64 * n
    win_f = np.arange(win_nfft // 2 + 1) * n / win_nfft
    hann = H.window("Hann", n)
    hann_resp = np.abs(np.fft.rfft(hann, win_nfft)) ** 2
    hann_resp = hann_resp / hann_resp[0]
    k_eff = float(k)
    bandwidth = S.bandwidth_hz(n, s.nw, C.FS)
    resolved = welch_resolved = None
    if s.lines == "zwei":
        a_bins, b_bins = s.f1_hz * n / C.FS, s.f2_hz * n / C.FS
        resolved = S.lines_resolved(est, C.PAD, a_bins, b_bins)
        welch_resolved = S.lines_resolved(welch, C.PAD, a_bins, b_bins)
    return Analysis(s, k, h, lam, f_hz, sig.x, est, welch, wk, welch_k_eff, expected, true_density, f_hz_f, f_stat, f_thr, found, g, fisher_thr, g > fisher_thr,
                    win_f, S.spectral_window(h, win_nfft), hann_resp, bandwidth, H.enbw(w) * C.FS / seg, k_eff, H.log_std_db(k_eff), H.log_std_db(welch_k_eff),
                    H.log_std_db(1.0), resolved, welch_resolved, S.ops_multitaper(n, k), wk * S.ops_fft(seg), S.ops_fft(n))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _white_batch(runs: int, n: int, seed: int) -> np.ndarray:
    return G.make_noise_batch(runs, n, (), 0.0, 1.0, G.SplitMix64(seed))


def sweep_concentration(nws=C.EXP_NW, n: int = C.EXP_N) -> dict:
    """Energie außerhalb des Bandes [-W, W] (1 - lambda) der ersten 2 NW + 2 Taper je NW und, für NW = EXP_CONC_BAND_NW, Rechteck, Hann und Blackman im selben Band."""
    taper_rows = []
    for nw in nws:
        k = 2 * nw + 2
        lam = T.concentration(T.dpss(n, nw, k), nw)
        taper_rows.append({"nw": nw, "lost": [float(1.0 - v) for v in lam]})
    band = C.EXP_CONC_BAND_NW / n
    classic = []
    for name in ("Rechteck", "Hann", "Blackman"):
        resp = np.abs(np.fft.rfft(H.window(name, n), 64 * n)) ** 2
        f = np.arange(len(resp)) / (64 * n)
        classic.append({"window": name, "outside": float(resp[f > band].sum() / resp.sum()), "enbw": H.enbw(H.window(name, n))})
    return {"tapers": taper_rows, "classic": classic, "band_nw": C.EXP_CONC_BAND_NW}


def _inner(n_bins: int) -> slice:
    return slice(8, n_bins - 8)


def sweep_equal_bandwidth(nws=C.EXP_NW, n: int = C.EXP_N, runs: int = C.EXP_BW_RUNS) -> list[dict]:
    """Weißes Rauschen: Multitaper (K = 2 NW - 1, gleiche Gewichte) gegen Welch (Hann, 50 %) bei gleicher Rauschbandbreite 2 NW Bins."""
    x = _white_batch(runs, n, C.SWEEP_SEED)
    rows = []
    for nw in nws:
        k = 2 * nw - 1
        h = T.dpss(n, nw, k)
        est = S.multitaper_equal(x, h, C.FS)
        inner = est[:, _inner(est.shape[1])]
        cv2 = float(((inner.std(axis=0) / inner.mean(axis=0)) ** 2).mean())
        std_mt = float((DB * np.log(inner)).std(axis=0).mean())
        seg = int(round(1.5 * n / (2 * nw)))
        w = H.window("Hann", seg)
        hop = H.hop_for(seg, 0.5)
        est_w, kw = H.welch(x, seg, hop, w, C.FS)
        inner_w = est_w[:, 2:-2]
        cv2_w = float(((inner_w.std(axis=0) / inner_w.mean(axis=0)) ** 2).mean())
        std_w = float((DB * np.log(inner_w)).std(axis=0).mean())
        rows.append({"nw": nw, "k": k, "bandwidth_bins": 2 * nw, "k_eff_mt": 1.0 / cv2, "std_mt": std_mt, "seg": seg, "k_welch": kw, "k_eff_welch": 1.0 / cv2_w,
                     "k_eff_welch_formula": H.k_effective(w, hop, kw), "std_welch": std_w, "welch_enbw_bins": H.enbw(w) * n / seg})
    return rows


def _usable_segs(n: int) -> list[int]:
    """Segmentlängen des Welch-Vergleichs, die in eine Folge der Länge n passen."""
    return [seg for seg in C.EXP_BEST_SEGS if seg <= n]


def _usable_nws(n: int) -> list:
    """NW-Werte des Multitaper-Vergleichs, deren Band (2 NW Bins) höchstens ein Viertel der Länge belegt."""
    return [nw for nw in C.EXP_BEST_NW if 2 * nw < n // 4]


def sweep_best_vs_best(ns=C.EXP_BEST_NS, backgrounds=C.EXP_BEST_BACKGROUNDS, runs: int = C.EXP_BEST_RUNS) -> list[dict]:
    """Gesamtfehler (RMSE in dB gegen die wahre Dichte): bester Welch (Hann, 50 %) gegen bestes Multitaper mit gleichen und mit adaptiven Gewichten, je n und Hintergrund."""
    rows = []
    for n in ns:
        for bg in backgrounds:
            poles, ar_sigma, white = noise_spec(bg)
            x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 1))
            f = np.arange(n // 2 + 1) * C.FS / n
            true = G.true_psd(f, C.FS, poles, ar_sigma, white)
            span = 10.0 * math.log10(float(true.max() / true.min()))
            sl = slice(4, n // 2 - 4)

            def rmse(est):
                e = DB * np.log(est[:, sl] / true[sl])
                return float(math.sqrt((e ** 2).mean()))

            welch_best = min((rmse(H.welch(x, seg, H.hop_for(seg, 0.5), H.window("Hann", seg), C.FS, n)[0][:, :n // 2 + 1]), seg)
                             for seg in _usable_segs(n))
            nws = _usable_nws(n)
            equal_best = adaptive_best = None
            for nw in nws:
                h = T.dpss(n, nw, 2 * nw - 1)
                lam = T.concentration(h, nw)
                e_eq = rmse(S.multitaper_equal(x, h, C.FS, n)[:, :n // 2 + 1])
                e_ad = rmse(S.multitaper_adaptive(x, h, lam, C.FS, n)[:, :n // 2 + 1])
                if equal_best is None or e_eq < equal_best[0]:
                    equal_best = (e_eq, nw)
                if adaptive_best is None or e_ad < adaptive_best[0]:
                    adaptive_best = (e_ad, nw)
            rows.append({"n": n, "background": bg, "span_db": span, "welch": welch_best[0], "welch_seg": welch_best[1], "equal": equal_best[0], "equal_nw": equal_best[1],
                         "adaptive": adaptive_best[0], "adaptive_nw": adaptive_best[1]})
    return rows


def _band_bias(f_frac, est_mean, true, bands) -> list[float]:
    out = []
    for lo, hi in bands:
        sel = (f_frac >= lo) & (f_frac <= hi)
        out.append(float(DB * np.log(est_mean[sel] / true[sel]).mean()))
    return out


VALLEY_BANDS = ((0.15, 0.25), (0.30, 0.35), (0.45, 0.4999))      # Gipfel, Flanke, Tal (Bruchteile der Abtastrate)


def sweep_valley(n: int = C.EXP_N, runs: int = C.EXP_VALLEY_RUNS) -> list[dict]:
    """Steiler Hintergrund: Mittelwert des Schätzers gegen die wahre Dichte in dB (Gipfel, Flanke, Tal)."""
    poles, ar_sigma, white = noise_spec("Steil")
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 2))
    f = np.arange(n // 2 + 1) / n
    true = G.true_psd(f * C.FS, C.FS, poles, ar_sigma, white)

    def bands(est):
        return _band_bias(f, est.mean(axis=0), true, VALLEY_BANDS)

    wh = H.window("Hann", n)
    seg = C.EXP_VALLEY_WELCH_SEG
    rows = [{"label": "Periodogramm, Rechteck", "bands": bands(np.abs(np.fft.rfft(x, axis=1)) ** 2 / (C.FS * n))},
            {"label": "Periodogramm, Hann", "bands": bands(np.abs(np.fft.rfft(x * wh, axis=1)) ** 2 / (C.FS * (wh ** 2).sum()))},
            {"label": f"Welch, Hann 50 %, Länge {seg}", "bands": bands(H.welch(x, seg, seg // 2, H.window("Hann", seg), C.FS, n)[0][:, :n // 2 + 1])}]
    for nw, weights in ((4, "gleich"), (4, "adaptiv"), (2, "gleich"), (8, "gleich")):
        h = T.dpss(n, nw, 2 * nw - 1)
        est = S.multitaper_adaptive(x, h, T.concentration(h, nw), C.FS, n) if weights == "adaptiv" else S.multitaper_equal(x, h, C.FS, n)
        rows.append({"label": f"Multitaper NW = {nw}, {weights}", "bands": bands(est[:, :n // 2 + 1])})
    return rows


def sweep_resolution(deltas=C.EXP_RES_DELTAS, nws=C.EXP_RES_NW, segs=C.EXP_RES_SEGS, n: int = C.EXP_N, runs: int = C.EXP_RES_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Zwei gleich starke Linien in weißem Rauschen (SNR je Linie), Abstand in Bins von f_s / n: Anteil der Läufe, in denen das Spektrum zwei getrennte Maxima mit einem Einbruch von mindestens 3 dB dazwischen zeigt, je Verfahren."""
    sigma = math.sqrt(0.5 / 10.0 ** (snr / 10.0))
    t = np.arange(n)
    pad = C.PAD
    methods = ([(f"Multitaper NW = {nw}", ("mt", nw)) for nw in nws] + [(f"Welch, Hann 50 %, Länge {seg}", ("welch", seg)) for seg in segs]
               + [("Periodogramm, Hann", ("hann", n))])
    rows = [{"label": label, "shares": []} for label, _ in methods]
    for d in deltas:
        rng = G.SplitMix64(C.SWEEP_SEED + 3)
        f1 = (200.0 + 3.0 * rng.uniform(runs)) / n
        f2 = f1 + d / n
        ph = 2.0 * math.pi * rng.uniform(2 * runs).reshape(2, runs)
        x = (np.cos(2.0 * np.pi * f1[:, None] * t[None, :] + ph[0][:, None]) + np.cos(2.0 * np.pi * f2[:, None] * t[None, :] + ph[1][:, None])
             + sigma * rng.normal(runs * n).reshape(runs, n))
        for row, (_, (kind, par)) in zip(rows, methods):
            if kind == "mt":
                est = S.multitaper_equal(x, T.dpss(n, par, 2 * par - 1), 1.0, pad * n)
            elif kind == "welch":
                est = H.welch(x, par, par // 2, H.window("Hann", par), 1.0, pad * n)[0]
            else:
                est = np.abs(np.fft.rfft(x * H.window("Hann", n), pad * n, axis=1)) ** 2
            hits = 0
            for i in range(runs):
                hits += S.lines_resolved(est[i], pad, f1[i] * n, f2[i] * n)
            row["shares"].append(hits / runs)
    for row in rows:
        row["deltas"] = list(deltas)
    return rows


def sweep_line_test(n: int = C.EXP_N, nw: float = C.EXP_TEST_NW, runs: int = C.EXP_TEST_RUNS, snrs=C.EXP_TEST_SNR, alpha: float = C.ALPHA) -> dict:
    """Linientest mit Falschalarm alpha je Aufzeichnung: harmonischer F-Test (Bonferroni über das aufgefüllte Raster) gegen Fishers g-Test (weißes Rauschen) und gegen
    Welch mit einer Schwelle aus Rauschläufen (bekannter Rauschpegel)."""
    k = int(2 * nw - 1)
    h = T.dpss(n, nw, k)
    nfft_f = C.PADF * n
    guard = C.F_GUARD
    f_thr = S.f_threshold(k, alpha / (nfft_f // 2))
    g_thr = S.fisher_threshold(n // 2 - 1, alpha)

    def f_alarm(x):
        return S.f_statistic(x, h, nfft_f)[:, guard:-guard].max(axis=1) > f_thr

    def g_alarm(x):
        return S.fisher_g(np.abs(np.fft.rfft(x, axis=1))[:, 1:n // 2] ** 2) > g_thr

    fa = []
    for bg in C.EXP_TEST_BACKGROUNDS:
        poles, ar_sigma, white = noise_spec(bg)
        xs = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 4))
        fa.append({"background": bg, "f": float(f_alarm(xs).mean()), "g": float(g_alarm(xs).mean())})
    noise = _white_batch(runs, n, C.SWEEP_SEED + 5)
    seg = C.EXP_TEST_SEG
    w = H.window("Hann", seg)
    wn, _ = H.welch(noise, seg, seg // 2, w, 1.0)
    w_thr = float(np.quantile(wn[:, 2:-2].max(axis=1), 1.0 - alpha))
    rng = G.SplitMix64(C.SWEEP_SEED + 6)
    t = np.arange(n)
    f_line = (205.0 + 0.37) / n
    detect = []
    for snr in snrs:
        ph = 2.0 * math.pi * rng.uniform(runs)
        x = amplitude(snr) * np.cos(2.0 * np.pi * f_line * t[None, :] + ph[:, None]) + noise
        xw, _ = H.welch(x, seg, seg // 2, w, 1.0)
        detect.append({"snr": snr, "f": float(f_alarm(x).mean()), "g": float(g_alarm(x).mean()), "welch": float((xw[:, 2:-2].max(axis=1) > w_thr).mean())})
    return {"alpha": alpha, "k": k, "nw": nw, "f_thr": f_thr, "g_thr": g_thr, "false_alarm": fa, "detect": detect, "runs": runs}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    base = (f"{a.k} Taper mit NW = {s.nw:g} ergeben {a.k} unabhängige Mittel auf der ganzen Aufzeichnung (Rauschbandbreite {de(a.bandwidth_hz, 1)} Hz); Welch bei gleicher "
            f"Bandbreite hat etwa {de(a.welch_k_eff, 1)}.")
    if s.lines == "zwei":
        if a.resolved:
            return "success", base + " Die beiden Linien erscheinen im Multitaper-Spektrum als **zwei Gipfel**."
        return "warning", base + " Die beiden Linien **verschmelzen** im Multitaper-Spektrum (Abstand unter der Bandbreite)."
    if s.lines == "eine":
        if a.f_stat is None:
            return "info", base + " Mit einem Taper ist kein F-Test möglich."
        if a.found:
            return "success", base + f" Der F-Test findet die Linie bei {de(a.found[0], 1)} Hz (Falschalarm {de(100 * C.ALPHA, 0)} % je Aufzeichnung)."
        return "warning", base + " Der F-Test findet die Linie **nicht**: Sie liegt unter der Schwelle."
    return "info", base


def verdict_concentration(res: dict) -> tuple[str, str]:
    row = [r for r in res["tapers"] if r["nw"] == res["band_nw"]][0]
    k = 2 * res["band_nw"] - 1
    by = {c["window"]: c for c in res["classic"]}
    return "info", (f"Bei NW = {res['band_nw']} liegt außerhalb des Bandes beim ersten Taper {sci(row['lost'][0])} der Energie, beim {k}. (letzter gut konzentrierter) {sci(row['lost'][k - 1])}, "
                    f"danach springt es auf {de(100 * row['lost'][k], 0)} %. Beim Rechteck liegen {de(100 * by['Rechteck']['outside'], 1)} %, bei Hann {sci(by['Hann']['outside'])} außerhalb.")


def verdict_equal_bandwidth(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["nw"] == 6][0]
    return "warning", (f"Bei gleicher Rauschbandbreite von {r['bandwidth_bins']} Bins liefert das Multitaper {de(r['k_eff_mt'], 1)} unabhängige Mittel, Welch mit Hann und 50 % "
                       f"Überlappung {de(r['k_eff_welch'], 1)}: **Das Multitaper senkt die Streuung nicht stärker als Welch.**")


def verdict_best(rows: list[dict]) -> tuple[str, str]:
    steep = [r for r in rows if r["background"] == "Steil" and r["n"] == max(x["n"] for x in rows)][0]
    smooth = [r for r in rows if r["background"] == "Resonanz" and r["n"] == max(x["n"] for x in rows)][0]
    return "warning", (f"Bei glattem Hintergrund (Resonanz, n = {smooth['n']}) liegt das beste Multitaper bei {de(smooth['equal'], 2)} dB gegen {de(smooth['welch'], 2)} dB für Welch; "
                       f"beim steilen Hintergrund ({de(steep['span_db'], 0)} dB Spannweite) bei {de(steep['equal'], 2)} dB (gleiche Gewichte) bzw. {de(steep['adaptive'], 2)} dB "
                       f"(adaptiv) gegen {de(steep['welch'], 2)} dB für Welch: **Welch ist dort nicht schlechter.**")


def verdict_valley(rows: list[dict]) -> tuple[str, str]:
    eq, ad = rows[3], rows[4]
    return "success", (f"Im Tal liegt das Multitaper mit gleichen Gewichten um {de(eq['bands'][2], 1)} dB über der wahren Dichte, mit adaptiven Gewichten um "
                       f"{de(ad['bands'][2], 1)} dB: **Die Gewichte holen die Leckage der höheren Taper zurück.**")


def verdict_resolution(rows: list[dict]) -> tuple[str, str]:
    mt = [r for r in rows if r["label"] == "Multitaper NW = 4"][0]
    first = next((d for d, v in zip(mt["deltas"], mt["shares"]) if v >= 0.95), None)
    text = (f"Beim Multitaper mit NW = 4 sind zwei gleich starke Linien ab {de(first, 0)} Bins Abstand in mindestens 95 % der Läufe getrennt (Rauschbandbreite 8 Bins)."
            if first is not None else "Beim Multitaper mit NW = 4 sind zwei Linien im untersuchten Bereich nicht sicher getrennt.")
    return "info", text + " **Die Auflösung folgt der Bandbreite: ab 2 NW Bins (gemessen für NW bis 4).**"


def verdict_line_test(res: dict) -> tuple[str, str]:
    fa = {r["background"]: r for r in res["false_alarm"]}
    det = [d for d in res["detect"] if d["snr"] == -10][0]
    return "success", (f"Falschalarm je Aufzeichnung bei Resonanz-Hintergrund: F-Test {de(100 * fa['Resonanz']['f'], 1)} %, Fisher-g {de(100 * fa['Resonanz']['g'], 0)} % (nominal "
                       f"{de(100 * res['alpha'], 0)} %). Bei −10 dB je Abtastwert findet der F-Test die Linie in {de(100 * det['f'], 0)} % der Läufe, Fisher-g (nur weißes Rauschen) "
                       f"in {de(100 * det['g'], 0)} %: **gegenüber Fishers g im weißen Rauschen kostet der F-Test in dieser Einstellung etwa 4 dB Empfindlichkeit.**")
