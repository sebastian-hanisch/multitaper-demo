"""Kern: Eigenspektren, Multitaper-Schätzer (gleiche und adaptive Gewichte), harmonischer F-Test, Fisher-g-Test, exakter Erwartungswert.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Taper h sind k x n mit Energie 1; Dichten in Leistung je Hz (zweiseitig, Skala 1 / fs).
"""

import math

import numpy as np

from mtp_taper import autocorrelation


def eigenspectra(x, h, nfft: int | None = None) -> np.ndarray:
    """Eigenspektren Y_k(f) = sum_t h_k[t] x[t] exp(-2 pi i f t): Läufe (letzte Achse = Zeit) x k x Bins."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.fft.rfft(x[..., None, :] * h, nfft or x.shape[-1], axis=-1)


def multitaper_equal(x, h, fs: float, nfft: int | None = None) -> np.ndarray:
    """Mittel der k Eigenspektren mit gleichen Gewichten: (1 / k) sum_k |Y_k|^2 / fs."""
    return (np.abs(eigenspectra(x, h, nfft)) ** 2).mean(axis=-2) / fs


def adaptive_weights(p, lam, variance, iters: int = 30):
    """Thomson-Gewichte für die Eigenspektren p (Läufe x k x Bins, Dichte): d_k^2 = lam_k (S / (lam_k S + (1 - lam_k) sigma^2))^2 mit S als Fixpunkt.

    `variance`: Leistung je Hz des Rauschens, sigma^2 / fs (je Lauf). Rückgabe (S, d2).
    """
    lam3 = np.asarray(lam, dtype=float)[None, :, None]
    var = np.asarray(variance, dtype=float).reshape(-1, 1, 1)
    s = p[:, :2, :].mean(axis=1)
    d2 = None
    for _ in range(iters):
        d2 = lam3 * (s[:, None, :] / (lam3 * s[:, None, :] + (1.0 - lam3) * var)) ** 2
        s_new = (d2 * p).sum(axis=1) / d2.sum(axis=1)
        done = float(np.max(np.abs(s_new - s) / s)) < 1e-7
        s = s_new
        if done:
            break
    return s, d2


def multitaper_adaptive(x, h, lam, fs: float, nfft: int | None = None) -> np.ndarray:
    """Multitaper-Schätzer mit adaptiven Gewichten (Thomson 1982); x darf mehrere Läufe haben."""
    x2 = np.atleast_2d(np.asarray(x, dtype=float))
    p = np.abs(eigenspectra(x2, h, nfft)) ** 2 / fs
    s, _ = adaptive_weights(p, lam, (x2 ** 2).mean(axis=-1) / fs)
    return s[0] if np.ndim(x) == 1 else s


def f_statistic(x, h, nfft: int | None = None) -> np.ndarray:
    """Harmonischer F-Test (Thomson): F(f) = (k - 1) |mu|^2 sum U_k(0)^2 / sum |Y_k - mu U_k(0)|^2 mit mu(f) = sum U_k(0) Y_k / sum U_k(0)^2, U_k(0) = sum h_k.

    Unter H0 (keine Linie bei f) F(2, 2k - 2)-verteilt. Muss auf einem aufgefüllten Raster ausgewertet werden, wenn die Linie zwischen Bins liegen kann.
    """
    h = np.asarray(h, dtype=float)
    k = h.shape[0]
    u0 = h.sum(axis=1)
    y = eigenspectra(x, h, nfft)
    mu = (u0[:, None] * y).sum(axis=-2) / (u0 ** 2).sum()
    res = y - mu[..., None, :] * u0[:, None]
    return (k - 1) * np.abs(mu) ** 2 * (u0 ** 2).sum() / (np.abs(res) ** 2).sum(axis=-2)


def f_threshold(k: int, tail: float) -> float:
    """Schwelle x mit P(F(2, 2k - 2) > x) = tail: x = (k - 1) (tail^(-1 / (k - 1)) - 1)."""
    if k < 2:
        raise ValueError("Der F-Test braucht mindestens zwei Taper")
    return (k - 1) * (tail ** (-1.0 / (k - 1)) - 1.0)


def fisher_g(p) -> np.ndarray:
    """Fishers g = max / Summe der Periodogrammwerte (letzte Achse)."""
    p = np.asarray(p, dtype=float)
    return p.max(axis=-1) / p.sum(axis=-1)


def fisher_threshold(m: int, alpha: float) -> float:
    """Schwelle für Fishers g bei m Ordinaten (weißes Gauß-Rauschen): P(g > x) = sum_{j=1}^{floor(1/x)} (-1)^(j-1) C(m, j) (1 - j x)^(m - 1) = alpha."""
    def tail(x):
        return sum((-1) ** (j - 1) * math.comb(m, j) * max(0.0, 1.0 - j * x) ** (m - 1) for j in range(1, int(1.0 / x) + 1))
    lo, hi = 1.0 / m, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_multitaper(h, r, fs: float, nfft: int | None = None) -> np.ndarray:
    """E[gleichgewichtetes Multitaper-Spektrum] für Rauschen mit Autokovarianz r (Lags 0 .. n - 1) exakt:
    (1 / k) sum_k sum_{|tau| < n} r(tau) c_k(tau) exp(-i omega tau) / fs mit den Taper-Autokorrelationen c_k; Bins 0 .. nfft / 2."""
    h = np.atleast_2d(np.asarray(h, dtype=float))
    n = h.shape[1]
    nfft = nfft or n
    c = autocorrelation(h)                                   # k x n
    s = (np.asarray(r, dtype=float)[None, :n] * c).mean(axis=0)
    return (2.0 * np.real(np.fft.rfft(s, nfft)) - s[0]) / fs


def spectral_window(h, nfft: int) -> np.ndarray:
    """Mittleres Betragsquadrat der Taperantworten (normiert auf den Wert 1 bei f = 0), Bins 0 .. nfft / 2: das Spektralfenster des Schätzers."""
    w = (np.abs(np.fft.rfft(np.asarray(h, dtype=float), nfft, axis=1)) ** 2).mean(axis=0)
    return w / w[0]


def lines_resolved(spec, pad: int, f1_bins: float, f2_bins: float, depth: float = 0.5) -> bool:
    """Zwei Linien gelten als aufgelöst, wenn das Spektrum um jede der beiden Linienlagen (Fenster ± halber Abstand) ein Maximum hat und das Minimum zwischen den
    Maxima unter depth mal dem kleineren Maximum liegt (Einbruch von mindestens 3 dB bei depth = 0,5). Frequenzen in Bins, das Spektrum liegt auf dem Raster 1 / pad."""
    spec = np.asarray(spec, dtype=float)
    lo_bin, hi_bin = sorted((f1_bins, f2_bins))
    c1, c2 = int(round(lo_bin * pad)), int(round(hi_bin * pad))
    half = max(1, int(round((c2 - c1) / 2.0)))
    seg1 = spec[max(c1 - half, 0):c1 + half + 1]
    seg2 = spec[max(c2 - half, 0):c2 + half + 1]
    i1 = max(c1 - half, 0) + int(np.argmax(seg1))
    i2 = max(c2 - half, 0) + int(np.argmax(seg2))
    if i1 >= i2:
        return False
    return bool(spec[i1:i2 + 1].min() < depth * min(spec[i1], spec[i2]))


def bandwidth_hz(n: int, nw: float, fs: float) -> float:
    """Volle Bandbreite 2 W = 2 nw fs / n in Hz (gleich der Rauschbandbreite bei gleichen Gewichten und k gleich 2 nw)."""
    return 2.0 * nw * fs / n


def ops_fft(n: int) -> int:
    """Komplexe Multiplikationen einer FFT der Länge n in der Radix-2-Näherung (n / 2) log2 n; exakt für Zweierpotenzen."""
    if n < 2:
        raise ValueError("FFT-Länge unter 2")
    return int(round(n / 2.0 * math.log2(n)))


def ops_multitaper(n: int, k: int) -> int:
    """Komplexe Multiplikationen der k FFTs der Länge n."""
    return k * ops_fft(n)
