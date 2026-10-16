"""Vergleichsverfahren aus Stück 3 (Welch mit Hann, 50 % Überlappung) und Hilfsfunktionen (wortgleich aus `bartlett-welch-demo`).

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Fenster sind periodisch (DFT-gerade).
"""

import math

import numpy as np

DB = 10.0 / math.log(10.0)
NAMES = ("Rechteck", "Hann", "Hamming", "Blackman")
COSINE_SUM = {
    "Hann": (0.5, 0.5),
    "Hamming": (0.54, 0.46),
    "Blackman": (0.42, 0.5, 0.08),
}


def window(name: str, n: int) -> np.ndarray:
    """Periodisches Fenster der Länge n."""
    if name == "Rechteck":
        return np.ones(n)
    if name in COSINE_SUM:
        m = np.arange(n)
        w = np.zeros(n)
        for k, a in enumerate(COSINE_SUM[name]):
            w += (-1) ** k * a * np.cos(2.0 * np.pi * k * m / n)
        return w
    raise ValueError(f"unbekanntes Fenster: {name}")


def enbw(w) -> float:
    """Äquivalente Rauschbandbreite in Bins: n sum(w^2) / (sum w)^2."""
    w = np.asarray(w, dtype=float)
    return float(len(w) * (w ** 2).sum() / w.sum() ** 2)


def n_segments(n: int, seg: int, hop: int) -> int:
    """Zahl der vollständigen Segmente der Länge seg im Abstand hop in einer Folge der Länge n."""
    if seg > n or hop < 1:
        raise ValueError("Segment länger als die Folge oder Schrittweite < 1")
    return (n - seg) // hop + 1


def hop_for(seg: int, overlap: float) -> int:
    """Schrittweite in Abtastwerten für den Überlappungsanteil overlap in [0, 1)."""
    return max(1, int(round(seg * (1.0 - overlap))))


def welch(x, seg: int, hop: int, w, fs: float, nfft: int | None = None):
    """Mittel der gefensterten Segment-Periodogramme (Leistungsdichte je Hz, zweiseitig, Bins 0 .. nfft / 2) und Zahl der Segmente.

    x darf mehrere Läufe haben (letzte Achse = Zeit). Skalierung je Segment: |X|^2 / (fs sum w^2); Rechteck ohne Überlappung ist Bartlett.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(w) != seg:
        raise ValueError("Fensterlänge ungleich Segmentlänge")
    n = x.shape[-1]
    k = n_segments(n, seg, hop)
    segs = np.lib.stride_tricks.sliding_window_view(x, seg, axis=-1)[..., ::hop, :]
    p = np.abs(np.fft.rfft(segs * w, nfft or seg, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    return p.mean(axis=-2), k


def segment_correlation(w, hop: int, j: int) -> float:
    """Korrelation der Periodogramme zweier Segmente im Abstand j Schritten für weißes Gauß-Rauschen: (sum w_m w_{m + j hop})^2 / (sum w^2)^2."""
    w = np.asarray(w, dtype=float)
    sh = j * hop
    if sh >= len(w):
        return 0.0
    return float(((w[:len(w) - sh] * w[sh:]).sum() / (w ** 2).sum()) ** 2)


def variance_factor(w, hop: int, k: int) -> float:
    """Var(Mittel) / S^2 für weißes Gauß-Rauschen: (1 / K) (1 + 2 sum_{j<K} (1 - j / K) rho_j). Ohne Überlappung (Bartlett): 1 / K.

    Gilt für Frequenzen fern von 0 und der Nyquist-Frequenz; der Spiegelterm |E[X_a X_b]|^2 der reellen Folge ist weggelassen (bei Länge 16 unter 2e-4 relativ)."""
    total = 1.0
    for j in range(1, k):
        total += 2.0 * (1.0 - j / k) * segment_correlation(w, hop, j)
    return total / k


def k_effective(w, hop: int, k: int) -> float:
    """Äquivalente Zahl unabhängiger Mittel: 1 / Streufaktor (= 1 / CV^2)."""
    return 1.0 / variance_factor(w, hop, k)


def digamma(x: float) -> float:
    """Digamma-Funktion für x > 0 (Rekursion bis x >= 15, dann asymptotische Reihe)."""
    r = 0.0
    while x < 15.0:
        r -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - x2 * (1.0 / 12 - x2 * (1.0 / 120 - x2 * (1.0 / 252 - x2 * (1.0 / 240))))


def trigamma(x: float) -> float:
    """Trigamma-Funktion für x > 0."""
    r = 0.0
    while x < 15.0:
        r += 1.0 / (x * x)
        x += 1.0
    x2 = 1.0 / (x * x)
    return r + 1.0 / x + 0.5 * x2 + (1.0 / x) * x2 * (1.0 / 6 - x2 * (1.0 / 30 - x2 * (1.0 / 42 - x2 * (1.0 / 30))))


def log_bias_db(k: float) -> float:
    """Mittelwert von 10 log10 des Mittels von k Exponentialwerten gegen 10 log10 des Erwartungswerts: DB (psi(k) - ln k); k = 1: -2,51 dB."""
    return DB * (digamma(k) - math.log(k))


def log_std_db(k: float) -> float:
    """Standardabweichung von 10 log10 des Mittels: DB sqrt(psi'(k)); k = 1: 5,57 dB."""
    return DB * math.sqrt(trigamma(k))
