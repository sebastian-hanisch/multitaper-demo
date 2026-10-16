"""Gemeinsamer Signal-Generator der Linie Spektral- und Wavelet-Analyse (WORTGLEICHE Kopie in allen Repos der Linie, kein Modul-Präfix).

Version 1 (2026-10-14). Nur numpy und math, kein Streamlit. Änderungen an diesem Modul betreffen alle Repos der Linie: Schnittstelle, Namen und
Reihenfolge des Zufallsverbrauchs nur ändern, wenn alle Kopien nachgezogen werden (tools/portfolio_audit.py, SHARED_GROUPS).

Modell: n Abtastwerte mit Abtastrate fs (Hz). Das Signal ist die Summe aus Linien (Frequenz in Hz, Amplitude, Phase), einem AR-Hintergrund
(farbiges Rauschen, bestimmt durch Pole in Frequenz und Radius) und weißem Rauschen. Die zweiseitige Leistungsdichte des Rauschanteils
(Leistung je Hz, |f| <= fs/2) ist
    S(f) = sigma_ar^2 / fs / |1 - sum_k a_k exp(-i 2 pi f k / fs)|^2  +  sigma_w^2 / fs,
eine Linie der Amplitude A trägt die Leistung A^2/2. Der Erwartungswert des Periodogramms
    I(f) = |sum_n x[n] exp(-i 2 pi f n / fs)|^2 / (n fs)
ist exakt berechenbar (Fejér-Faltung über die Autokovarianz) und das Orakel der Demos.

Zufall nur über einen übergebenen `SplitMix64`. Verbrauch je `make_signal` in fester Reihenfolge: (1) falls Pole vorhanden n + L normalverteilte
Innovationen (L = Länge der abgeschnittenen Impulsantwort, Funktion `ar_length`), (2) falls white_sigma > 0 n normalverteilte Werte.
Der AR-Filter läuft über die abgeschnittene Impulsantwort per FFT-Faltung (Fehler < 1e-10 relativ), der Vorlauf ist damit die Länge L.
Plattformhinweis: Die Werte laufen über `log`, `cos`, `exp` der C-Bibliothek und können sich zwischen Plattformen in der letzten Stelle
unterscheiden; Tests dürfen Float-Werte nur mit Toleranz prüfen.
"""

import math
from dataclasses import dataclass

import numpy as np

GENERATOR_VERSION = 1

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_TAIL = 1e-12          # relative Restgröße der abgeschnittenen Impulsantwort
_L_MIN, _L_MAX = 64, 60000


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)


def ar_from_poles(poles) -> np.ndarray:
    """AR-Koeffizienten a_1..a_p (x[n] = sum a_k x[n-k] + e[n]) aus Polen (f, r): f als Bruchteil der Abtastrate in [0, 0,5], r in [0, 1).

    0 < f < 0,5 ergibt ein konjugiertes Paar (zwei Koeffizienten), f = 0 einen reellen Pol +r, f = 0,5 einen reellen Pol -r.
    """
    roots = []
    for f, r in poles:
        if not (0.0 <= f <= 0.5 and 0.0 <= r < 1.0):
            raise ValueError(f"Pol (f={f}, r={r}) außerhalb von [0, 0,5] x [0, 1)")
        if f == 0.0:
            roots.append(complex(r, 0.0))
        elif f == 0.5:
            roots.append(complex(-r, 0.0))
        else:
            z = r * complex(math.cos(2 * math.pi * f), math.sin(2 * math.pi * f))
            roots.extend([z, z.conjugate()])
    if not roots:
        return np.zeros(0)
    coef = np.poly(np.array(roots)).real          # 1 + c1 z^-1 + ... : x[n] + c1 x[n-1] + ... = e[n]
    return -coef[1:]


def ar_length(a) -> int:
    """Länge L der abgeschnittenen Impulsantwort (und des Vorlaufs): Restgröße < 1e-12 relativ zum größten Polradius."""
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return 0
    rmax = float(np.max(np.abs(np.roots(np.r_[1.0, -a]))))
    if rmax >= 1.0:
        raise ValueError("AR-Prozess nicht stationär (Polradius >= 1)")
    if rmax <= 0.0:
        return _L_MIN
    return int(min(_L_MAX, max(_L_MIN, math.ceil(math.log(_TAIL) / math.log(rmax)) + len(a))))


def ar_impulse_response(a, length: int) -> np.ndarray:
    """h[0] = 1, h[k] = sum_j a_j h[k-j]: Impulsantwort von 1/A(z), abgeschnitten auf `length`."""
    a = np.asarray(a, dtype=float)
    p = len(a)
    h = np.zeros(length)
    h[0] = 1.0
    for k in range(1, length):
        m = min(k, p)
        h[k] = float(np.dot(a[:m], h[k - 1::-1][:m]))
    return h


def ar_autocov(a, sigma2: float, nlags: int) -> np.ndarray:
    """Exakte Autokovarianz r(0..nlags-1) eines AR(p) mit Innovationsvarianz sigma2 (Yule-Walker, danach Rekursion)."""
    a = np.asarray(a, dtype=float)
    p = len(a)
    A = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    b[0] = sigma2
    for tau in range(p + 1):
        A[tau, tau] += 1.0
        for k in range(1, p + 1):
            A[tau, abs(tau - k)] -= a[k - 1]
    r = np.zeros(max(nlags, p + 1))
    r[: p + 1] = np.linalg.solve(A, b)
    for tau in range(p + 1, len(r)):
        r[tau] = float(np.dot(a, r[tau - 1:tau - p - 1:-1]))
    return r[:nlags]


def ar_unit_sigma(a) -> float:
    """Innovations-Standardabweichung, bei der die Varianz des AR-Prozesses 1 ist (ohne Pole: 1)."""
    return 1.0 / math.sqrt(float(ar_autocov(np.asarray(a, dtype=float), 1.0, 1)[0]))


def _ar_noise(a, sigma: float, n: int, rng: SplitMix64, runs: int = 0) -> np.ndarray:
    """AR-Rauschen der Länge n (runs = 0) oder als runs x n (ein Block Innovationen)."""
    length = ar_length(a)
    shape = (runs if runs else 1, n + length)
    e = sigma * rng.normal(shape[0] * shape[1]).reshape(shape)
    h = ar_impulse_response(a, length)
    nfft = 1 << int(math.ceil(math.log2(n + 2 * length)))
    x = np.fft.irfft(np.fft.rfft(e, nfft, axis=1) * np.fft.rfft(h, nfft), nfft, axis=1)[:, length:length + n]
    return x if runs else x[0]


@dataclass(frozen=True)
class Signal:
    x: np.ndarray       # Messwerte: Linien + Rauschen
    clean: np.ndarray   # nur die Linien
    noise: np.ndarray   # nur das Rauschen


def line_signal(n: int, fs: float, lines) -> np.ndarray:
    """Summe der Linien A cos(2 pi f t + phase) an den Zeitpunkten t = k / fs."""
    t = np.arange(n) / fs
    s = np.zeros(n)
    for f, amp, phase in lines:
        s += amp * np.cos(2.0 * np.pi * f * t + phase)
    return s


def make_signal(n: int, fs: float, lines, poles, ar_sigma: float, white_sigma: float, rng: SplitMix64) -> Signal:
    """Ein Signal. `poles` wie in `ar_from_poles`; ar_sigma ist die Innovations-Standardabweichung (Varianz des Hintergrunds
    = ar_sigma^2 / ar_unit_sigma(a)^2)."""
    a = ar_from_poles(poles)
    clean = line_signal(n, fs, lines)
    noise = np.zeros(n)
    if len(a) and ar_sigma > 0:
        noise = noise + _ar_noise(a, ar_sigma, n, rng)
    if white_sigma > 0:
        noise = noise + white_sigma * rng.normal(n)
    return Signal(clean + noise, clean, noise)


def make_noise_batch(runs: int, n: int, poles, ar_sigma: float, white_sigma: float, rng: SplitMix64) -> np.ndarray:
    """runs x n Rauschrealisierungen in je einem Block (Innovationen, danach weißes Rauschen); für Sweeps."""
    a = ar_from_poles(poles)
    out = np.zeros((runs, n))
    if len(a) and ar_sigma > 0:
        out = out + _ar_noise(a, ar_sigma, n, rng, runs=runs)
    if white_sigma > 0:
        out = out + white_sigma * rng.normal(runs * n).reshape(runs, n)
    return out


def true_psd(f, fs: float, poles, ar_sigma: float, white_sigma: float) -> np.ndarray:
    """Zweiseitige Leistungsdichte des Rauschanteils (Leistung je Hz) an den Frequenzen f (Hz)."""
    a = ar_from_poles(poles)
    w = 2.0 * np.pi * np.asarray(f, dtype=float) / fs
    den = np.ones(w.shape, dtype=complex)
    for k, ak in enumerate(a, start=1):
        den = den - ak * np.exp(-1j * w * k)
    ar = ar_sigma ** 2 / fs / np.abs(den) ** 2 if len(a) else np.zeros(w.shape)
    return ar + white_sigma ** 2 / fs


def noise_autocov(poles, ar_sigma: float, white_sigma: float, nlags: int) -> np.ndarray:
    """Autokovarianz r(0..nlags-1) des Rauschanteils (AR plus weiß)."""
    a = ar_from_poles(poles)
    r = ar_autocov(a, ar_sigma ** 2, nlags) if len(a) and ar_sigma > 0 else np.zeros(nlags)
    r = r.copy()
    r[0] += white_sigma ** 2
    return r


def expected_periodogram(f, fs: float, r, s=None) -> np.ndarray:
    """E[I(f)] des unverzerrten Rechteckfenster-Periodogramms exakt: n = len(r), r die Autokovarianz des Rauschens (Lags 0..n-1), s die
    deterministische Folge (Linien, feste Phase) oder None.

    E[|X|^2] = |sum s e^{-iwn}|^2 + sum_tau (n - |tau|) r(tau) cos(w tau); geteilt durch n * fs.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    f = np.atleast_1d(np.asarray(f, dtype=float))
    w = 2.0 * np.pi * f / fs
    tau = np.arange(n)
    wts = (n - tau) * r
    noise = np.empty(len(f))
    for lo in range(0, len(f), 256):                      # Blöcke: höchstens 256 x n Kosinus gleichzeitig
        c = np.cos(np.outer(w[lo:lo + 256], tau))
        noise[lo:lo + 256] = wts[0] + 2.0 * (c[:, 1:] @ wts[1:])
    det = np.zeros(len(f))
    if s is not None:
        s = np.asarray(s, dtype=float)
        nn = np.arange(len(s))
        for lo in range(0, len(f), 256):
            det[lo:lo + 256] = np.abs(np.exp(-1j * np.outer(w[lo:lo + 256], nn)) @ s) ** 2
    return (det + noise) / (n * fs)
