"""Slepian-Taper (diskrete prolate sphäroidale Folgen, DPSS) in reinem numpy.

Die k größten Eigenvektoren der Tridiagonalmatrix B mit Diagonale ((n - 1 - 2 t) / 2)^2 cos(2 pi W) und Nebendiagonale t (n - t) / 2 (Percival und Walden 1993) sind die
Taper; die Eigenwerte stehen mit der Konzentration lambda_k im Band [-W, W] nicht in Beziehung, diese wird über die Autokorrelation der Taper berechnet.
Rechnung: Eigenwerte per Sturm-Bisektion (vektorisiert über die k gesuchten), Eigenvektoren per inverser Iteration mit dem Thomas-Algorithmus.
"""

import math

import numpy as np


def _tridiagonal(n: int, nw: float):
    t = np.arange(n)
    d = ((n - 1 - 2 * t) / 2.0) ** 2 * math.cos(2.0 * math.pi * nw / n)
    e = t[1:] * (n - t[1:]) / 2.0
    return d, e


def _sturm_count(d, e2, x):
    """Zahl der Eigenwerte kleiner als x (Sturm-Folge), für ein Feld von Werten x."""
    q = d[0] - x
    count = (q < 0).astype(int)
    for j in range(1, len(d)):
        q = np.where(q == 0.0, 1e-300, q)
        q = d[j] - x - e2[j - 1] / q
        count += q < 0
    return count


def _inverse_iteration(d, e, shift, seed_vector, steps: int = 3):
    n = len(d)
    dd = d - shift
    x = seed_vector
    for _ in range(steps):
        cp = np.empty(n)
        dp = np.empty(n)
        cp[0] = e[0] / dd[0]
        dp[0] = x[0] / dd[0]
        for t in range(1, n):
            den = dd[t] - e[t - 1] * cp[t - 1]
            cp[t] = e[t] / den if t < n - 1 else 0.0
            dp[t] = (x[t] - e[t - 1] * dp[t - 1]) / den
        y = np.empty(n)
        y[-1] = dp[-1]
        for t in range(n - 2, -1, -1):
            y[t] = dp[t] - cp[t] * y[t + 1]
        x = y / np.linalg.norm(y)
    return x


_CACHE: dict = {}


def dpss(n: int, nw: float, k: int) -> np.ndarray:
    """Die k Slepian-Taper der Länge n und Zeit-Bandbreite-Produkt nw als k x n (Energie 1, symmetrische Fassung).

    Vorzeichen nach Percival und Walden: gerade Taper haben positive Summe, ungerade beginnen mit einem positiven Buckel. k <= n.
    """
    if n < 3 or k < 1 or k > n or nw <= 0 or nw >= n / 2:
        raise ValueError("Ungültige Taper-Parameter")
    key = (n, float(nw), k)
    if key in _CACHE:
        return _CACHE[key].copy()
    d, e = _tridiagonal(n, nw)
    e2 = e * e
    off = np.r_[0.0, e] + np.r_[e, 0.0]
    lo, hi = float((d - off).min()), float((d + off).max())
    a = np.full(k, lo)
    b = np.full(k, hi)
    target = n - 1 - np.arange(k)                 # der j-größte Eigenwert hat n - 1 - j Eigenwerte unter sich
    for _ in range(60):
        m = 0.5 * (a + b)
        up = _sturm_count(d, e2, m) <= target
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    lam = 0.5 * (a + b)
    out = np.empty((k, n))
    rng = np.random.default_rng(12345)
    for j in range(k):
        out[j] = _inverse_iteration(d, e, lam[j] + 1e-9 * (hi - lo), rng.normal(size=n))
    thresh = max(1e-7, 1.0 / n)
    for j in range(k):
        if j % 2 == 0:
            if out[j].sum() < 0:
                out[j] = -out[j]
        else:
            first = out[j][out[j] ** 2 > thresh][0]
            if first < 0:
                out[j] = -out[j]
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = out
    return out.copy()


def autocorrelation(h) -> np.ndarray:
    """Autokorrelation c(tau) = sum_m h[m] h[m + tau] für tau = 0 .. n - 1 jedes Tapers (Zeilen), per FFT."""
    h = np.atleast_2d(np.asarray(h, dtype=float))
    n = h.shape[1]
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(h, size, axis=1)
    return np.fft.irfft(np.abs(spec) ** 2, size, axis=1)[:, :n]


def concentration(h, nw: float) -> np.ndarray:
    """Anteil der Energie jedes Tapers im Band [-W, W], W = nw / n: lambda = 2 W c(0) + 2 sum_{tau >= 1} c(tau) sin(2 pi W tau) / (pi tau)."""
    h = np.atleast_2d(np.asarray(h, dtype=float))
    n = h.shape[1]
    w = nw / n
    c = autocorrelation(h)
    tau = np.arange(1, n)
    kernel = np.sin(2.0 * math.pi * w * tau) / (math.pi * tau)
    return 2.0 * w * c[:, 0] + 2.0 * (c[:, 1:] @ kernel)
