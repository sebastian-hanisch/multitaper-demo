"""Multitaper-Kern: Eigenspektren, Schätzer, adaptive Gewichte, F-Test, Fisher-g, Erwartungswert, Auflösungskriterium, Aufwand."""

import math

import numpy as np
import pytest

import mtp_spectrum as S
import mtp_taper as T


def test_eigenspectra_by_a_direct_loop(mini_tapers):
    x = np.array([1.0, -2.0, 0.5])
    y = S.eigenspectra(x, mini_tapers, 8)
    assert y.shape == (3, 5)
    for k in range(3):
        for j in range(5):
            ref = sum(mini_tapers[k, t] * x[t] * np.exp(-2j * np.pi * j * t / 8) for t in range(3))
            assert y[k, j] == pytest.approx(ref, abs=1e-12)


def test_equal_weight_estimate_is_the_mean_of_the_squared_eigenspectra(mini_tapers):
    x = np.array([1.0, -2.0, 0.5])
    est = S.multitaper_equal(x, mini_tapers, 2.0, 4)
    y = S.eigenspectra(x, mini_tapers, 4)
    assert est == pytest.approx((np.abs(y) ** 2).mean(axis=0) / 2.0, abs=1e-12)
    runs = S.multitaper_equal(np.vstack([x, 2 * x]), mini_tapers, 1.0, 4)
    assert runs.shape == (2, 3) and runs[1] == pytest.approx(4 * runs[0], abs=1e-12)


def test_f_statistic_by_a_direct_loop():
    rng = np.random.default_rng(5)
    n, k = 16, 4
    h = T.dpss(n, 2, k)
    x = rng.normal(size=n) + 0.8 * np.cos(2 * np.pi * 3 * np.arange(n) / n)
    f = S.f_statistic(x, h, 32)
    u0 = h.sum(axis=1)
    for j in (0, 5, 6, 7, 16):
        y = np.array([sum(h[m, t] * x[t] * np.exp(-2j * np.pi * j * t / 32) for t in range(n)) for m in range(k)])
        mu = (u0 * y).sum() / (u0 ** 2).sum()
        ref = (k - 1) * abs(mu) ** 2 * (u0 ** 2).sum() / (np.abs(y - mu * u0) ** 2).sum()
        assert f[j] == pytest.approx(ref, rel=1e-10)


def test_f_statistic_is_large_for_a_line_on_the_grid_and_small_for_noise():
    n = 256
    h = T.dpss(n, 4, 7)
    t = np.arange(n)
    rng = np.random.default_rng(6)
    noise = rng.normal(size=n)
    x = 2.0 * np.cos(2 * np.pi * 40 * t / n + 0.3) + noise
    f = S.f_statistic(x, h, 8 * n)
    assert f[8 * 40] > 100 and np.median(f) < 2 and int(np.argmax(f)) == 8 * 40


def test_f_threshold_closed_form():
    assert S.f_threshold(2, 0.5) == pytest.approx(1.0, abs=1e-12)              # F(2, 2): P(F > x) = 1 / (1 + x)
    assert S.f_threshold(7, 0.01) == pytest.approx(6.926608140191303, abs=1e-9)
    assert S.f_threshold(7, 1e-5) > S.f_threshold(7, 1e-2) > S.f_threshold(7, 0.5) > 0
    with pytest.raises(ValueError):
        S.f_threshold(1, 0.01)


def test_fisher_threshold_and_statistic():
    # m = 2: P(g > x) = 2 (1 - x) für x >= 1/2, also alpha = 0,1 bei x = 0,95
    assert S.fisher_threshold(2, 0.1) == pytest.approx(0.95, abs=1e-9)
    assert S.fisher_threshold(511, 0.01) == pytest.approx(0.021026, abs=1e-5)
    assert S.fisher_g([1.0, 1.0, 2.0]) == pytest.approx(0.5)
    assert S.fisher_g([[1.0, 3.0], [2.0, 2.0]]) == pytest.approx([0.75, 0.5])


def test_adaptive_weights_limits():
    h = T.dpss(64, 2, 3)
    lam = T.concentration(h, 2)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 64))
    p = np.abs(S.eigenspectra(x, h, 64)) ** 2
    var = (x ** 2).mean(axis=1)
    s, d2 = S.adaptive_weights(p, lam, var)
    assert s.shape == (2, 33) and d2.shape == (2, 3, 33)
    assert np.all(d2 > 0) and np.all(d2 <= 1.0 / lam[None, :, None] + 1e-12)       # d^2 = lam (S / (lam S + ...))^2 <= 1 / lam
    # Fixpunkt: erneutes Anwenden der Gleichung ändert S nicht
    again = (d2 * p).sum(axis=1) / d2.sum(axis=1)
    assert again == pytest.approx(s, rel=1e-5)
    # alle lambda gleich 1: gleiche Gewichte
    s1, d1 = S.adaptive_weights(p, np.ones(3), var)
    assert s1 == pytest.approx(p.mean(axis=1), rel=1e-9) and np.allclose(d1, 1.0)


def test_adaptive_estimator_shapes_and_single_series():
    h = T.dpss(64, 2, 3)
    lam = T.concentration(h, 2)
    x = np.random.default_rng(8).normal(size=64)
    est = S.multitaper_adaptive(x, h, lam, 1.0, 64)
    assert est.shape == (33,) and np.all(est > 0)
    two = S.multitaper_adaptive(np.vstack([x, x]), h, lam, 1.0, 64)
    assert two.shape == (2, 33) and two[0] == pytest.approx(est, rel=1e-9)


def test_expected_value_by_the_double_sum():
    n, k = 12, 2
    h = T.dpss(n, 2, k)
    r = 0.7 ** np.arange(n)
    nfft = 24
    exp = S.expected_multitaper(h, r, 2.0, nfft)
    ref = np.zeros(nfft // 2 + 1)
    for j in range(len(ref)):
        om = 2 * np.pi * j / nfft
        total = 0.0
        for kk in range(k):
            for m in range(n):
                for mp in range(n):
                    total += h[kk, m] * h[kk, mp] * r[abs(m - mp)] * np.cos(om * (m - mp))
        ref[j] = total / k / 2.0
    assert exp == pytest.approx(ref, abs=1e-12)


def test_spectral_window_is_normalised_and_box_like():
    h = T.dpss(256, 4, 7)
    w = S.spectral_window(h, 64 * 256)
    assert w[0] == 1.0 and w.min() >= 0.0
    f = np.arange(len(w)) / 64.0                              # in Bins
    assert w[f <= 3.0].min() > 0.5                            # nahezu flach im Band
    assert w[f >= 6.0].max() < 5e-3                           # steil außerhalb


def test_lines_resolved_by_hand():
    pad = 2
    spec = np.array([1.0, 3.0, 6.0, 4.0, 2.0, 5.0, 7.0, 3.0, 1.0, 0.5])
    assert S.lines_resolved(spec, pad, 1.0, 3.0)                       # Maxima bei Index 2 und 6, Tal 2 gegen 6: 2 < 0,5 * 6
    assert not S.lines_resolved(spec, pad, 1.0, 3.0, depth=0.3)        # Einbruch auf 2 liegt nicht unter 0,3 * 6
    flat = np.array([1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0])
    assert not S.lines_resolved(flat, 1, 2.0, 4.0)
    assert not S.lines_resolved(spec, pad, 3.0, 1.0, depth=0.3)        # Reihenfolge der Linien egal
    peak = np.array([0.0, 1.0, 9.0, 1.0, 0.0])
    assert not S.lines_resolved(peak, 1, 1.0, 3.0)                     # ein Gipfel: beide Maxima fallen zusammen


def test_bandwidth_and_operation_counts():
    assert S.bandwidth_hz(1024, 4, 1000.0) == pytest.approx(7.8125)
    assert S.ops_fft(256) == 1024 and S.ops_fft(2) == 1 and S.ops_fft(192) == round(96 * math.log2(192))
    assert S.ops_multitaper(256, 7) == 7 * 1024
    with pytest.raises(ValueError):
        S.ops_fft(1)
