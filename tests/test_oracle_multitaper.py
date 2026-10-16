"""Orakel (scipy und Simulation): Taper, Konzentration, F-Verteilung, Erwartungswert, K_eff = K, Welch-Vergleich."""

import math

import numpy as np
import pytest
from scipy import signal as sps
from scipy import stats
from scipy.signal import windows as spw

import mtp_constants as C
import mtp_evaluation as ev
import mtp_spectrum as S
import mtp_taper as T
import mtp_welch as H
import signal_generator as G


@pytest.mark.parametrize("n,nw,k", [(16, 2, 3), (64, 2.5, 4), (128, 4, 7), (257, 3, 5), (512, 6, 11)])
def test_dpss_and_ratios_equal_scipy(n, nw, k):
    ref, lam = spw.dpss(n, nw, Kmax=k, sym=True, norm=2, return_ratios=True)
    h = T.dpss(n, nw, k)
    assert h == pytest.approx(ref, abs=1e-9)
    assert T.concentration(h, nw) == pytest.approx(lam, abs=1e-10)


def test_f_threshold_equals_scipy_quantiles():
    for k in (2, 3, 7, 15):
        for tail in (0.5, 0.05, 0.01, 1e-4):
            assert S.f_threshold(k, tail) == pytest.approx(float(stats.f.isf(tail, 2, 2 * k - 2)), rel=1e-9)


def test_f_statistic_under_the_null_follows_f_2_2k_minus_2():
    n, k = 128, 5
    h = T.dpss(n, 3, k)
    x = G.make_noise_batch(1500, n, (), 0.0, 1.0, G.SplitMix64(51))
    f = S.f_statistic(x, h, 4 * n)[:, 40:-40:8].ravel()              # entfernte (nahezu unkorrelierte) Frequenzen
    result = stats.kstest(f, stats.f(2, 2 * k - 2).cdf)
    assert result.pvalue > 0.01


def test_f_statistic_null_distribution_holds_for_coloured_noise_too():
    n, k = 128, 5
    h = T.dpss(n, 3, k)
    poles = ((0.2, 0.9),)
    x = G.make_noise_batch(1500, n, poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0, G.SplitMix64(52))
    f = S.f_statistic(x, h, 4 * n)[:, 40:-40:8].ravel()
    # Die Näherung (lokal flaches Spektrum im Band) ist bei gefärbtem Rauschen nicht exakt: Abstand der Verteilungsfunktion unter 0,03, Falschalarm je Bin nahe Nennwert
    assert stats.kstest(f, stats.f(2, 2 * k - 2).cdf).statistic < 0.03
    assert np.mean(f > S.f_threshold(k, 0.05)) == pytest.approx(0.05, abs=0.006)
    assert np.mean(f > S.f_threshold(k, 0.01)) == pytest.approx(0.01, abs=0.003)


@pytest.mark.parametrize("background", ["weiß", "Resonanz", "Steil"])
def test_expected_value_equals_the_simulated_mean(background):
    n, nw, k, runs = 64, 3, 5, 6000
    poles, ar_sigma, white = ev.noise_spec(background)
    h = T.dpss(n, nw, k)
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(53))
    est = S.multitaper_equal(x, h, C.FS, 2 * n)
    exp = S.expected_multitaper(h, G.noise_autocov(poles, ar_sigma, white, n), C.FS, 2 * n)
    se = est.std(axis=0) / math.sqrt(runs)
    inner = slice(2, n)
    assert np.all(np.abs(est.mean(axis=0)[inner] - exp[inner]) < 4.5 * se[inner])


def test_expected_value_with_lines_equals_the_simulated_mean():
    s = ev.Settings(n=256, nw=3, k=5, lines="zwei", gap=6.0, snr=0, rel=-6, background="Resonanz", seed=9)
    a = ev.analyse(s)
    poles, ar_sigma, white = ev.noise_spec(s.background)
    sig, _ = ev.build_signal(s)
    x = sig.clean[None, :] + G.make_noise_batch(3000, s.n, poles, ar_sigma, white, G.SplitMix64(54))
    est = S.multitaper_equal(x, a.h, C.FS, C.PAD * s.n)
    se = est.std(axis=0) / math.sqrt(3000)
    inner = slice(8, len(est[0]) - 8)
    assert np.all(np.abs(est.mean(axis=0)[inner] - a.expected[inner]) < 4.5 * se[inner] + 1e-12)


def test_independent_means_equal_k_for_white_noise_with_equal_weights():
    n, runs = 256, 4000
    x = G.make_noise_batch(runs, n, (), 0.0, 1.0, G.SplitMix64(55))
    for nw in (2, 3, 4):
        k = 2 * nw - 1
        est = S.multitaper_equal(x, T.dpss(n, nw, k), 1.0)
        inner = est[:, 8:-8]
        cv2 = ((inner.std(axis=0) / inner.mean(axis=0)) ** 2).mean()
        assert 1.0 / cv2 == pytest.approx(k, rel=0.04)


def test_the_comparator_welch_equals_scipy_welch():
    x = G.SplitMix64(5).normal(1000)
    seg = 192
    w = H.window("Hann", seg)
    est, k = H.welch(x, seg, 96, w, 2.0)
    f, p = sps.welch(x, fs=2.0, window=w, nperseg=seg, noverlap=96, detrend=False, scaling="density", return_onesided=True)
    p = p.copy()
    p[1:-1] /= 2.0
    assert est == pytest.approx(p, rel=1e-10, abs=1e-14) and k == (1000 - seg) // 96 + 1


def test_hann_window_equals_scipy_periodic_window():
    for n in (8, 31, 256):
        assert H.window("Hann", n) == pytest.approx(sps.get_window("hann", n, fftbins=True), abs=1e-12)
