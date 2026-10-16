"""Gezielte Tests für Stellen, die der Mutationstest als ungeprüft meldete (Grenzen, Randfälle, Gleichstände, unabhängige Referenzfassungen)."""

import numpy as np
import pytest

import mtp_constants as C
import mtp_evaluation as ev
import mtp_spectrum as S
import mtp_taper as T


def test_inner_bins_skip_eight_at_each_end():
    assert ev._inner(100) == slice(8, 92) and list(range(20))[ev._inner(20)] == [8, 9, 10, 11]


def test_peaks_above_edges_ties_order_and_distance():
    v = [0.0, 5.0, 1.0, 0.0, 4.0, 4.0, 0.0, 6.0, 0.0]
    assert ev._peaks_above(v, 0.5, 1) == [1, 5, 7]                             # Plateau: der letzte Punkt zählt (>= links, > rechts), Ränder zählen nie
    assert ev._peaks_above([9.0, 1.0, 9.0], 0.5, 1) == []                      # Randpunkte sind keine Gipfel
    assert ev._peaks_above([0.0, 1.0, 0.0], 1.0, 1) == []                      # Schwelle ist strikt
    assert ev._peaks_above([0.0, 1.0, 0.0], 0.99, 1) == [1]
    assert ev._peaks_above([0.0, 2.0, 0.0, 2.0, 0.0], 0.5, 1) == [1, 3]       # gleich hohe Gipfel
    assert ev._peaks_above([0.0, 3.0, 0.0, 2.0, 0.0], 0.5, 2) == [1, 3]       # Abstand genau min_distance bleibt
    assert ev._peaks_above([0.0, 3.0, 0.0, 2.0, 0.0], 0.5, 3) == [1]          # näher als min_distance: der kleinere fällt weg
    assert ev._peaks_above([0.0, 2.0, 0.0, 3.0, 0.0], 0.5, 3) == [3]          # der höhere bleibt, auch wenn er hinten liegt
    assert ev._peaks_above([0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0], 0.5, 1) == [1, 3, 5]   # aufsteigend sortiert zurück
    assert ev._peaks_above([0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 3.0, 0.0], 0.1, 3) == [1, 4, 7]


def test_analysis_frequency_grids_and_thresholds():
    s = ev.Settings(n=256, nw=3, k=5)
    a = ev.analyse(s)
    nf = C.PADF * 256
    assert len(a.f_hz_f) == nf // 2 + 1 and a.f_hz_f[1] == pytest.approx(C.FS / nf) and a.f_hz_f[-1] == pytest.approx(C.FS / 2)
    assert len(a.f_hz) == C.PAD * 256 // 2 + 1 and a.f_hz[-1] == pytest.approx(C.FS / 2)
    assert a.f_thr == pytest.approx(S.f_threshold(5, C.ALPHA / (nf // 2)), rel=1e-12)
    p0 = np.abs(np.fft.rfft(a.x)) ** 2
    assert a.fisher_g == pytest.approx(float((p0[1:-1]).max() / p0[1:-1].sum()), rel=1e-12)
    assert a.fisher_thr == pytest.approx(S.fisher_threshold(256 // 2 - 1, C.ALPHA), rel=1e-12)
    assert a.fisher_alarm == (a.fisher_g > a.fisher_thr)


def test_window_grid_and_hann_response():
    n = 128
    a = ev.analyse(ev.Settings(n=n, nw=3, k=5))
    assert len(a.window_f) == 32 * n + 1 and a.window_f[1] == pytest.approx(1.0 / 64.0) and a.window_f[-1] == pytest.approx(n / 2.0)
    assert a.window_mt[0] == 1.0 and a.window_hann[0] == pytest.approx(1.0)
    # Hann: Antwort bei einem Bin Abstand die Hälfte der Amplitude, also ein Viertel der Leistung; bei zwei Bins null
    assert a.window_hann[64] == pytest.approx(0.25, abs=1e-9) and a.window_hann[128] == pytest.approx(0.0, abs=1e-9)


def test_f_test_works_with_exactly_two_tapers_and_not_with_one():
    a2 = ev.analyse(ev.Settings(n=256, nw=2, k=2, background="weiß", snr=10))
    a1 = ev.analyse(ev.Settings(n=256, nw=2, k=1))
    assert a2.k == 2 and a2.f_stat is not None and a2.f_thr == pytest.approx(S.f_threshold(2, C.ALPHA / (C.PADF * 256 // 2)))
    assert a1.f_stat is None


def test_f_test_ignores_exactly_the_eight_edge_bins():
    n = 256
    for bins, expect_found in ((7.5, False), (8.5, True)):
        s = ev.Settings(n=n, nw=2, k=3, lines="eine", line=bins / n, snr=25, background="weiß")
        a = ev.analyse(s)
        assert (len(a.found) == 1) == expect_found
    assert C.F_GUARD == 8 * C.PADF


def test_usable_segment_lengths_and_nw_values():
    assert ev._usable_segs(128) == [16, 32, 64, 128] and ev._usable_segs(16) == [16] and ev._usable_segs(300) == [16, 32, 64, 128, 256]
    assert ev._usable_nws(64) == [2, 3, 4, 6]                                  # 2 NW < 16
    assert ev._usable_nws(128) == [2, 3, 4, 6, 8, 12] and ev._usable_nws(48) == [2, 3, 4]


def test_band_bias_includes_the_lower_and_upper_band_edge():
    f = np.array([0.1, 0.2, 0.3, 0.4])
    est = np.array([1.0, 10.0, 100.0, 1000.0])
    out = ev._band_bias(f, est, np.ones(4), ((0.2, 0.3), (0.1, 0.1), (0.35, 0.45)))
    assert out == pytest.approx([15.0, 0.0, 30.0], abs=1e-9)


def test_resolution_verdict_accepts_exactly_ninety_five_percent():
    rows = [{"label": "Multitaper NW = 4", "deltas": [2.0, 4.0, 8.0], "shares": [0.0, 0.94, 0.95]}]
    assert "ab 8 Bins Abstand" in ev.verdict_resolution(rows)[1]


def test_spectral_window_by_a_direct_sum():
    h = T.dpss(16, 2, 3)
    nfft = 32
    w = S.spectral_window(h, nfft)
    ref = []
    for j in range(nfft // 2 + 1):
        ref.append(np.mean([abs(sum(h[k, t] * np.exp(-2j * np.pi * j * t / nfft) for t in range(16))) ** 2 for k in range(3)]))
    assert w == pytest.approx(np.array(ref) / ref[0], rel=1e-10)


def test_fisher_threshold_matches_the_closed_form_for_three_ordinates():
    # m = 3: P(g > x) = 3 (1 - x)^2 für x >= 1/2 (nur j = 1), also alpha = 0,03 bei x = 1 - sqrt(0,01) = 0,9
    assert S.fisher_threshold(3, 0.03) == pytest.approx(0.9, abs=1e-9)
    # zwei Summanden für x < 1/2: P = 3 (1 - x)^2 - 3 (1 - 2x)^2; bei x = 0,4 gleich 3 * 0,36 - 3 * 0,04 = 0,96
    assert S.fisher_threshold(3, 0.96) == pytest.approx(0.4, abs=1e-9)


def test_lines_resolved_needs_two_distinct_maxima_and_a_dip():
    spec = np.array([0.0, 4.0, 1.0, 4.0, 0.0])
    assert S.lines_resolved(spec, 1, 1.0, 3.0)                                 # Einbruch auf 1 < 0,5 * 4
    assert not S.lines_resolved(np.array([0.0, 4.0, 2.0, 4.0, 0.0]), 1, 1.0, 3.0)     # Einbruch gleich depth * Maximum zählt nicht
    assert not S.lines_resolved(np.array([0.0, 4.0, 3.0, 2.0, 0.0]), 1, 1.0, 3.0)     # kein zweiter Gipfel


def test_taper_cache_is_cleared_after_sixteen_entries():
    T._CACHE.clear()
    for n in range(8, 8 + 17):
        T.dpss(n, 1.0, 1)
    assert len(T._CACHE) == 17
    T.dpss(40, 1.0, 1)
    assert len(T._CACHE) == 1


def test_taper_rejects_series_shorter_than_three():
    for n in (1, 2):
        with pytest.raises(ValueError):
            T.dpss(n, 0.4, 1)
    assert T.dpss(3, 1, 3).shape == (3, 3)


def test_tapers_agree_for_the_tiniest_valid_series_and_for_wide_bands():
    h = T.dpss(4, 1.5, 4)
    assert h @ h.T == pytest.approx(np.eye(4), abs=1e-9)
    h = T.dpss(40, 19.5, 5)
    assert h @ h.T == pytest.approx(np.eye(5), abs=1e-8)


def test_adaptive_weights_stop_early_and_without_effect_on_the_result():
    h = T.dpss(64, 2, 3)
    lam = T.concentration(h, 2)
    x = np.random.default_rng(9).normal(size=(1, 64))
    p = np.abs(S.eigenspectra(x, h, 64)) ** 2
    s_few, _ = S.adaptive_weights(p, lam, (x ** 2).mean(axis=1), iters=1)
    s_many, _ = S.adaptive_weights(p, lam, (x ** 2).mean(axis=1), iters=50)
    assert not np.allclose(s_few, s_many, rtol=1e-6)                           # eine Iteration reicht nicht
    again, _ = S.adaptive_weights(p, lam, (x ** 2).mean(axis=1), iters=60)
    assert again == pytest.approx(s_many, rel=1e-6)
