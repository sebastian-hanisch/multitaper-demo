"""Randfälle: kleinste und größte Länge, ein Taper, alle Gewichte und Hintergründe, Linien an den Reglergrenzen, Taper-Zahl gegen NW."""

import numpy as np
import pytest

import mtp_constants as C
import mtp_evaluation as ev


@pytest.mark.parametrize("weights", C.WEIGHTS)
@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_weight_and_background_runs_with_two_lines(weights, background):
    a = ev.analyse(ev.Settings(n=256, nw=3, k=5, weights=weights, background=background, lines="zwei", gap=7.0))
    assert np.isfinite(a.est).all() and (a.est > 0).all() and np.isfinite(a.welch).all() and np.isfinite(a.f_stat).all()
    assert (a.expected is None) == (weights == "adaptiv")
    if a.expected is not None:
        assert np.isfinite(a.expected).all() and (a.expected > 0).all()


def test_one_taper_has_no_f_test_and_fisher_still_works():
    a = ev.analyse(ev.Settings(n=256, nw=2, k=1))
    assert a.k == 1 and a.f_stat is None and a.f_thr is None and a.found == [] and np.isfinite(a.fisher_g)
    assert a.k_eff == 1.0 and a.std_db == pytest.approx(a.full_std_db)


def test_taper_count_is_clamped_to_two_nw():
    a = ev.analyse(ev.Settings(n=256, nw=2, k=24))
    assert a.k == 4 and a.h.shape == (4, 256)


def test_smallest_and_largest_records_with_the_extreme_nw():
    for n, nw in ((128, 12), (4096, 2)):
        a = ev.analyse(ev.Settings(n=n, nw=nw, k=2 * nw - 1))
        assert a.h.shape == (2 * nw - 1, n) and np.isfinite(a.est).all()
        assert a.welch_k >= 1 and a.ops == (2 * nw - 1) * (n // 2) * int(round(np.log2(n)))


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
@pytest.mark.parametrize("gap", [C.GAP_MIN, C.GAP_MAX])
def test_lines_at_the_slider_limits_stay_below_nyquist(line, gap):
    for n in (128, 4096):
        s = ev.Settings(n=n, line=line, gap=gap, lines="zwei")
        assert 0 < s.f1_hz < C.FS / 2 and 0 < s.f2_hz < C.FS / 2 and abs(s.f2_hz - s.f1_hz) == pytest.approx(gap * C.FS / n)
        assert np.isfinite(ev.analyse(s).est).all()


def test_extreme_levels_and_seed_limits():
    for rel in (C.REL_MIN, C.REL_MAX):
        for snr in (C.SNR_MIN, C.SNR_MAX):
            assert np.isfinite(ev.analyse(ev.Settings(n=256, nw=3, k=5, lines="zwei", rel=rel, snr=snr)).f_stat).all()
    for seed in (0, C.SEED_MAX):
        assert np.isfinite(ev.analyse(ev.Settings(n=256, nw=3, k=5, seed=seed)).est).all()


def test_no_lines_means_no_resolution_and_the_expected_value_is_the_noise_term():
    a = ev.analyse(ev.Settings(n=256, nw=3, k=5, lines="keine"))
    assert a.resolved is None and a.welch_resolved is None
    a1 = ev.analyse(ev.Settings(n=256, nw=3, k=5, lines="eine"))
    assert a1.resolved is None and a1.expected.max() > a.expected.max()


def test_a_strong_single_line_is_found_at_the_right_frequency():
    s = ev.Settings(n=512, nw=4, k=7, lines="eine", line=0.3, snr=0, background="weiß")
    a = ev.analyse(s)
    assert len(a.found) == 1 and abs(a.found[0] - s.f1_hz) < C.FS / s.n


def test_f_test_ignores_the_edge_bins():
    a = ev.analyse(ev.Settings(n=256, nw=3, k=5, lines="eine", line=C.LINE_MIN, snr=20, background="weiß"))
    assert all(f > 8 * C.FS / 256 for f in a.found)
