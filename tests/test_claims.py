"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import math
from pathlib import Path

import numpy as np
import pytest

import mtp_constants as C
import mtp_evaluation as ev
from sweeps_cache import get
import mtp_spectrum as S
import mtp_taper as T


def preset(name, **over):
    return ev.Settings(**{**C.PRESETS[name], **over})


def row(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())][0]


def test_equal_bandwidth_preset_welch_has_more_independent_means():
    a = ev.analyse(preset("Gleiche Bandbreite"))
    assert a.k == 7 and a.welch_k_eff > a.k_eff and 8.0 < a.welch_k_eff < 9.0
    assert a.bandwidth_hz == pytest.approx(a.welch_enbw_hz, rel=1e-9)             # gleiche Rauschbandbreite (8 Bins)


def test_steep_preset_valley_is_too_high_by_three_point_two_decibels_and_adaptive_weights_fix_it():
    s = preset("Steiler Hintergrund")
    a = ev.analyse(s)
    valley = a.f_hz >= 450.0
    assert 10 * math.log10(float(np.mean(a.expected[valley] / a.true_density[valley]))) == pytest.approx(3.2, abs=0.3)
    ad = ev.analyse(ev.Settings(**{**C.PRESETS["Steiler Hintergrund"], "weights": "adaptiv"}))
    assert ad.expected is None
    runs = get("valley")
    assert row(runs, label="Multitaper NW = 4, gleich")["bands"][2] > 3.0 > abs(row(runs, label="Multitaper NW = 4, adaptiv")["bands"][2])


def test_two_lines_preset_is_resolved_at_the_bandwidth_and_not_with_wider_bands():
    s = preset("Zwei Linien an der Bandgrenze")
    assert s.gap == 2 * s.nw and ev.analyse(s).resolved is True
    for nw in (5, 6):
        a = ev.analyse(ev.Settings(**{**C.PRESETS["Zwei Linien an der Bandgrenze"], "nw": nw, "k": 2 * nw - 1}))
        assert a.resolved is False


def test_coloured_noise_preset_f_test_finds_the_line_and_fisher_g_alarms_without_a_line():
    s = preset("Linie im gefärbten Rauschen")
    a = ev.analyse(s)
    assert len(a.found) == 1 and abs(a.found[0] - s.f1_hz) < 2.0
    none = ev.analyse(ev.Settings(**{**C.PRESETS["Linie im gefärbten Rauschen"], "lines": "keine"}))
    assert none.fisher_alarm is True and none.found == []


def test_short_record_preset_numbers():
    s = preset("Kurze Aufzeichnung")
    assert s.n == 128 and s.welch_seg == 32
    best = row(get("best_vs_best"), n=128, background="Resonanz")
    assert best["equal"] < best["welch"] and (best["welch"] - best["equal"]) / best["welch"] == pytest.approx(0.09, abs=0.02)


def test_first_screen_claims_about_the_tapers():
    # „die ersten 2NW - 1 Taper haben mindestens 92 % ihrer Energie in [-W, W] (das erste mehr als 99,99 %); Taper Nr. 2NW nur noch etwa 70 %"
    for nw in (2, 3, 4, 6):
        lam = T.concentration(T.dpss(1024, nw, 2 * nw), nw)
        assert lam[:2 * nw - 1].min() > 0.92 and lam[0] > 0.9999
        assert 0.65 < lam[2 * nw - 1] < 0.75


@pytest.mark.parametrize("nw", [8, 10, 12])
def test_taper_concentration_for_large_nw_is_at_least_ninety_percent_but_not_ninety_two(nw):
    """App: mindestens 90 % (für NW bis 6 mindestens 92 %); der Regler bietet NW bis 12."""
    lam = T.concentration(T.dpss(1024, nw, 2 * nw), nw)
    assert lam[:2 * nw - 1].min() > 0.90 and lam[:2 * nw - 1].min() < 0.92 and lam[0] > 0.9999


def test_mean_energy_outside_the_band_of_the_first_seven_tapers_is_about_one_percent():
    """App-Einleitung: bei den ersten 2NW - 1 Tapern liegt im Mittel rund 1 % der Energie außerhalb des Bandes (NW = 4, n = 1024)."""
    lam = T.concentration(T.dpss(1024, 4, 7), 4)
    assert float(np.mean(1.0 - lam)) == pytest.approx(0.0102, abs=0.001)


def _noise_bandwidth_bins(n, nw, k):
    """(Integral der Fensterantwort)^2 / Integral des Quadrats, in Bins f_s / n, für gleich gewichtete Taper."""
    h = T.dpss(n, nw, k)
    c = sum(np.correlate(hk, hk, "full") for hk in h) / k
    return float(n * c[n - 1] ** 2 / (c ** 2).sum())


def test_noise_bandwidth_is_two_nw_bins_only_for_k_near_two_nw_minus_one():
    """App-Kachel Rauschbandbreite: näherungsweise 2·NW Bins bei K = 2NW - 1 bis 2NW, sonst schmaler (NW = 4: K = 1: 2,85; 4: 5,85; 7: 7,81; 8: 8,66; NW = 2, K = 3: 3,70)."""
    got = [_noise_bandwidth_bins(1024, 4, k) for k in (1, 4, 7, 8)]
    assert got == pytest.approx([2.85, 5.85, 7.81, 8.66], abs=0.02)
    assert _noise_bandwidth_bins(1024, 2, 3) == pytest.approx(3.70, abs=0.02)
    assert got[0] < 0.4 * 8.0 and 0.95 * 8.0 < got[2] < 8.0 < got[3]


def test_welch_on_a_finer_segment_grid_removes_most_of_the_advantage_of_the_multitaper():
    """README: Welch in Schritten von 16 Messwerten (n = 1024) 1,14 dB gleichauf mit dem Multitaper 1,14 dB; n = 128 (Schritt 4): 2,70 gegen 2,57 dB, rund 5 %."""
    import signal_generator as G
    import mtp_welch as H

    def welch_fine(n, step):
        poles, ar_sigma, white = ev.noise_spec("Resonanz")
        x = G.make_noise_batch(C.EXP_BEST_RUNS, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 1))
        f = np.arange(n // 2 + 1) * C.FS / n
        true = G.true_psd(f, C.FS, poles, ar_sigma, white)
        sl = slice(4, n // 2 - 4)

        def rmse(est):
            e = ev.DB * np.log(est[:, sl] / true[sl])
            return float(math.sqrt((e ** 2).mean()))
        return min(rmse(H.welch(x, seg, H.hop_for(seg, 0.5), H.window("Hann", seg), C.FS, n)[0][:, :n // 2 + 1]) for seg in range(16, n // 2 + 1, step))

    best = get("best_vs_best")
    r1024, r128 = row(best, n=1024, background="Resonanz"), row(best, n=128, background="Resonanz")
    fine1024, fine128 = welch_fine(1024, 16), welch_fine(128, 4)
    assert fine1024 == pytest.approx(1.14, abs=0.01) and fine1024 == pytest.approx(r1024["equal"], abs=0.02)
    assert fine128 == pytest.approx(2.70, abs=0.02) and (fine128 - r128["equal"]) / fine128 == pytest.approx(0.049, abs=0.01)
    assert (r128["welch"] - r128["equal"]) / r128["welch"] == pytest.approx(0.091, abs=0.01)


def test_steep_background_with_fewer_tapers_the_multitaper_matches_welch():
    """README: Steil mit weniger Tapern (K = 13 bei NW = 8) 1,49 dB gegen Welch 1,47 dB; die 2,32 dB folgen aus K = 2NW - 1."""
    import signal_generator as G
    n = 1024
    poles, ar_sigma, white = ev.noise_spec("Steil")
    x = G.make_noise_batch(C.EXP_BEST_RUNS, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 1))
    f = np.arange(n // 2 + 1) * C.FS / n
    true = G.true_psd(f, C.FS, poles, ar_sigma, white)
    sl = slice(4, n // 2 - 4)
    e = ev.DB * np.log(S.multitaper_equal(x, T.dpss(n, 8, 13), C.FS, n)[:, :n // 2 + 1][:, sl] / true[sl])
    rm = float(math.sqrt((e ** 2).mean()))
    welch = row(get("best_vs_best"), n=1024, background="Steil")
    assert rm == pytest.approx(1.49, abs=0.02) and rm == pytest.approx(welch["welch"], abs=0.05) and rm < 0.7 * welch["equal"]


def test_f_test_false_alarm_is_not_valid_for_two_sharp_resonances():
    """README/App: in 'Zwei scharfe Resonanzen' (Pole 0,998) rund 22 % statt 1 %; Verteilungsfunktion weicht an den Resonanzen um mehr als 0,5 von F(2, 2K - 2) ab;
    in 'Resonanz' und 'Steil' 0,05 bis 0,25 an den Gipfeln, gepoolter Falschalarm unter dem Nennwert."""
    import signal_generator as G
    n, nw, k, runs = 1024, 4, 7, 2000
    h = T.dpss(n, nw, k)
    nfft = C.PADF * n
    thr = S.f_threshold(k, C.ALPHA / (nfft // 2))

    def run(bg):
        poles, ar_sigma, white = ev.noise_spec(bg)
        xs = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 4))
        return S.f_statistic(xs, h, nfft)

    def ks(F, frac):
        from scipy import stats
        return float(stats.kstest(F[:, int(round(frac * nfft))], stats.f(2, 2 * k - 2).cdf).statistic)

    sharp = run("Zwei scharfe Resonanzen")
    assert float((sharp[:, C.F_GUARD:-C.F_GUARD].max(axis=1) > thr).mean()) == pytest.approx(0.22, abs=0.04)
    assert ks(sharp, 0.1) > 0.5 and ks(sharp, 0.3) > 0.5
    for bg in ("Resonanz", "Steil"):
        F = run(bg)
        assert float((F[:, C.F_GUARD:-C.F_GUARD].max(axis=1) > thr).mean()) < 0.01
    assert 0.03 < ks(run("Resonanz"), 0.2) < 0.12
    assert 0.15 < ks(run("Steil"), 0.2) < 0.30


def test_orthogonal_tapers_give_k_independent_means_but_welch_gets_more_at_equal_bandwidth():
    rows = get("equal_bandwidth")
    for r in rows:
        assert r["k_eff_mt"] == pytest.approx(r["k"], rel=0.04)
        assert r["k_eff_welch"] > r["k_eff_mt"] * 1.1
        assert r["k_eff_welch"] == pytest.approx(r["k_eff_welch_formula"], rel=0.03)
        assert r["std_welch"] < r["std_mt"]


def test_best_against_best_pattern():
    rows = get("best_vs_best")
    for r in rows:
        if r["background"] == "Steil":
            assert r["equal"] > 1.4 * r["welch"] and r["adaptive"] < r["equal"]
        if r["background"] == "Resonanz":
            assert r["equal"] < r["welch"] and (r["welch"] - r["equal"]) / r["welch"] < 0.10
    assert all(r["span_db"] > 30 for r in rows)


def test_multitaper_resolves_from_the_bandwidth_on_and_hann_welch_needs_more():
    rows = {r["label"]: r for r in get("resolution")}
    deltas = C.EXP_RES_DELTAS
    for nw, label in ((2, "Multitaper NW = 2"), (3, "Multitaper NW = 3"), (4, "Multitaper NW = 4")):
        shares = dict(zip(deltas, rows[label]["shares"]))
        assert shares[2.0 * nw] >= 0.95 and shares[2.0 * nw - 2.0] == 0.0
    # Welch Hann mit Rauschbandbreite 6 Bins (L = 256) braucht 12 Bins, das Multitaper mit NW = 3 (ebenfalls 6 Bins) nur 6
    assert dict(zip(deltas, rows["Multitaper NW = 3"]["shares"]))[6.0] == 1.0
    assert dict(zip(deltas, rows["Welch, Hann 50 %, Länge 256"]["shares"]))[6.0] < 0.1


def test_line_test_claims():
    t = get("line_test")
    fa = {r["background"]: r for r in t["false_alarm"]}
    assert all(r["f"] <= 0.02 for r in t["false_alarm"])
    assert fa["Resonanz"]["g"] > 0.9 and fa["Steil"]["g"] > 0.9 and fa["weiß"]["g"] < 0.03
    det = {d["snr"]: d for d in t["detect"]}
    assert det[-10]["f"] > 0.6 and det[-10]["g"] > 0.95 and det[-18]["f"] < 0.1 and det[-8]["f"] > 0.9
    assert all(d["f"] <= d["g"] + 0.02 for d in t["detect"])                   # gültiger Falschalarm kostet Empfindlichkeit


def test_f_threshold_closed_form_matches_the_text():
    assert S.f_threshold(7, 0.01) == pytest.approx(6.93, abs=0.01)
    assert S.f_threshold(7, C.ALPHA / (C.PADF * 1024 // 2)) == pytest.approx(45.7, abs=0.1)


def test_operation_counts_in_the_text():
    a = ev.analyse(ev.Settings(n=1024, nw=4, k=7))
    assert a.ops == 7 * 5120 and a.ops_full == 5120 and a.ops / a.ops_welch > 2.0


def test_fast_dpss_computation():
    import time
    t0 = time.time()
    T._CACHE.clear()
    T.dpss(1024, 4, 7)
    assert time.time() - t0 < 10.0


# --- README: die Zahlen der Befunde-Tabelle stehen im Text und stimmen mit den Messungen überein --------------------------------------------------


def _readme():
    return (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def test_readme_equal_bandwidth_and_best_numbers():
    text = _readme()
    bw = get("equal_bandwidth")
    assert " / ".join(ev.de(r["k_eff_mt"], 1) for r in bw) in text and " / ".join(ev.de(r["k_eff_welch"], 1) for r in bw) in text
    best = get("best_vs_best")
    r1 = row(best, n=1024, background="Resonanz")
    r2 = row(best, n=1024, background="Steil")
    assert f"{ev.de(r1['welch'], 2)} dB (Welch) gegen {ev.de(r1['equal'], 2)} dB (Multitaper, gleiche Gewichte)" in text
    assert f"{ev.de(r2['welch'], 2)} dB (Welch) gegen {ev.de(r2['equal'], 2)} dB (gleich) und {ev.de(r2['adaptive'], 2)} dB (adaptiv)" in text


def test_readme_valley_resolution_and_line_test_numbers():
    text = _readme()
    valley = {r["label"]: r["bands"][2] for r in get("valley")}
    assert f"Multitaper mit NW = 4 und gleichen Gewichten um {ev.de(valley['Multitaper NW = 4, gleich'], 1)} dB" in text
    assert f"adaptiv {ev.de(abs(valley['Multitaper NW = 4, adaptiv']), 1)} dB" in text
    assert f"NW = 2: {ev.de(valley['Multitaper NW = 2, gleich'], 1)}; NW = 8: {ev.de(valley['Multitaper NW = 8, gleich'], 1)}" in text
    assert f"Rechteck-Periodogramm {ev.de(valley['Periodogramm, Rechteck'], 1)} dB" in text
    t = get("line_test")
    fa = {r["background"]: r for r in t["false_alarm"]}
    assert f"F-Test {ev.de(100 * fa['Resonanz']['f'], 1)} %, Fisher-g {ev.de(100 * fa['Resonanz']['g'], 0)} %" in text
    det = {d["snr"]: d for d in t["detect"]}
    assert " / ".join(ev.de(100 * det[s]["f"], 0) for s in (-18, -14, -12, -10, -8)) in text
    assert " / ".join(ev.de(100 * det[s]["g"], 0) for s in (-18, -14, -12, -10, -8)) in text
