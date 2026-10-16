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
