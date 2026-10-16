"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import mtp_evaluation as ev


def make_analysis(**kw):
    z = np.zeros(1)
    base = dict(settings=ev.Settings(), k=7, h=z, lam=z, f_hz=z, x=z, est=z, welch=z, welch_k=9, welch_k_eff=8.6, expected=None, true_density=z, f_hz_f=z, f_stat=z, f_thr=45.7,
                found=[], fisher_g=0.03, fisher_thr=0.021, fisher_alarm=True, window_f=z, window_mt=z, window_hann=z, bandwidth_hz=7.8125, welch_enbw_hz=7.8, k_eff=7.0,
                std_db=1.7, welch_std_db=1.5, full_std_db=5.57, resolved=None, welch_resolved=None, ops=35840, ops_welch=7000, ops_full=5120)
    base.update(kw)
    return ev.Analysis(**base)


def test_main_verdict_without_lines():
    kind, text = ev.verdict_main(make_analysis(settings=ev.Settings(lines="keine")))
    assert kind == "info"
    assert text == ("7 Taper mit NW = 4 ergeben 7 unabhängige Mittel auf der ganzen Aufzeichnung (Rauschbandbreite 7,8 Hz); Welch bei gleicher Bandbreite hat etwa 8,6.")


def test_main_verdict_with_one_line_found_missed_and_without_f_test():
    s = ev.Settings(lines="eine")
    kind, text = ev.verdict_main(make_analysis(settings=s, found=[349.6]))
    assert kind == "success" and "Der F-Test findet die Linie bei 349,6 Hz (Falschalarm 1 % je Aufzeichnung)." in text
    kind, text = ev.verdict_main(make_analysis(settings=s, found=[]))
    assert kind == "warning" and "findet die Linie **nicht**" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, f_stat=None, f_thr=None, k=1))
    assert kind == "info" and "Mit einem Taper ist kein F-Test möglich." in text


def test_main_verdict_two_lines():
    s = ev.Settings(lines="zwei")
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=True))
    assert kind == "success" and "**zwei Gipfel**" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=False))
    assert kind == "warning" and "**verschmelzen**" in text


def test_main_verdict_uses_the_fractional_nw_without_trailing_zeros():
    _, text = ev.verdict_main(make_analysis(settings=ev.Settings(nw=2.5, lines="keine")))
    assert "mit NW = 2,5" not in text and "NW = 2.5" in text            # :g liefert den Punkt; die App erlaubt nur ganze NW


def test_concentration_verdict():
    res = {"band_nw": 4, "tapers": [{"nw": 2, "lost": [0.1] * 6}, {"nw": 4, "lost": [2.9e-10, 2.8e-8, 1.2e-6, 3.2e-5, 5.9e-4, 7.5e-3, 6.3e-2, 0.30, 0.70, 0.94]}],
           "classic": [{"window": "Rechteck", "outside": 2.5e-2, "enbw": 1.0}, {"window": "Hann", "outside": 1.4e-5, "enbw": 1.5}]}
    kind, text = ev.verdict_concentration(res)
    assert kind == "info"
    assert "beim ersten Taper 2,9e-10 der Energie, beim 7. (letzter gut konzentrierter) 6,3e-02, danach springt es auf 30 %." in text
    assert "Beim Rechteck liegen 2,5 %, bei Hann 1,4e-05 außerhalb." in text


def test_equal_bandwidth_verdict_reads_nw_six():
    rows = [{"nw": 4, "bandwidth_bins": 8, "k_eff_mt": 7.1, "k_eff_welch": 8.6}, {"nw": 6, "bandwidth_bins": 12, "k_eff_mt": 11.1, "k_eff_welch": 14.4}]
    kind, text = ev.verdict_equal_bandwidth(rows)
    assert kind == "warning" and "von 12 Bins liefert das Multitaper 11,1 unabhängige Mittel, Welch mit Hann und 50 % Überlappung 14,4" in text


def test_best_verdict_picks_the_longest_record():
    rows = [{"n": 128, "background": "Resonanz", "span_db": 31.6, "welch": 9.0, "equal": 9.0, "adaptive": 9.0},
            {"n": 1024, "background": "Resonanz", "span_db": 33.1, "welch": 1.23, "equal": 1.14, "adaptive": 1.15},
            {"n": 1024, "background": "Steil", "span_db": 56.7, "welch": 1.47, "equal": 2.32, "adaptive": 1.55}]
    kind, text = ev.verdict_best(rows)
    assert kind == "warning"
    assert "(Resonanz, n = 1024) liegt das beste Multitaper bei 1,14 dB gegen 1,23 dB für Welch" in text
    assert "(57 dB Spannweite) bei 2,32 dB (gleiche Gewichte) bzw. 1,55 dB (adaptiv) gegen 1,47 dB für Welch" in text


def test_valley_verdict_uses_rows_three_and_four():
    rows = [{"bands": [0, 0, 9.0]}, {"bands": [0, 0, 0.0]}, {"bands": [0, 0, 0.0]}, {"bands": [0.4, 1.2, 3.24]}, {"bands": [0.3, -0.1, -0.02]}]
    kind, text = ev.verdict_valley(rows)
    assert kind == "success" and "mit gleichen Gewichten um 3,2 dB" in text and "mit adaptiven Gewichten um -0,0 dB" in text


def test_resolution_verdict_for_nw_four():
    rows = [{"label": "Multitaper NW = 2", "deltas": [2.0, 4.0, 8.0], "shares": [0.0, 1.0, 1.0]},
            {"label": "Multitaper NW = 4", "deltas": [2.0, 4.0, 8.0], "shares": [0.0, 0.5, 0.96]}]
    kind, text = ev.verdict_resolution(rows)
    assert kind == "info" and "ab 8 Bins Abstand in mindestens 95 % der Läufe getrennt (Rauschbandbreite 8 Bins)" in text
    rows[1]["shares"] = [0.0, 0.0, 0.5]
    kind, text = ev.verdict_resolution(rows)
    assert "nicht sicher getrennt" in text


def test_line_test_verdict():
    res = {"alpha": 0.01, "false_alarm": [{"background": "Resonanz", "f": 0.005, "g": 1.0}, {"background": "weiß", "f": 0.0075, "g": 0.0075}],
           "detect": [{"snr": -12, "f": 0.4, "g": 0.97, "welch": 0.98}, {"snr": -10, "f": 0.7025, "g": 1.0, "welch": 1.0}]}
    kind, text = ev.verdict_line_test(res)
    assert kind == "success"
    assert "bei Resonanz-Hintergrund: F-Test 0,5 %, Fisher-g 100 % (nominal 1 %)" in text and "findet der F-Test die Linie in 70 % der Läufe, Fisher-g (nur weißes Rauschen) in 100 %" in text


def test_scientific_notation_uses_a_decimal_comma():
    assert ev.sci(2.944e-10) == "2,9e-10" and ev.sci(0.0633) == "6,3e-02"


def test_k_options_and_defaults():
    assert ev.k_options(2) == [1, 2, 3, 4] and ev.k_options(12)[-1] == 24
    s = ev.Settings(nw=2, k=7)
    assert s.k_used == 4 and ev.Settings(nw=4, k=7).k_used == 7 and ev.Settings(nw=3, k=0).k_used == 1
    assert ev.Settings(n=1024, nw=4).welch_seg == 192 and ev.Settings(n=128, nw=12).welch_seg == 8
