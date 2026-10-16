"""Festwerte der Experimente (feste Ströme ab Seed 400000): fangen Vorzeichen, Exponenten, Normierungen und Lagen, die Toleranz-Tests übersehen.

Die Werte stammen aus dem ersten Lauf des geprüften Codes (Handrechnung und scipy-Orakel stehen in test_taper.py, test_spectrum.py und test_oracle_multitaper.py)."""

import pytest

import mtp_constants as C
import mtp_evaluation as ev
from sweeps_cache import get

TOL = dict(abs=2e-3)


def test_concentration_fixed_values():
    res = get("concentration")
    lost = {r["nw"]: r["lost"] for r in res["tapers"]}
    assert [len(lost[nw]) for nw in (2, 3, 4, 6)] == [6, 8, 10, 14]
    assert lost[2] == pytest.approx([5.724e-05, 2.438e-03, 4.061e-02, 2.782e-01, 7.253e-01, 9.570e-01], rel=1e-3, abs=1e-9)
    assert lost[3] == pytest.approx([1.348e-07, 9.244e-06, 2.850e-04, 5.085e-03, 5.386e-02, 2.922e-01, 7.102e-01, 9.448e-01], rel=1e-3, abs=1e-12)
    assert lost[4] == pytest.approx([2.944e-10, 2.767e-08, 1.210e-06, 3.244e-05, 5.898e-04, 7.495e-03, 6.334e-02, 3.012e-01, 7.006e-01, 9.358e-01], rel=1e-3, abs=1e-12)
    assert lost[6][3:] == pytest.approx([5.571e-10, 1.739e-08, 4.094e-07, 7.517e-06, 1.096e-04, 1.274e-03, 1.159e-02, 7.667e-02, 3.127e-01, 6.885e-01, 9.229e-01], rel=2e-3, abs=1e-12)
    classic = {c["window"]: c for c in res["classic"]}
    assert res["band_nw"] == 4
    assert classic["Rechteck"]["outside"] == pytest.approx(2.4862e-02, rel=1e-3) and classic["Hann"]["outside"] == pytest.approx(1.364e-05, rel=1e-2)
    assert classic["Blackman"]["outside"] == pytest.approx(1.132e-06, rel=2e-2)
    assert [classic[w]["enbw"] for w in ("Rechteck", "Hann", "Blackman")] == pytest.approx([1.0, 1.5, 1.7268], abs=1e-4)


def test_equal_bandwidth_fixed_values():
    rows = get("equal_bandwidth")
    assert [r["nw"] for r in rows] == [2, 3, 4, 6] and [r["k"] for r in rows] == [3, 5, 7, 11] and [r["bandwidth_bins"] for r in rows] == [4, 6, 8, 12]
    assert [r["k_eff_mt"] for r in rows] == pytest.approx([3.0124, 5.0332, 7.0624, 11.1224], **TOL)
    assert [r["std_mt"] for r in rows] == pytest.approx([2.7363, 2.0438, 1.6986, 1.335], **TOL)
    assert [r["seg"] for r in rows] == [384, 256, 192, 128] and [r["k_welch"] for r in rows] == [4, 7, 9, 15]
    assert [r["k_eff_welch"] for r in rows] == pytest.approx([3.8622, 6.7273, 8.625, 14.41], **TOL)
    assert [r["k_eff_welch_formula"] for r in rows] == pytest.approx([3.84, 6.6818, 8.5765, 14.2606], **TOL)
    assert [r["std_welch"] for r in rows] == pytest.approx([2.3536, 1.734, 1.517, 1.1626], **TOL)
    assert [r["welch_enbw_bins"] for r in rows] == pytest.approx([4.0, 6.0, 8.0, 12.0], abs=1e-9)


def test_best_vs_best_fixed_values():
    rows = get("best_vs_best")
    expect = [(128, "Resonanz", 31.566, 2.8248, 32, 2.5672, 3, 2.7428, 3), (128, "Steil", 56.699, 3.4338, 64, 5.385, 2, 4.0971, 3),
              (128, "Zwei scharfe Resonanzen", 45.555, 5.7545, 64, 5.575, 2, 6.0502, 2), (256, "Resonanz", 32.998, 1.9418, 64, 1.9319, 4, 2.0211, 4),
              (256, "Steil", 56.699, 2.5609, 64, 3.9492, 3, 2.7446, 4), (256, "Zwei scharfe Resonanzen", 46.888, 3.9433, 128, 3.9408, 2, 4.3876, 3),
              (1024, "Resonanz", 33.093, 1.229, 64, 1.1362, 12, 1.1533, 12), (1024, "Steil", 56.699, 1.4725, 128, 2.3228, 6, 1.5458, 8),
              (1024, "Zwei scharfe Resonanzen", 55.574, 2.2933, 256, 2.2795, 4, 2.4529, 4)]
    assert len(rows) == len(expect)
    for r, (n, bg, span, welch, seg, eq, eq_nw, ad, ad_nw) in zip(rows, expect):
        assert (r["n"], r["background"], r["welch_seg"], r["equal_nw"], r["adaptive_nw"]) == (n, bg, seg, eq_nw, ad_nw)
        assert (r["span_db"], r["welch"], r["equal"], r["adaptive"]) == pytest.approx((span, welch, eq, ad), abs=2e-3)


def test_valley_fixed_values():
    rows = get("valley")
    expect = {"Periodogramm, Rechteck": [0.5953, 4.8348, 9.0522], "Periodogramm, Hann": [0.0917, -0.098, 0.0127], "Welch, Hann 50 %, Länge 256": [0.3541, -0.0254, 0.0105],
              "Multitaper NW = 4, gleich": [0.4006, 1.2187, 3.2443], "Multitaper NW = 4, adaptiv": [0.3027, -0.093, -0.0157],
              "Multitaper NW = 2, gleich": [0.1846, 1.0122, 2.8431], "Multitaper NW = 8, gleich": [1.011, 1.4184, 3.6081]}
    assert [r["label"] for r in rows] == list(expect)
    for r in rows:
        assert r["bands"] == pytest.approx(expect[r["label"]], **TOL)


def test_resolution_fixed_values():
    rows = get("resolution")
    expect = {"Multitaper NW = 2": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], "Multitaper NW = 3": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "Multitaper NW = 4": [0.0, 0.0, 0.0, 0.0, 0.9667, 1.0],
              "Multitaper NW = 6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.78], "Welch, Hann 50 %, Länge 512": [0.0, 0.0, 0.46, 1.0, 1.0, 1.0],
              "Welch, Hann 50 %, Länge 256": [0.0, 0.0, 0.0, 0.0, 0.4533, 1.0], "Welch, Hann 50 %, Länge 128": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              "Periodogramm, Hann": [0.4333, 1.0, 1.0, 1.0, 1.0, 1.0]}
    assert [r["label"] for r in rows] == list(expect)
    for r in rows:
        assert r["deltas"] == list(C.EXP_RES_DELTAS)
        assert r["shares"] == pytest.approx(expect[r["label"]], abs=1e-3)


def test_line_test_fixed_values():
    t = get("line_test")
    assert (t["k"], t["nw"], t["runs"], t["alpha"]) == (7, 4, 400, 0.01)
    assert t["f_thr"] == pytest.approx(45.7064, abs=1e-3) and t["g_thr"] == pytest.approx(0.02103, abs=1e-4)
    fa = {r["background"]: (r["f"], r["g"]) for r in t["false_alarm"]}
    assert fa["weiß"] == pytest.approx((0.0075, 0.0075), abs=1e-4) and fa["Resonanz"] == pytest.approx((0.005, 1.0), abs=1e-4)
    assert fa["Steil"] == pytest.approx((0.0075, 1.0), abs=1e-4)
    expect = {-18: (0.0225, 0.0925, 0.12), -16: (0.0525, 0.3025, 0.315), -14: (0.13, 0.66, 0.745), -12: (0.385, 0.9675, 0.9825), -10: (0.7025, 1.0, 1.0), -8: (0.9425, 1.0, 1.0)}
    assert [d["snr"] for d in t["detect"]] == list(expect)
    for d in t["detect"]:
        assert (d["f"], d["g"], d["welch"]) == pytest.approx(expect[d["snr"]], abs=1e-4)
