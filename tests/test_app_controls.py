"""AppTest: Auswahlfelder, Permalink, Taper-Zahl gegen NW, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mtp_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs(background):
    at = _run()
    at.selectbox(key="background_select").set_value(background).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("weights", C.WEIGHTS)
def test_both_weightings_run(weights):
    at = _run()
    at.radio(key="weights_select").set_value(weights).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_line_controls_appear_only_when_there_are_lines():
    at = _run({"ln": "keine"})
    assert not any(label.startswith(("Linie 1", "Abstand", "Linie 2")) for label in _labels(at))
    at.radio(key="lines_select").set_value("eine").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    labels = _labels(at)
    assert "Linie 1 (Bruchteil der Abtastrate)" in labels and "Linie 1 gegen Rauschen (dB je Abtastwert)" in labels and "Abstand der Linie 2 (Bins von f_s/n)" not in labels
    at.radio(key="lines_select").set_value("zwei").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Abstand der Linie 2 (Bins von f_s/n)" in _labels(at) and "Linie 2 gegen Linie 1 (dB)" in _labels(at)


def test_line_values_survive_switching_the_lines_away_and_back():
    at = _run({"ln": "zwei", "f": "0.3", "d": "7.5", "rel": "-12", "snr": "-5"})
    by = {x.label: x for x in at.slider}
    assert by["Linie 1 (Bruchteil der Abtastrate)"].value == 0.3 and by["Abstand der Linie 2 (Bins von f_s/n)"].value == 7.5
    at.radio(key="lines_select").set_value("keine").run(timeout=TIMEOUT)
    assert not at.exception and not any(label.startswith("Linie") for label in _labels(at))
    at.radio(key="lines_select").set_value("zwei").run(timeout=TIMEOUT)
    by = {x.label: x for x in at.slider}
    assert by["Linie 2 gegen Linie 1 (dB)"].value == -12 and by["Linie 1 gegen Rauschen (dB je Abtastwert)"].value == -5


def test_lowering_nw_pulls_the_taper_count_down_and_the_options_follow():
    at = _run({"nw": "12", "k": "20"})
    assert at.select_slider(key="k_select").value == 20
    at.select_slider(key="nw_select").set_value(2).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="k_select").value == 4
    assert [int(o) for o in at.select_slider(key="k_select").options] == [1, 2, 3, 4]


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "512", "nw": "6", "k": "9", "wt": "adaptiv", "bg": "Steil", "ln": "zwei", "f": "9", "d": "0", "rel": "-999", "snr": "999", "seed": "42"})
    assert at.select_slider(key="n_select").value == 512 and at.select_slider(key="nw_select").value == 6 and at.select_slider(key="k_select").value == 9
    assert at.radio(key="weights_select").value == "adaptiv" and at.selectbox(key="background_select").value == "Steil" and at.radio(key="lines_select").value == "zwei"
    by = {x.label: x for x in at.slider}
    assert by["Linie 1 (Bruchteil der Abtastrate)"].value == C.LINE_MAX and by["Abstand der Linie 2 (Bins von f_s/n)"].value == C.GAP_MIN
    assert by["Linie 2 gegen Linie 1 (dB)"].value == C.REL_MIN and by["Linie 1 gegen Rauschen (dB je Abtastwert)"].value == C.SNR_MAX
    at2 = _run({"n": "3000", "nw": "7", "wt": "unsinn", "bg": "unsinn", "ln": "drei"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.select_slider(key="nw_select").value == C.DEFAULT_NW
    assert at2.radio(key="weights_select").value == C.DEFAULT_WEIGHTS and at2.radio(key="lines_select").value == C.DEFAULT_LINES


def test_permalink_with_too_many_tapers_is_clamped():
    at = _run({"nw": "2", "k": "24"})
    assert at.select_slider(key="k_select").value == 4


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Kurze Aufzeichnung").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.select_slider(key="n_select").value == 128 and at.select_slider(key="nw_select").value == 3 and at.select_slider(key="k_select").value == 5


def test_one_taper_shows_the_f_test_note():
    at = _run({"nw": "2", "k": "1"})
    assert any("kein F-Test" in i.value for i in at.info)


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("label", ["Abstand der Linie 2 (Bins von f_s/n)", "Linie 2 gegen Linie 1 (dB)"])
@pytest.mark.parametrize("edge", ["min", "max"])
def test_second_line_sliders_at_their_limits(label, edge):
    at = _run({"ln": "zwei"})
    slider = [x for x in at.slider if x.label == label][0]
    value = slider.min if edge == "min" else slider.max
    slider.set_value(int(value) if isinstance(slider.value, int) else value).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
