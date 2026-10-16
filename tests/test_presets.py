"""Presets: Vollständigkeit, gültige Werte, Standardwerte; Zahl der Taper höchstens 2 NW."""

import mtp_constants as C
import mtp_evaluation as ev
import mtp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["nw"] in C.NW_CHOICES and 1 <= p["k"] <= P.k_max(p["nw"]), name
        assert p["weights"] in C.WEIGHTS and p["background"] in C.BACKGROUNDS and p["lines"] in C.LINES
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.GAP_MIN <= p["gap"] <= C.GAP_MAX and C.REL_MIN <= p["rel"] <= C.REL_MAX
        assert C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX
        assert 2 * p["nw"] < p["n"] / 2


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.nw in C.NW_CHOICES and s.k in ev.k_options(s.nw) and s.weights in C.WEIGHTS and s.background in C.BACKGROUNDS
    assert s.k == 2 * s.nw - 1


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_are_short_enough_for_the_buttons():
    assert max(len(n) for n in C.PRESETS) <= 36


def test_taper_count_limits():
    assert P.k_max(2) == 4 and P.k_max(12) == 24 and P.k_max(2.5) == 5


def test_second_line_never_leaves_the_band():
    for line in (C.LINE_MIN, C.LINE_MAX):
        for n in C.N_CHOICES:
            for gap in (C.GAP_MIN, 8.0, C.GAP_MAX):
                f2 = ev.Settings(n=n, line=line, gap=gap).f2_hz
                assert 2.0 * C.FS / n <= f2 <= C.FS / 2 - 2.0 * C.FS / n + 1e-9


def test_second_line_sits_exactly_at_the_limit_above_the_first_and_goes_below_just_after_it():
    n = 128                                   # Grenze: 500 - 2 * 1000 / 128 = 484,375 Hz; ein Bin sind 7,8125 Hz
    s_at = ev.Settings(n=n, line=0.43, gap=(484.375 - 430.0) / (C.FS / n))
    assert s_at.f2_hz == 484.375
    s_over = ev.Settings(n=n, line=0.43, gap=(484.375 - 430.0) / (C.FS / n) + 0.5)
    assert s_over.f2_hz == 430.0 - (484.375 - 430.0 + 0.5 * C.FS / n)
