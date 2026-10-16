"""Generator der Linie (wortgleich in allen Repos): Strom, Pole, AR-Filter, Autokovarianz, Dichte, Erwartungswert des Periodogramms."""

import math

import numpy as np
import pytest

import signal_generator as gen

MASK = (1 << 64) - 1
_trapz = getattr(np, "trapezoid", None) or np.trapz


def splitmix_reference(seed, count):
    """Unabhängige Python-Fassung des SplitMix64-Stroms (Steele, Lea, Flood 2014)."""
    state, out = seed & MASK, []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**63 + 5])
def test_splitmix_stream_matches_the_reference(seed):
    assert gen.SplitMix64(seed).u64(25).tolist() == splitmix_reference(seed, 25)


def test_the_stream_does_not_depend_on_how_it_is_drawn():
    a, b = gen.SplitMix64(7), gen.SplitMix64(7)
    assert np.concatenate([a.uniform(3), a.uniform(4)]).tolist() == b.uniform(7).tolist()


def test_normal_has_mean_zero_and_unit_variance():
    z = gen.SplitMix64(11).normal(60000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_poles_to_coefficients_by_hand():
    assert gen.ar_from_poles(()).tolist() == []
    assert gen.ar_from_poles(((0.0, 0.9),)) == pytest.approx([0.9])
    assert gen.ar_from_poles(((0.5, 0.5),)) == pytest.approx([-0.5])
    # Paar r exp(+-i 2 pi f): a1 = 2 r cos(2 pi f), a2 = -r^2
    a = gen.ar_from_poles(((0.25, 0.8),))
    assert a == pytest.approx([0.0, -0.64], abs=1e-12)
    a = gen.ar_from_poles(((1.0 / 12.0, 0.9),))
    assert a == pytest.approx([2 * 0.9 * math.cos(math.pi / 6), -0.81])


def test_poles_roundtrip_through_the_polynomial_roots():
    poles = ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9))
    a = gen.ar_from_poles(poles)
    roots = np.roots(np.r_[1.0, -a])
    assert sorted(np.round(np.abs(roots), 6).tolist()) == pytest.approx(sorted([0.99, 0.99, 0.9, 0.9]), abs=1e-5)


def test_invalid_poles_are_rejected():
    for bad in (((0.6, 0.5),), ((0.1, 1.0),), ((-0.1, 0.5),)):
        with pytest.raises(ValueError):
            gen.ar_from_poles(bad)


def test_a_non_stationary_process_is_rejected():
    with pytest.raises(ValueError):
        gen.ar_length([1.1])


def test_impulse_response_of_ar1_is_geometric():
    h = gen.ar_impulse_response([0.9], 6)
    assert h == pytest.approx([1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049])


def test_ar_autocov_of_ar1_by_hand():
    # AR(1), a = 0,5, sigma^2 = 1: r(k) = a^k / (1 - a^2)
    r = gen.ar_autocov([0.5], 1.0, 5)
    assert r == pytest.approx([(0.5 ** k) / 0.75 for k in range(5)])


def test_unit_sigma_gives_unit_variance():
    for poles in (((0.0, 0.9),), ((0.2, 0.97),), ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9))):
        a = gen.ar_from_poles(poles)
        assert gen.ar_autocov(a, gen.ar_unit_sigma(a) ** 2, 1)[0] == pytest.approx(1.0)
    assert gen.ar_unit_sigma(np.zeros(0)) == 1.0


def test_make_signal_is_reproducible_and_the_clean_part_is_the_lines():
    kw = dict(n=300, fs=1000.0, lines=[(100.0, 1.5, 0.7)], poles=((0.2, 0.9),), ar_sigma=1.0, white_sigma=0.5)
    a = gen.make_signal(rng=gen.SplitMix64(4), **kw)
    b = gen.make_signal(rng=gen.SplitMix64(4), **kw)
    assert np.array_equal(a.x, b.x)
    t = np.arange(300) / 1000.0
    assert np.allclose(a.clean, 1.5 * np.cos(2 * np.pi * 100.0 * t + 0.7))
    assert np.allclose(a.x, a.clean + a.noise)


def test_noise_batch_has_unit_variance_and_independent_rows():
    poles = ((0.2, 0.97),)
    sg = gen.ar_unit_sigma(gen.ar_from_poles(poles))
    x = gen.make_noise_batch(60, 1500, poles, sg, 0.0, gen.SplitMix64(3))
    assert x.shape == (60, 1500)
    assert x.var() == pytest.approx(1.0, rel=0.08)
    cors = [abs(np.corrcoef(x[i], x[i + 1])[0, 1]) for i in range(0, 40, 2)]
    assert np.mean(cors) < 0.2 and not np.allclose(x[0], x[1])


def test_line_power_is_half_the_squared_amplitude():
    s = gen.line_signal(10000, 1000.0, [(125.0, 2.0, 0.3)])
    assert (s ** 2).mean() == pytest.approx(2.0, rel=1e-3)


def test_true_psd_integrates_to_the_variance():
    poles = ((0.2, 0.9),)
    a = gen.ar_from_poles(poles)
    sg = 1.7
    f = np.linspace(-500.0, 500.0, 200001)
    area = _trapz(gen.true_psd(f, 1000.0, poles, sg, 0.0), f)
    assert area == pytest.approx(gen.ar_autocov(a, sg ** 2, 1)[0], rel=1e-4)
    # weißes Rauschen: Dichte sigma_w^2 / fs
    assert gen.true_psd([0.0, 200.0], 1000.0, (), 0.0, 2.0).tolist() == [0.004, 0.004]


def test_expected_periodogram_matches_the_double_sum_definition():
    n, fs = 12, 4.0
    r = gen.ar_autocov([0.6, -0.2], 1.3, n)
    s = np.cos(2 * np.pi * 0.7 * np.arange(n) / fs + 0.4)
    f = np.array([0.0, 0.3, 1.0, 2.0])
    got = gen.expected_periodogram(f, fs, r, s)
    for fk, g in zip(f, got):
        w = 2 * np.pi * fk / fs
        noise = sum(r[abs(i - j)] * np.cos(w * (i - j)) for i in range(n) for j in range(n))
        det = abs(sum(s[i] * np.exp(-1j * w * i) for i in range(n))) ** 2
        assert g == pytest.approx((noise + det) / (n * fs), rel=1e-10)


def test_ar_length_of_no_poles_and_of_a_zero_pole():
    assert gen.ar_length(np.zeros(0)) == 0
    assert gen.ar_length([0.0]) == 64                      # Polradius 0: kürzeste zulässige Impulsantwort
    assert gen.ar_length([0.5]) == 64 and gen.ar_length([0.9]) > 200 and gen.ar_length([0.99]) > gen.ar_length([0.9])


def test_batch_of_one_run_equals_the_single_signal_noise():
    poles = ((0.2, 0.9),)
    sg = gen.ar_unit_sigma(gen.ar_from_poles(poles))
    batch = gen.make_noise_batch(1, 200, poles, sg, 0.5, gen.SplitMix64(5))[0]
    single = gen.make_signal(200, 1000.0, [], poles, sg, 0.5, gen.SplitMix64(5)).noise
    assert np.array_equal(batch, single)


def test_no_ar_draws_without_poles_or_without_innovation_strength():
    # ohne Pole: keine Innovationen, selbst bei ar_sigma > 0; ohne Stärke: keine Innovationen, selbst mit Polen. Prüfung am Verbrauch des Stroms.
    white_only = gen.SplitMix64(9)
    gen.make_signal(50, 1000.0, [], (), 0.0, 1.0, white_only)
    for poles, sg in (((), 1.0), (((0.2, 0.9),), 0.0)):
        rng = gen.SplitMix64(9)
        sig = gen.make_signal(50, 1000.0, [], poles, sg, 1.0, rng)
        assert rng.state == white_only.state
        assert np.array_equal(sig.noise, gen.make_signal(50, 1000.0, [], (), 0.0, 1.0, gen.SplitMix64(9)).noise)
        rng_b = gen.SplitMix64(9)
        batch = gen.make_noise_batch(1, 50, poles, sg, 1.0, rng_b)
        assert rng_b.state == white_only.state and np.array_equal(batch[0], sig.noise)


def test_noise_autocov_of_pure_white_and_pure_ar():
    assert gen.noise_autocov((), 1.0, 0.0, 4).tolist() == [0.0, 0.0, 0.0, 0.0]        # ohne Pole kein AR-Anteil
    assert gen.noise_autocov((), 0.0, 2.0, 3).tolist() == [4.0, 0.0, 0.0]            # weißes Rauschen: Varianz sigma_w^2
    mixed = gen.noise_autocov(((0.0, 0.5),), 1.0, 2.0, 3)
    assert mixed[0] == pytest.approx(1 / 0.75 + 4.0) and mixed[1] == pytest.approx(0.5 / 0.75)


def test_a_pole_at_radius_zero_is_allowed_and_a_pole_on_the_unit_circle_is_not():
    assert gen.ar_from_poles(((0.2, 0.0),)).tolist() == [0.0, -0.0]
    with pytest.raises(ValueError):
        gen.ar_length([1.0])


def test_ar_length_adds_the_order_to_the_decay_length():
    a = [0.99]
    assert gen.ar_length(a) == math.ceil(math.log(1e-12) / math.log(0.99)) + 1
    assert gen.ar_length([0.5, 0.0]) == 64 and gen.ar_length([0.9, 0.0]) == math.ceil(math.log(1e-12) / math.log(0.9)) + 2


def test_white_noise_is_not_drawn_when_its_strength_is_zero():
    poles = ((0.2, 0.9),)
    a = gen.ar_from_poles(poles)
    expect = gen.SplitMix64(7)
    expect.normal(60 + gen.ar_length(a))                                  # nur die Innovationen
    rng = gen.SplitMix64(7)
    gen.make_signal(60, 1000.0, [], poles, 1.0, 0.0, rng)
    assert rng.state == expect.state
    rng_b = gen.SplitMix64(7)
    gen.make_noise_batch(3, 60, poles, 1.0, 0.0, rng_b)
    expect_b = gen.SplitMix64(7)
    expect_b.normal(3 * (60 + gen.ar_length(a)))
    assert rng_b.state == expect_b.state


def test_ar_autocov_of_white_noise_without_poles():
    assert gen.ar_autocov(np.zeros(0), 2.5, 4).tolist() == [2.5, 0.0, 0.0, 0.0]
    assert gen.ar_autocov([0.5], 1.0, 1).shape == (1,)
