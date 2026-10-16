"""Slepian-Taper: Mini-Instanz von Hand, Eigenschaften (Orthonormalität, Symmetrie, Vorzeichen), Konzentration."""

import math

import numpy as np
import pytest

import mtp_taper as T


def test_mini_instance_n3_nw1_by_hand(mini_tapers):
    h = T.dpss(3, 1, 3)
    assert h == pytest.approx(mini_tapers, abs=5e-5)
    # Eigenwert des ersten Tapers: lambda^2 + 0,5 lambda - 2 = 0, also (-0,5 + sqrt(8,25)) / 2 = 1,18614
    d, e = T._tridiagonal(3, 1)
    assert d == pytest.approx([-0.5, 0.0, -0.5], abs=1e-12) and e == pytest.approx([1.0, 1.0], abs=1e-12)
    matrix = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    assert matrix @ h[0] == pytest.approx(0.5 * (-0.5 + math.sqrt(8.25)) * h[0], abs=1e-9)


def test_mini_concentration_ratios():
    lam = T.concentration(T.dpss(3, 1, 3), 1)
    assert lam == pytest.approx([0.9936, 0.8045, 0.2019], abs=5e-5)


def test_tapers_are_orthonormal():
    h = T.dpss(128, 3, 6)
    assert h @ h.T == pytest.approx(np.eye(6), abs=1e-10)


def test_symmetry_and_signs():
    h = T.dpss(64, 2, 4)
    for k in range(4):
        mirrored = h[k][::-1]
        assert mirrored == pytest.approx(h[k] if k % 2 == 0 else -h[k], abs=1e-9)
    assert h[0].sum() > 0 and h[2].sum() > 0                 # gerade Taper: positive Summe
    assert abs(h[1].sum()) < 1e-9 and abs(h[3].sum()) < 1e-9
    for k in (1, 3):
        first = h[k][h[k] ** 2 > max(1e-7, 1.0 / 64)][0]
        assert first > 0                                       # ungerade Taper beginnen mit positivem Buckel


def test_concentration_decreases_and_jumps_after_two_nw():
    nw = 3
    lam = T.concentration(T.dpss(256, nw, 2 * nw + 2), nw)
    assert np.all(np.diff(lam) < 0)
    assert lam[2 * nw - 2] > 0.9 and lam[2 * nw + 1] < 0.1                   # Taper 2NW - 1 gut, Taper 2NW + 2 schlecht
    assert 1.0 - lam[0] < 1e-6


def test_concentration_matches_an_independent_matrix_form():
    n, nw = 32, 2
    h = T.dpss(n, nw, 3)
    m = np.arange(n)
    diff = m[:, None] - m[None, :]
    w = nw / n
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(diff == 0, 2.0 * w, np.sin(2.0 * math.pi * w * diff) / (math.pi * diff))
    ref = np.einsum("km,mn,kn->k", h, a, h)
    assert T.concentration(h, nw) == pytest.approx(ref, abs=1e-12)


def test_autocorrelation_matches_a_loop():
    rng = np.random.default_rng(2)
    h = rng.normal(size=(2, 9))
    c = T.autocorrelation(h)
    for k in range(2):
        for tau in range(9):
            assert c[k, tau] == pytest.approx(float((h[k, :9 - tau] * h[k, tau:]).sum()), abs=1e-12)


def test_cache_returns_copies_and_validates_parameters():
    a = T.dpss(32, 2, 2)
    a[0, 0] = 99.0
    assert T.dpss(32, 2, 2)[0, 0] != 99.0
    for bad in ((8, 2, 0), (8, 2, 9), (8, 0, 1), (8, 4, 1), (8, -1, 1)):
        with pytest.raises(ValueError):
            T.dpss(*bad)


def test_taper_count_may_equal_n_for_a_tiny_series():
    h = T.dpss(4, 1, 4)
    assert h @ h.T == pytest.approx(np.eye(4), abs=1e-9)
