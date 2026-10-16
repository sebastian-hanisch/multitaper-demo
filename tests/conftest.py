"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_tapers():
    """n = 3, NW = 1 (W = 1/3) von Hand: Tridiagonalmatrix mit Diagonale (-0,5; 0; -0,5) und Nebendiagonale (1; 1) hat die Eigenwerte 1,1861 / -0,5 / -1,6861.

    Symmetrischer Eigenvektor (1; 1,6861; 1) / 2,2007 = (0,4544; 0,7662; 0,4544), antisymmetrischer (0,7071; 0; -0,7071), dritter (0,5418; -0,6426; 0,5418).
    """
    return np.array([[0.4544, 0.7662, 0.4544], [0.7071, 0.0, -0.7071], [0.5418, -0.6426, 0.5418]])
