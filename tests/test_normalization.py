"""
tests/test_normalization.py
============================
Verifica le proprieta' di invarianza della normalizzazione geometrica:
traslazione, scala e rotazione planare non devono alterare il risultato,
mentre gesture geometricamente diverse devono produrre vettori diversi.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.normalization.geometric import (
    normalize_landmarks,
    flatten,
    finger_extension_ratios,
    pinch_distance,
)
from gesture_engine.normalization.filters import EMAFilter, OneEuroFilter


def _fake_open_hand():
    """Costruisce 21 landmark plausibili di una mano aperta (a ventaglio)."""
    pts = np.zeros((21, 3))
    pts[0] = [0.5, 0.8, 0.0]  # wrist
    finger_bases = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                     "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    angles = {"thumb": -0.9, "index": -0.3, "middle": 0.0, "ring": 0.3, "pinky": 0.6}
    for finger, idxs in finger_bases.items():
        angle = angles[finger]
        for j, idx in enumerate(idxs):
            dist = 0.05 * (j + 1)
            pts[idx] = [0.5 + np.sin(angle) * dist, 0.8 - np.cos(angle) * dist, 0.0]
    return pts


def test_translation_invariance():
    raw = _fake_open_hand()
    shifted = raw + np.array([0.2, -0.1, 0.0])

    norm_a = flatten(normalize_landmarks(raw))
    norm_b = flatten(normalize_landmarks(shifted))

    assert np.allclose(norm_a, norm_b, atol=1e-6)


def test_scale_invariance():
    raw = _fake_open_hand()
    scaled = raw.copy()
    scaled[:, :2] = (scaled[:, :2] - raw[0, :2]) * 2.0 + raw[0, :2]  # scala 2x attorno al polso

    norm_a = flatten(normalize_landmarks(raw))
    norm_b = flatten(normalize_landmarks(scaled))

    assert np.allclose(norm_a, norm_b, atol=1e-5)


def test_rotation_invariance_planar():
    raw = _fake_open_hand()
    theta = np.radians(35)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = raw.copy()
    origin = raw[0, :2]
    rotated[:, :2] = (raw[:, :2] - origin) @ rot.T + origin

    norm_a = flatten(normalize_landmarks(raw))
    norm_b = flatten(normalize_landmarks(rotated))

    assert np.allclose(norm_a, norm_b, atol=1e-4)


def test_finger_extension_ratios_open_hand_higher_than_fist():
    open_hand = normalize_landmarks(_fake_open_hand())

    fist = _fake_open_hand()
    # "chiudi" le dita avvicinando le punte al polso
    for tip_idx in (4, 8, 12, 16, 20):
        fist[tip_idx] = fist[0] + (fist[tip_idx] - fist[0]) * 0.15
    fist_norm = normalize_landmarks(fist)

    ratios_open = finger_extension_ratios(open_hand)
    ratios_fist = finger_extension_ratios(fist_norm)

    for finger in ratios_open:
        assert ratios_open[finger] > ratios_fist[finger]


def test_ema_filter_smooths_noise():
    f = EMAFilter(alpha=0.3)
    base = _fake_open_hand()
    out1 = f.filter(base)
    noisy = base + np.random.RandomState(0).normal(scale=0.05, size=base.shape)
    out2 = f.filter(noisy)
    # l'output filtrato deve essere piu' vicino al valore base che al rumore puro
    assert np.linalg.norm(out2 - base) < np.linalg.norm(noisy - base)


def test_one_euro_filter_runs_and_converges():
    f = OneEuroFilter()
    base = _fake_open_hand()
    t = 0.0
    out = None
    for _ in range(20):
        out = f.filter(base, timestamp=t)
        t += 1 / 30.0
    assert np.allclose(out, base, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
