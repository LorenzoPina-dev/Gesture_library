"""
tests/test_recognition_and_events.py
======================================
Verifica Livello 1 (rule-based), Livello 2 (k-NN open-set), Livello 3 (FSM)
e l'Event Bus (multi-callback, cooldown).
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.normalization.geometric import normalize_landmarks
from gesture_engine.recognition.rule_based import RuleBasedRecognizer
from gesture_engine.recognition.knn_classifier import EmbeddingKNNClassifier, UNKNOWN_LABEL
from gesture_engine.recognition.state_machine import GestureSequenceFSM, SequenceDefinition, SwipeDetector
from gesture_engine.events.event_bus import EventBus


def _fake_hand(fingers_extended: int):
    """Costruisce landmark con un numero controllato di dita "estese"."""
    pts = np.zeros((21, 3))
    pts[0] = [0.5, 0.8, 0.0]
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    finger_idxs = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                   "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    angles = {"thumb": -0.9, "index": -0.3, "middle": 0.0, "ring": 0.3, "pinky": 0.6}

    for i, finger in enumerate(fingers):
        idxs = finger_idxs[finger]
        angle = angles[finger]
        extended = i < fingers_extended
        if extended:
            # dita distese: distanza dal polso cresce monotonamente verso la punta
            dists = [0.05, 0.10, 0.15, 0.20]
        else:
            # dita piegate: la punta ripiega verso il polso, piu' vicina persino
            # dell'MCP (curl realistico verso il palmo)
            dists = [0.05, 0.10, 0.07, 0.03]
        for j, idx in enumerate(idxs):
            pts[idx] = [0.5 + np.sin(angle) * dists[j], 0.8 - np.cos(angle) * dists[j], 0.0]
    return pts


@pytest.fixture
def cfg():
    return EngineConfig()


def test_rule_based_fist(cfg):
    rb = RuleBasedRecognizer(cfg.rule_based)
    hand = normalize_landmarks(_fake_hand(fingers_extended=0))
    result = rb.recognize(hand)
    assert result.label == "fist"
    assert result.finger_count <= 1


def test_rule_based_open_palm(cfg):
    rb = RuleBasedRecognizer(cfg.rule_based)
    hand = normalize_landmarks(_fake_hand(fingers_extended=5))
    result = rb.recognize(hand)
    assert result.finger_count == 5
    assert result.label == "open_palm"


def test_knn_open_set_rejection(cfg):
    knn = EmbeddingKNNClassifier(cfg.knn)
    rng = np.random.RandomState(42)
    known_cluster = rng.normal(loc=1.0, scale=0.02, size=(6, 128)).astype(np.float32)
    knn.enroll("gesture_a", known_cluster)

    # query vicina al cluster enrollato -> deve essere riconosciuta
    near_query = known_cluster.mean(axis=0) + rng.normal(scale=0.01, size=128).astype(np.float32)
    result_near = knn.predict(near_query)
    assert result_near.label == "gesture_a"

    # query lontana (ortogonale, cluster completamente diverso) -> UNKNOWN
    far_query = -known_cluster.mean(axis=0)
    result_far = knn.predict(far_query)
    assert result_far.label == UNKNOWN_LABEL


def test_knn_enroll_and_remove(cfg):
    knn = EmbeddingKNNClassifier(cfg.knn)
    knn.enroll("a", np.random.randn(5, 128).astype(np.float32))
    knn.enroll("b", np.random.randn(5, 128).astype(np.float32))
    assert knn.known_classes() == {"a": 5, "b": 5}

    removed = knn.remove_class("a")
    assert removed == 5
    assert "a" not in knn.known_classes()


def test_event_bus_multi_callback_and_cooldown(cfg):
    bus = EventBus(cfg.event_bus)
    calls = []
    bus.subscribe("evt", lambda e: calls.append(("cb1", e.payload)))
    bus.subscribe("evt", lambda e: calls.append(("cb2", e.payload)))

    emitted_1 = bus.emit("evt", {"v": 1}, now=0.0)
    emitted_2 = bus.emit("evt", {"v": 2}, now=0.05)  # dentro il cooldown di default (0.4s)
    emitted_3 = bus.emit("evt", {"v": 3}, now=0.5)   # oltre il cooldown

    assert emitted_1 is True
    assert emitted_2 is False
    assert emitted_3 is True
    assert len(calls) == 4  # 2 callback x 2 emissioni effettive


def test_sequence_fsm_completes_in_order(cfg):
    completed = []
    fsm = GestureSequenceFSM(cfg.state_machine, lambda name: completed.append(name))
    fsm.register_sequence(SequenceDefinition("boom", ("fist", "open_palm")))

    fsm.update("fist", now=0.0)
    fsm.update("open_palm", now=0.5)

    assert completed == ["boom"]


def test_sequence_fsm_times_out(cfg):
    completed = []
    fsm = GestureSequenceFSM(cfg.state_machine, lambda name: completed.append(name))
    fsm.register_sequence(SequenceDefinition("boom", ("fist", "open_palm"), timeout_s=1.0))

    fsm.update("fist", now=0.0)
    fsm.update("open_palm", now=2.0)  # oltre il timeout di 1.0s

    assert completed == []


def test_swipe_detector_emits_on_fast_motion(cfg):
    events = []
    detector = SwipeDetector(cfg.state_machine, lambda e: events.append(e))

    detector.update(np.array([0.1, 0.5]), now=0.0)
    detector.update(np.array([0.9, 0.5]), now=0.1)  # velocita' ~8 unita'/s > soglia (4.0)

    assert len(events) == 1
    assert events[0].direction == "right"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
