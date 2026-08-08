"""
gesture_engine/sign_language/template_store.py
=================================================
Persistenza dei template di segni (sequenze di feature per-frame) enrollati
dal vivo. Analoga a EmbeddingKNNClassifier (recognition/knn_classifier.py)
ma per sequenze di lunghezza variabile invece che singoli embedding statici:
niente retraining, "Few-Shot Dynamic Enrollment" come per le custom gesture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SignTemplate:
    label: str
    sequence: np.ndarray  # (T, FRAME_FEATURE_DIM), float32


class SignTemplateStore:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.templates: List[SignTemplate] = []

    def load(self) -> None:
        self.templates = []
        if not os.path.exists(self.database_path):
            return
        data = np.load(self.database_path, allow_pickle=True)
        for label, seq in zip(data["labels"], data["sequences"]):
            self.templates.append(SignTemplate(label=str(label), sequence=np.asarray(seq, dtype=np.float32)))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        labels = np.array([t.label for t in self.templates], dtype=object)
        sequences = np.array([t.sequence for t in self.templates], dtype=object)
        np.savez(self.database_path, labels=labels, sequences=sequences)

    def add(self, label: str, sequence: np.ndarray) -> None:
        self.templates.append(SignTemplate(label=label, sequence=np.asarray(sequence, dtype=np.float32)))

    def known_labels(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.templates:
            counts[t.label] = counts.get(t.label, 0) + 1
        return counts

    def remove_label(self, label: str) -> int:
        before = len(self.templates)
        self.templates = [t for t in self.templates if t.label != label]
        return before - len(self.templates)
