"""
knn_classifier.py
==================
Livello 2 (parte 2/2): classificatore k-NN a Distanza del Coseno sugli
embedding a 128-d prodotti da GestureEmbeddingNet (o dalla sua versione
ONNX, vedi onnx_inference.py).

Caratteristiche:
  - Open-Set & Outlier Rejection: se la distanza media dei k vicini piu'
    prossimi supera knn.max_cosine_distance, la predizione e' "UNKNOWN".
  - Persistenza su disco (npz) del database di embedding enrollati, cosi'
    che le gesture registrate dall'utente sopravvivano al riavvio.
  - Few-Shot Dynamic Enrollment: aggiungere una nuova classe (o piu' campioni
    a una classe esistente) e' O(1) rispetto al training della rete: basta
    aggiungere righe alla matrice di embedding, senza retraining.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

UNKNOWN_LABEL = "UNKNOWN"


@dataclass
class KNNResult:
    label: str
    mean_distance: float          # distanza coseno media dei k vicini della classe vincente
    confidence: float             # 1 - mean_distance, clampato in [0,1], comodo per la UI
    neighbor_labels: List[str]
    neighbor_distances: List[float]


class EmbeddingKNNClassifier:
    def __init__(self, knn_config):
        self.cfg = knn_config
        self._embeddings: np.ndarray = np.zeros((0, 128), dtype=np.float32)
        self._labels: List[str] = []

    # ------------------------------------------------------------------ #
    # Persistenza
    # ------------------------------------------------------------------ #
    def load(self, path: Optional[str] = None) -> None:
        path = path or self.cfg.database_path
        if not os.path.exists(path):
            self._embeddings = np.zeros((0, 128), dtype=np.float32)
            self._labels = []
            return
        data = np.load(path, allow_pickle=True)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._labels = list(data["labels"])

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.cfg.database_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            embeddings=self._embeddings,
            labels=np.array(self._labels, dtype=object),
        )

    # ------------------------------------------------------------------ #
    # Enrollment (few-shot, dinamico, senza retraining della rete)
    # ------------------------------------------------------------------ #
    def enroll(self, label: str, embeddings: np.ndarray) -> None:
        """Aggiunge N vettori embedding (N, 128) associati a `label` al database in memoria.
        Chiamare .save() per persistere su disco."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        self._embeddings = np.concatenate([self._embeddings, embeddings], axis=0)
        self._labels.extend([label] * embeddings.shape[0])

    def remove_class(self, label: str) -> int:
        """Rimuove tutti i campioni di una classe. Ritorna il numero di campioni rimossi."""
        mask = np.array([lbl != label for lbl in self._labels])
        removed = int((~mask).sum())
        self._embeddings = self._embeddings[mask]
        self._labels = [l for l, keep in zip(self._labels, mask) if keep]
        return removed

    def known_classes(self) -> Dict[str, int]:
        """Ritorna {classe: numero_campioni} per il database corrente."""
        counts: Dict[str, int] = {}
        for lbl in self._labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # Inferenza
    # ------------------------------------------------------------------ #
    def predict(self, query_embedding: np.ndarray) -> KNNResult:
        if self._embeddings.shape[0] == 0:
            return KNNResult(UNKNOWN_LABEL, 1.0, 0.0, [], [])

        q = query_embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        db = self._embeddings
        db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-8)

        cosine_sim = db_norm @ q  # (N,)
        cosine_dist = 1.0 - cosine_sim

        k = min(self.cfg.k, cosine_dist.shape[0])
        nearest_idx = np.argsort(cosine_dist)[:k]
        nearest_labels = [self._labels[i] for i in nearest_idx]
        nearest_dists = [float(cosine_dist[i]) for i in nearest_idx]

        # voto per distanza minima media, per classe
        per_class_dists: Dict[str, List[float]] = {}
        for lbl, d in zip(nearest_labels, nearest_dists):
            per_class_dists.setdefault(lbl, []).append(d)

        best_label, best_mean_dist = min(
            ((lbl, float(np.mean(ds))) for lbl, ds in per_class_dists.items()),
            key=lambda item: item[1],
        )

        if best_mean_dist > self.cfg.max_cosine_distance:
            return KNNResult(
                UNKNOWN_LABEL,
                best_mean_dist,
                0.0,
                nearest_labels,
                nearest_dists,
            )

        confidence = float(np.clip(1.0 - best_mean_dist, 0.0, 1.0))
        return KNNResult(best_label, best_mean_dist, confidence, nearest_labels, nearest_dists)
