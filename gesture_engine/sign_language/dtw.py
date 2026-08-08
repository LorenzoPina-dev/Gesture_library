"""
gesture_engine/sign_language/dtw.py
=====================================
Dynamic Time Warping per confrontare due sequenze di feature per-frame di
lunghezza diversa (la stessa persona non esegue mai un segno esattamente
alla stessa velocita' due volte). Implementazione O(N*M) in numpy puro,
senza dipendenze esterne (niente scipy/fastdtw da installare).

E' la stessa idea concettuale del rigetto open-set del k-NN a distanza di
coseno usato per le gesture statiche (vedi recognition/knn_classifier.py),
applicata pero' a intere sequenze temporali anziche' a singoli embedding.
"""

from __future__ import annotations

import numpy as np


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """seq_a: (N, D), seq_b: (M, D). Ritorna il costo di allineamento DTW
    normalizzato per la lunghezza del cammino, cosi' da poter confrontare
    coppie di sequenze di lunghezza diversa sulla stessa scala."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    diff = seq_a[:, None, :] - seq_b[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, M) distanza euclidea locale

    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row = cost[i - 1]
        acc_prev = acc[i - 1]
        acc_curr = acc[i]
        for j in range(1, m + 1):
            acc_curr[j] = row[j - 1] + min(acc_prev[j], acc_curr[j - 1], acc_prev[j - 1])

    return float(acc[n, m] / (n + m))


def best_match(sequence: np.ndarray, templates, max_len: int = 150):
    """Confronta `sequence` con una lista di SignTemplate e ritorna
    (label_migliore, distanza_migliore). Le sequenze troppo lunghe vengono
    sotto-campionate uniformemente a `max_len` frame per limitare il costo
    computazionale del DTW (O(N*M)) senza perdere la forma della traiettoria."""
    query = _downsample(sequence, max_len)

    best_label, best_dist = None, float("inf")
    for tpl in templates:
        candidate = _downsample(tpl.sequence, max_len)
        dist = dtw_distance(query, candidate)
        if dist < best_dist:
            best_label, best_dist = tpl.label, dist
    return best_label, best_dist


def _downsample(seq: np.ndarray, max_len: int) -> np.ndarray:
    if len(seq) <= max_len:
        return seq
    idx = np.linspace(0, len(seq) - 1, max_len).astype(int)
    return seq[idx]
