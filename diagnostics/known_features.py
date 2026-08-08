"""
diagnostics/known_features.py
==============================
Estrae, a partire dai vettori di input a 69-d della embedding network
(build_embedding_vector: 63 shape normalizzata + 6 orientamento), un
dizionario di feature esplicite e continue gia' note "per costruzione"
(non serve la rete per calcolarle: sono geometria pura). Servono come
"ground truth" contro cui testare cosa la rete ha effettivamente imparato
a rappresentare (linear probing, correlazione per-dimensione, colorazione
delle proiezioni 2D).

Nota: lavora direttamente sui vettori 69-d gia' salvati in
data/training_dataset.npz, senza bisogno dei landmark grezzi originali,
perche' build_embedding_vector e' deterministica e invertibile per la
parte che ci interessa (i 6 valori di orientamento sono sin/cos degli
angoli, quindi ricostruibili con atan2; i primi 63 valori sono gia' la
shape normalizzata (21,3), quindi pinch/curl/estensione dita si ricalcolano
direttamente su quella).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from gesture_engine.normalization.geometric import (
    FINGER_TIPS,
    FINGER_MCPS,
    FINGER_PIPS,
    WRIST,
)


def extract_known_features(vectors: np.ndarray) -> Dict[str, np.ndarray]:
    """
    vectors: (N, 69) float32, prodotti da build_embedding_vector.
    Ritorna un dict {nome_feature: array (N,)} di feature continue note,
    utile per: colorare le proiezioni 2D, correlazione per-dimensione,
    linear probing.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    shape = vectors[:, :63].reshape(-1, 21, 3)   # (N, 21, 3) normalizzato
    orient = vectors[:, 63:69]                    # sin/cos di roll,pitch,yaw

    roll = np.arctan2(orient[:, 0], orient[:, 3])
    pitch = np.arctan2(orient[:, 1], orient[:, 4])
    yaw = np.arctan2(orient[:, 2], orient[:, 5])

    wrist = shape[:, WRIST, :]

    features: Dict[str, np.ndarray] = {
        "orient_roll_rad": roll,
        "orient_pitch_rad": pitch,
        "orient_yaw_rad": yaw,
        "pinch_thumb_index": np.linalg.norm(
            shape[:, FINGER_TIPS["thumb"], :] - shape[:, FINGER_TIPS["index"], :], axis=1
        ),
    }

    # rapporto di estensione per dito: dist(wrist,tip) / dist(wrist,mcp)
    for finger, tip_idx in FINGER_TIPS.items():
        mcp_idx = FINGER_MCPS[finger]
        d_tip = np.linalg.norm(shape[:, tip_idx, :] - wrist, axis=1)
        d_mcp = np.linalg.norm(shape[:, mcp_idx, :] - wrist, axis=1)
        features[f"extension_{finger}"] = d_tip / (d_mcp + 1e-8)

    # curl medio (escluso pollice), stessa formula di average_curl in geometric.py
    curl_vals = []
    for finger in ("index", "middle", "ring", "pinky"):
        tip = shape[:, FINGER_TIPS[finger], :]
        pip = shape[:, FINGER_PIPS[finger], :]
        mcp = shape[:, FINGER_MCPS[finger], :]
        d_tip_mcp = np.linalg.norm(tip - mcp, axis=1)
        d_pip_mcp = np.linalg.norm(pip - mcp, axis=1)
        curl_vals.append(d_tip_mcp / (d_pip_mcp + 1e-8))
    features["average_curl"] = (np.mean(curl_vals, axis=0)) / 2.0

    return {k: v.astype(np.float64) for k, v in features.items()}
