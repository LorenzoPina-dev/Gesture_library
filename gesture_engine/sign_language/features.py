"""
gesture_engine/sign_language/features.py
==========================================
Estrazione delle feature per-frame usate dal riconoscitore di lingua dei
segni. A differenza del controllo gesture "puntuale" (Livello 1/2 di
gesture_engine.recognition, pensato per classificare UNA posa statica per
frame), qui serve rappresentare l'INTERA sequenza temporale di un segno -
forma delle mani + traiettoria + posizione approssimativa nello spazio di
segnazione - per poterla confrontare con dei template registrati tramite
Dynamic Time Warping (vedi dtw.py).

Limitazioni note (importanti, vedi anche i docstring degli script):
  - Nessun landmark di volto/postura: i marcatori NON manuali (espressioni
    del viso, movimento delle labbra, inclinazioni della testa - spesso
    grammaticalmente rilevanti, es. per interrogative o negazioni in molte
    lingue dei segni) NON sono catturati. Questo sistema riconosce solo la
    componente manuale dei segni.
  - La posizione nello spazio di segnazione e' approssimata dalla posizione
    grezza (coordinate immagine) del polso: e' affidabile solo se
    l'inquadratura e' ragionevolmente stabile/centrata sul segnante.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from gesture_engine.normalization.geometric import normalize_landmarks, flatten

# Blocco di feature per singola mano:
#   63  shape normalizzata (invariante a posizione/scala/rotazione 3D)
#   2   posizione grezza del polso in coordinate immagine [0,1] (x, y)
#   1   flag di presenza (1.0 se la mano e' rilevata in questo frame)
HAND_BLOCK_DIM = 63 + 2 + 1  # 66

# + 1 feature globale: distanza tra i due polsi (0 se manca una mano),
# utile per i segni a due mani che implicano contatto/vicinanza reciproca.
FRAME_FEATURE_DIM = HAND_BLOCK_DIM * 2 + 1  # 133

_LEFT_WRIST_SLICE = slice(63, 65)
_RIGHT_WRIST_SLICE = slice(HAND_BLOCK_DIM + 63, HAND_BLOCK_DIM + 65)
_LEFT_PRESENCE_IDX = HAND_BLOCK_DIM - 1
_RIGHT_PRESENCE_IDX = HAND_BLOCK_DIM * 2 - 1


def _hand_block(det) -> np.ndarray:
    if det is None:
        return np.zeros(HAND_BLOCK_DIM, dtype=np.float32)
    shape = flatten(normalize_landmarks(det.landmarks)).astype(np.float32)  # (63,)
    wrist_xy = det.landmarks[0][:2].astype(np.float32)                      # (2,)
    presence = np.array([1.0], dtype=np.float32)
    return np.concatenate([shape, wrist_xy, presence])


def split_detections_by_hand(detections: List) -> Dict[str, Optional[object]]:
    """Mappa la lista di HandDetection del frame corrente su {'Left':.., 'Right':..}
    usando l'etichetta di handedness di MediaPipe. Se compaiono due rilevazioni con
    la stessa etichetta (raro, errore di tracking), si tiene solo la prima."""
    out: Dict[str, Optional[object]] = {"Left": None, "Right": None}
    for det in detections:
        label = det.handedness if det.handedness in ("Left", "Right") else None
        if label and out[label] is None:
            out[label] = det
    return out


def build_frame_feature(detections_by_hand: Dict[str, Optional[object]]) -> np.ndarray:
    """Costruisce il vettore di feature (FRAME_FEATURE_DIM,) per un singolo frame."""
    left_block = _hand_block(detections_by_hand.get("Left"))
    right_block = _hand_block(detections_by_hand.get("Right"))

    left_det = detections_by_hand.get("Left")
    right_det = detections_by_hand.get("Right")
    if left_det is not None and right_det is not None:
        dist = float(np.linalg.norm(left_det.landmarks[0][:2] - right_det.landmarks[0][:2]))
    else:
        dist = 0.0

    return np.concatenate([left_block, right_block, np.array([dist], dtype=np.float32)])


def motion_energy(prev_feat: Optional[np.ndarray], curr_feat: np.ndarray) -> float:
    """Massima variazione di posizione del polso (sinistra o destra) rispetto
    al frame precedente. Usata per segmentare il video in segni distinti
    individuando le brevi pause tra un segno e il successivo."""
    if prev_feat is None:
        return 0.0
    d_left = float(np.linalg.norm(curr_feat[_LEFT_WRIST_SLICE] - prev_feat[_LEFT_WRIST_SLICE]))
    d_right = float(np.linalg.norm(curr_feat[_RIGHT_WRIST_SLICE] - prev_feat[_RIGHT_WRIST_SLICE]))
    return max(d_left, d_right)


def hands_present(feat: np.ndarray) -> bool:
    return feat[_LEFT_PRESENCE_IDX] > 0.5 or feat[_RIGHT_PRESENCE_IDX] > 0.5
