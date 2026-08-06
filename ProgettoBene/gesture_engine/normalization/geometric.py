"""
geometric.py
============
Normalizzazione geometrica dei 21 landmark della mano per renderli
invarianti a posizione, distanza dalla camera e rotazione nel piano
immagine. Questo e' il passo che rende affidabili sia le regole
euristiche (Livello 1) sia l'embedding di deep metric learning (Livello 2).

Convenzione landmark MediaPipe Hands (21 punti, indice 0..20):
0  Wrist
1-4   Thumb (CMC, MCP, IP, TIP)
5-8   Index (MCP, PIP, DIP, TIP)
9-12  Middle (MCP, PIP, DIP, TIP)
13-16 Ring (MCP, PIP, DIP, TIP)
17-20 Pinky (MCP, PIP, DIP, TIP)
"""

from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def landmarks_to_array(landmarks) -> np.ndarray:
    """Converte una lista di 21 oggetti con .x/.y/.z (o tuple) in un array (21, 3)."""
    pts = np.zeros((21, 3), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        if hasattr(lm, "x"):
            pts[i] = (lm.x, lm.y, lm.z)
        else:
            pts[i] = lm
    return pts


def normalize_landmarks(raw_points: np.ndarray) -> np.ndarray:
    """
    Normalizza (21, 3) landmark grezzi (coordinate immagine normalizzate 0..1
    fornite da MediaPipe) applicando:
      1. Traslazione: origine sul polso (landmark 0)
      2. Rotazione planare: allinea l'asse polso->base_medio all'asse Y
      3. Scala: divide per la distanza polso->base_medio (hand_scale)

    Ritorna un array (21, 3) di float64, piatto e pronto per essere
    flattenato a 63-d per rule-based ed embedding network.
    """
    pts = raw_points.copy().astype(np.float64)

    # 1. Traslazione: origine sul polso
    origin = pts[WRIST].copy()
    pts -= origin

    # 2. Rotazione planare (solo x,y) cosi' che polso->base_medio punti verso "su" (0,-1)
    ref = pts[MIDDLE_MCP][:2]
    ref_norm = np.linalg.norm(ref)
    if ref_norm > 1e-8:
        angle = np.arctan2(ref[0], -ref[1])  # angolo rispetto all'asse (0,-1)
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts[:, :2] = pts[:, :2] @ rot.T

    # 3. Scala: normalizza per hand_scale = distanza polso -> base_medio
    hand_scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if hand_scale < 1e-8:
        hand_scale = 1e-8
    pts /= hand_scale

    return pts


def hand_scale_of(raw_points: np.ndarray) -> float:
    """Ritorna la scala reale (in coordinate immagine) polso->base_medio, utile
    per riportare soglie normalizzate a soglie in pixel se necessario."""
    return float(np.linalg.norm(raw_points[MIDDLE_MCP] - raw_points[WRIST]))


def flatten(normalized_points: np.ndarray) -> np.ndarray:
    """(21,3) -> (63,) vettore piatto, input della embedding network."""
    return normalized_points.reshape(-1).astype(np.float32)


def finger_extension_ratios(normalized_points: np.ndarray) -> dict:
    """
    Calcola, per ogni dito, il rapporto (distanza wrist->tip) / (distanza wrist->mcp).
    Un rapporto elevato indica dito esteso; un rapporto vicino a 1 (o minore)
    indica dito piegato. Usato dal riconoscitore rule-based (Livello 1).
    """
    ratios = {}
    wrist = normalized_points[WRIST]
    for finger, tip_idx in FINGER_TIPS.items():
        mcp_idx = FINGER_MCPS[finger]
        d_tip = np.linalg.norm(normalized_points[tip_idx] - wrist)
        d_mcp = np.linalg.norm(normalized_points[mcp_idx] - wrist)
        ratios[finger] = d_tip / (d_mcp + 1e-8)
    return ratios


def pinch_distance(normalized_points: np.ndarray, finger_a: str = "thumb", finger_b: str = "index") -> float:
    """Distanza normalizzata (gia' scalata da normalize_landmarks) tra due punte di dita."""
    a = normalized_points[FINGER_TIPS[finger_a]]
    b = normalized_points[FINGER_TIPS[finger_b]]
    return float(np.linalg.norm(a - b))


def average_curl(normalized_points: np.ndarray) -> float:
    """
    Stima media di "chiusura" della mano: media, su tutte le dita eccetto il pollice,
    del rapporto (distanza tip->mcp) / (distanza pip->mcp). Vicino a 1 = dito disteso,
    minore = dito piegato verso il palmo. Usato per rilevare il "pugno".
    """
    values = []
    for finger in ("index", "middle", "ring", "pinky"):
        tip = normalized_points[FINGER_TIPS[finger]]
        pip = normalized_points[FINGER_PIPS[finger]]
        mcp = normalized_points[FINGER_MCPS[finger]]
        d_tip_mcp = np.linalg.norm(tip - mcp)
        d_pip_mcp = np.linalg.norm(pip - mcp)
        values.append(d_tip_mcp / (d_pip_mcp + 1e-8))
    return float(np.mean(values)) / 2.0  # normalizzato empiricamente in [~0,~1]
