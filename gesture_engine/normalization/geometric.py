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


def wrist_orientation_matrix(raw_points: np.ndarray) -> np.ndarray:
    """
    Costruisce la base ortonormale 3D locale della mano (x="destra" attraverso
    il palmo, y=polso->base_medio cioe' "su", z=normale al palmo) espressa
    nel sistema di riferimento della camera, a partire dai landmark GREZZI
    (prima di qualunque normalizzazione). E' la matrice di rotazione che
    porta la mano nel suo orientamento canonico.

    Ritorna una matrice 3x3 le cui colonne sono i tre assi locali della mano
    espressi nel frame camera (R: hand-local -> camera). Se i landmark sono
    degeneri (mano di taglio, punti coincidenti) ritorna l'identita'.
    """
    pts = raw_points.astype(np.float64)
    y_axis = pts[MIDDLE_MCP] - pts[WRIST]
    x_ref = pts[FINGER_MCPS["pinky"]] - pts[FINGER_MCPS["index"]]

    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return np.eye(3)
    y_axis = y_axis / y_norm

    z_axis = np.cross(x_ref, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return np.eye(3)
    z_axis = z_axis / z_norm

    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return np.eye(3)
    x_axis = x_axis / x_norm

    return np.column_stack([x_axis, y_axis, z_axis])


def wrist_roll_pitch_yaw(raw_points: np.ndarray) -> np.ndarray:
    """
    Ritorna (roll, pitch, yaw) in radianti: orientamento 3D del polso rispetto
    alla camera, calcolato PRIMA di qualunque normalizzazione rotazionale.
    E' esattamente l'informazione che normalize_landmarks() rimuove per
    rendere la shape invariante alla rotazione — qui la recuperiamo come
    feature esplicita separata (vedi orientation_features), cosi' gesture
    identiche nella forma ma con significato opposto in base all'orientamento
    (es. pollice su vs pollice giu') restano distinguibili dalla rete.
    """
    R = wrist_orientation_matrix(raw_points)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def orientation_features(raw_points: np.ndarray) -> np.ndarray:
    """
    Codifica (roll, pitch, yaw) come 6 valori (sin, cos) per ciascun angolo,
    invece dell'angolo grezzo in radianti: evita la discontinuita' di
    wraparound a +-pi (179 gradi e -179 gradi sono vicinissimi nella realta'
    ma numericamente lontani), che confonderebbe sia la Triplet Loss sia la
    distanza del coseno usata dal k-NN. Ritorna un vettore (6,) float32:
    [sin(roll), sin(pitch), sin(yaw), cos(roll), cos(pitch), cos(yaw)].
    """
    angles = wrist_roll_pitch_yaw(raw_points)
    return np.concatenate([np.sin(angles), np.cos(angles)]).astype(np.float32)


def normalize_landmarks(raw_points: np.ndarray) -> np.ndarray:
    """
    Normalizza (21, 3) landmark grezzi (coordinate immagine normalizzate 0..1
    fornite da MediaPipe) applicando:
      1. Traslazione: origine sul polso (landmark 0)
      2. Rotazione 3D COMPLETA (roll, pitch, yaw): riporta la mano in un
         orientamento canonico, invariante alla rotazione nello spazio (non
         solo nel piano immagine come in precedenza)
      3. Scala: divide per la distanza polso->base_medio (hand_scale)

    Nota: questa invarianza rotazionale e' voluta per lo SHAPE (forma della
    mano: quante dita estese, pugno, pinch, ecc.) ed e' cio' che consumano
    rule_based.py e, come componente "forma", l'embedding network. L'unica
    informazione di orientamento che questa funzione rimuove va recuperata
    esplicitamente con orientation_features()/wrist_roll_pitch_yaw() per le
    gesture il cui significato dipende dall'orientamento (vedi build_embedding_vector).

    Ritorna un array (21, 3) di float64, pronto per essere flattenato.
    """
    pts = raw_points.copy().astype(np.float64)

    # 1. Traslazione: origine sul polso
    origin = pts[WRIST].copy()
    pts -= origin

    # 2. Rotazione 3D completa: porta la mano nel suo frame locale canonico
    R = wrist_orientation_matrix(raw_points)
    pts = pts @ R  # world/camera -> hand-local frame

    # 3. Scala: normalizza per hand_scale = distanza polso -> base_medio
    hand_scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if hand_scale < 1e-8:
        hand_scale = 1e-8
    pts /= hand_scale

    return pts


def build_embedding_vector(raw_points: np.ndarray) -> np.ndarray:
    """
    Vettore di input (69,) per la rete di embedding (Livello 2): concatena
    la shape normalizzata e invariante alla rotazione (63-d, da
    normalize_landmarks + flatten) con le feature esplicite di orientamento
    del polso (6-d, sin/cos di roll/pitch/yaw, da orientation_features).

    Cosi' la rete puo' sia generalizzare per FORMA indipendentemente da come
    e' orientata la mano nello spazio, sia distinguere gesture con la stessa
    forma ma significato opposto in base alla rotazione (es. pollice su/giu',
    che sono quasi la stessa forma ruotata di ~180 gradi).

    Usare SEMPRE questa funzione (non flatten(normalize_landmarks(...)) da
    solo) come input della embedding network, sia in training che in
    enrollment che a runtime, per coerenza.
    """
    normalized = normalize_landmarks(raw_points)
    shape_flat = flatten(normalized)
    orient = orientation_features(raw_points)
    return np.concatenate([shape_flat, orient]).astype(np.float32)


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
