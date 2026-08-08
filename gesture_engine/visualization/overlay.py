"""
overlay.py
==========
Rendering di debug grafico sul frame: landmark levigati, connessioni tra
le dita, ed etichette testuali con nome della gesture rilevata e livello
di confidenza/distanza per ciascun livello del motore di riconoscimento.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

# Connessioni standard MediaPipe Hands (coppie di indici landmark)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pollice
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (0, 9), (9, 10), (10, 11), (11, 12),     # medio
    (0, 13), (13, 14), (14, 15), (15, 16),   # anulare
    (0, 17), (17, 18), (18, 19), (19, 20),   # mignolo
    (5, 9), (9, 13), (13, 17),               # palmo
]

LANDMARK_COLOR = (0, 255, 128)
CONNECTION_COLOR = (255, 180, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)


def draw_hand(frame: np.ndarray, raw_points_px: np.ndarray, viz_config) -> None:
    """Disegna landmark e connessioni sul frame (in-place). raw_points_px: (21,2) in pixel."""
    if viz_config.draw_connections:
        for a, b in HAND_CONNECTIONS:
            pa = tuple(raw_points_px[a].astype(int))
            pb = tuple(raw_points_px[b].astype(int))
            cv2.line(frame, pa, pb, CONNECTION_COLOR, 2, cv2.LINE_AA)

    if viz_config.draw_landmarks:
        for p in raw_points_px:
            cv2.circle(frame, tuple(p.astype(int)), 4, LANDMARK_COLOR, -1, cv2.LINE_AA)


def _put_text_with_bg(frame: np.ndarray, text: str, org, scale=0.6, thickness=1) -> int:
    """Disegna testo con sfondo semi-opaco per leggibilita'. Ritorna l'altezza usata."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 2), TEXT_BG_COLOR, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    return th + baseline + 10


def draw_debug_panel(
    frame: np.ndarray,
    hand_index: int,
    rule_label: Optional[str] = None,
    knn_label: Optional[str] = None,
    knn_confidence: Optional[float] = None,
    fps: Optional[float] = None,
    origin=(10, 30),
    viz_config=None,
) -> None:
    """Disegna un pannello testuale con le predizioni correnti."""
    if viz_config is not None and not viz_config.show_debug_text:
        return

    x, y = origin
    y += hand_index * 90

    lines = [f"Mano #{hand_index + 1}"]
    if rule_label is not None:
        lines.append(f"Livello1 (regole): {rule_label}")
    if knn_label is not None:
        conf_str = f" ({knn_confidence:.2f})" if knn_confidence is not None else ""
        lines.append(f"Livello2 (k-NN): {knn_label}{conf_str}")

    for line in lines:
        y += _put_text_with_bg(frame, line, (x, y))

    if fps is not None and (viz_config is None or viz_config.show_fps) and hand_index == 0:
        _put_text_with_bg(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 30))


def landmarks_to_pixel_coords(normalized_xy: np.ndarray, frame_shape) -> np.ndarray:
    """Converte coordinate normalizzate [0,1] (come fornite da MediaPipe) in pixel."""
    h, w = frame_shape[:2]
    pts = normalized_xy.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts
