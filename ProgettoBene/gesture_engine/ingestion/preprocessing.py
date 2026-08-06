"""
preprocessing.py
=================
Pre-processing opzionale del frame prima dell'inferenza MediaPipe.
Implementa CLAHE (Contrast Limited Adaptive Histogram Equalization) sul
canale L dello spazio colore LAB, utile in condizioni di scarsa illuminazione
o con webcam di bassa qualita', senza alterare i colori (canali a,b intatti).
"""

from __future__ import annotations

import cv2
import numpy as np


class ContrastEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
        )

    def apply(self, bgr_frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_frame(bgr_frame: np.ndarray, preprocessing_config, enhancer: ContrastEnhancer = None) -> np.ndarray:
    """Applica la pipeline di pre-processing configurata a un frame BGR."""
    frame = bgr_frame
    if preprocessing_config.enable_clahe:
        if enhancer is None:
            enhancer = ContrastEnhancer(
                clip_limit=preprocessing_config.clahe_clip_limit,
                tile_grid_size=preprocessing_config.clahe_tile_grid_size,
            )
        frame = enhancer.apply(frame)
    return frame
