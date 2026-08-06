"""
camera_stream.py
=================
Wrapper leggero attorno a cv2.VideoCapture che gestisce apertura, chiusura,
retry, e calcolo del timestamp relativo in millisecondi richiesto da
MediaPipe Tasks in RunningMode.VIDEO.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraStream:
    def __init__(self, camera_config):
        self.cfg = camera_config
        self._cap: Optional[cv2.VideoCapture] = None
        self._start_time_ms: Optional[float] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.cfg.device_index, cv2.CAP_DSHOW if _is_windows() else 0)
        if not self._cap.isOpened():
            # fallback senza backend specifico (utile su Linux/Mac)
            self._cap = cv2.VideoCapture(self.cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Impossibile aprire la webcam all'indice {self.cfg.device_index}. "
                f"Verifica che non sia in uso da un'altra applicazione."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)
        self._start_time_ms = time.time() * 1000.0

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Ritorna (successo, frame_bgr, frame_timestamp_ms) dove il timestamp
        e' monotono e relativo all'apertura dello stream (richiesto da MediaPipe)."""
        if self._cap is None:
            raise RuntimeError("CameraStream non aperto. Chiama .open() prima di .read().")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None, 0
        if self.cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000.0 - self._start_time_ms)
        return True, frame, timestamp_ms

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def _is_windows() -> bool:
    import platform
    return platform.system().lower().startswith("win")
