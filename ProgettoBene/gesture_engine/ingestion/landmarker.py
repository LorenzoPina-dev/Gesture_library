"""
landmarker.py
=============
Wrapper attorno a MediaPipe Tasks HandLandmarker, configurato in
RunningMode.VIDEO cosi' da poter passare frame_timestamp_ms espliciti e
garantire una sincronizzazione temporale corretta (necessaria per il
tracking interno di MediaPipe e per il Livello 3 - State Machine temporale).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


@dataclass
class HandDetection:
    handedness: str            # "Left" o "Right"
    handedness_score: float
    landmarks: np.ndarray       # (21, 3) coordinate normalizzate immagine [0,1] + z relativo
    world_landmarks: Optional[np.ndarray] = None  # (21, 3) in metri, se disponibile


class HandLandmarkerEngine:
    """
    Incapsula il ciclo di vita di mp.tasks.vision.HandLandmarker in modalita' VIDEO.
    """

    def __init__(self, landmarker_config, project_root: Optional[str] = None):
        self.cfg = landmarker_config
        self._landmarker: Optional[mp_vision.HandLandmarker] = None
        self._project_root = project_root or os.getcwd()

    def _resolve_model_path(self) -> str:
        path = self.cfg.model_asset_path
        if os.path.isabs(path) and os.path.exists(path):
            return path
        candidate = os.path.join(self._project_root, path)
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(path):
            return path
        raise FileNotFoundError(
            f"Modello HandLandmarker non trovato in '{candidate}'.\n"
            f"Scarica 'hand_landmarker.task' da:\n"
            f"  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/latest/hand_landmarker.task\n"
            f"e posizionalo nella cartella 'models/' del progetto."
        )

    def open(self) -> None:
        model_path = self._resolve_model_path()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_hand_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame: np.ndarray, frame_timestamp_ms: int) -> List[HandDetection]:
        if self._landmarker is None:
            raise RuntimeError("HandLandmarkerEngine non aperto. Chiama .open() prima di .detect().")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        detections: List[HandDetection] = []
        if not result.hand_landmarks:
            return detections

        for i, hand_lms in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float64)

            world_pts = None
            if getattr(result, "hand_world_landmarks", None):
                try:
                    world_hand = result.hand_world_landmarks[i]
                    world_pts = np.array([[lm.x, lm.y, lm.z] for lm in world_hand], dtype=np.float64)
                except (IndexError, TypeError):
                    world_pts = None

            handedness_label = "Unknown"
            handedness_score = 0.0
            if result.handedness and len(result.handedness) > i and result.handedness[i]:
                cat = result.handedness[i][0]
                handedness_label = cat.category_name
                handedness_score = cat.score

            detections.append(
                HandDetection(
                    handedness=handedness_label,
                    handedness_score=handedness_score,
                    landmarks=pts,
                    world_landmarks=world_pts,
                )
            )
        return detections

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
