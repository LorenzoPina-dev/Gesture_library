"""
pipeline.py
===========
GestureEngine: orchestratore end-to-end dell'intera pipeline.

    Ingestion -> Preprocessing -> Landmark Detection -> Filtering anti-jitter
    -> Normalizzazione geometrica -> Riconoscimento (Livelli 1/2/3) -> Event Bus

E' pensato come API pubblica principale della libreria: un utente puo'
istanziare GestureEngine, sottoscrivere eventi tramite .on(...) e chiamare
.run() (loop bloccante con finestra di preview) oppure .step() (per
integrarlo nel proprio loop, es. dentro una UI grafica).

Esempio minimo:

    from gesture_engine import GestureEngine

    engine = GestureEngine()
    engine.on("gesture.fist", lambda e: print("Pugno rilevato!"))
    engine.on("gesture.pinch", lambda e: print("Pinch:", e.payload))
    engine.run()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from gesture_engine.config import EngineConfig
from gesture_engine.ingestion import CameraStream, HandLandmarkerEngine, preprocess_frame
from gesture_engine.ingestion.preprocessing import ContrastEnhancer
from gesture_engine.normalization import (
    landmarks_to_array,
    normalize_landmarks,
    flatten,
    build_filter,
    build_embedding_vector,
)
from gesture_engine.recognition import (
    RuleBasedRecognizer,
    EmbeddingKNNClassifier,
    EmbeddingInferenceEngine,
    GestureSequenceFSM,
    SequenceDefinition,
    SwipeDetector,
    SwipeEvent,
    UNKNOWN_LABEL,
)
from gesture_engine.events import EventBus, Event
from gesture_engine.enrollment import EnrollmentManager
from gesture_engine.visualization import draw_hand, draw_debug_panel, landmarks_to_pixel_coords


@dataclass
class HandFrameResult:
    """Risultato completo per una singola mano in un singolo frame: utile per
    chi vuole integrare il motore senza passare dal loop grafico (.run())."""
    handedness: str
    raw_points: np.ndarray            # (21,3) coordinate immagine normalizzate [0,1]
    normalized_points: np.ndarray     # (21,3) invarianti a scala/traslazione/rotazione
    embedding_input: np.ndarray       # (69,) shape normalizzata (63) + orientamento polso (6), input della embedding network
    rule_label: str
    rule_finger_count: int
    knn_label: str
    knn_confidence: float


class GestureEngine:
    """Classe principale della libreria: incapsula l'intera pipeline e
    fornisce un'API di sottoscrizione eventi in stile Observer/Pub-Sub."""

    def __init__(self, config: Optional[EngineConfig] = None, project_root: Optional[str] = None):
        self.cfg = config or EngineConfig.load()
        self._project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # --- Sotto-sistemi ---
        self._camera = CameraStream(self.cfg.camera)
        self._contrast_enhancer = ContrastEnhancer(
            clip_limit=self.cfg.preprocessing.clahe_clip_limit,
            tile_grid_size=self.cfg.preprocessing.clahe_tile_grid_size,
        )
        self._landmarker = HandLandmarkerEngine(self.cfg.landmarker, project_root=self._project_root)
        self._rule_based = RuleBasedRecognizer(self.cfg.rule_based)
        self._embedding_engine = EmbeddingInferenceEngine(self.cfg.embedding, project_root=self._project_root)
        self._knn = EmbeddingKNNClassifier(self.cfg.knn)
        self.event_bus = EventBus(self.cfg.event_bus)
        self.enrollment = None  # creato dopo l'apertura della embedding engine (vedi .open())

        # filtri anti-jitter, uno per mano/slot di tracking (fino a num_hands)
        self._filters = [build_filter(self.cfg.filter) for _ in range(self.cfg.landmarker.num_hands)]

        # Livello 3
        self._sequence_fsm = GestureSequenceFSM(self.cfg.state_machine, self._on_sequence_complete)
        self._swipe_detector = SwipeDetector(self.cfg.state_machine, self._on_swipe)

        self._is_open = False
        self._running = False
        self._last_frame_time = time.time()
        self._fps = 0.0

    # ====================================================================
    # Ciclo di vita
    # ====================================================================
    def open(self) -> None:
        if self._is_open:
            return
        self._camera.open()
        self._landmarker.open()
        self._embedding_engine.open()
        self._knn.load()
        self.enrollment = EnrollmentManager(self._embedding_engine, self._knn, self.cfg.knn)
        self._is_open = True
        self.event_bus.emit_immediate("engine.started")

    def close(self) -> None:
        if not self._is_open:
            return
        self._camera.release()
        self._landmarker.close()
        self._embedding_engine.close()
        self._is_open = False
        self.event_bus.emit_immediate("engine.stopped")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ====================================================================
    # API di sottoscrizione eventi (facciata comoda su EventBus)
    # ====================================================================
    def on(self, event_name: str, callback: Callable[[Event], None]) -> None:
        """Sottoscrive `callback` a `event_name`. Piu' callback per lo stesso
        evento sono supportati (Multi-Callback Support)."""
        self.event_bus.subscribe(event_name, callback)

    def register_sequence(self, name: str, steps: List[str], timeout_s: Optional[float] = None) -> None:
        """Registra una nuova gesture sequenziale di Livello 3, es.:
        engine.register_sequence("boom_explosion", ["fist", "open_palm"])"""
        self._sequence_fsm.register_sequence(SequenceDefinition(name=name, steps=tuple(steps), timeout_s=timeout_s))

    # ====================================================================
    # Elaborazione di un singolo frame (usabile anche fuori da .run())
    # ====================================================================
    def step(self) -> List[HandFrameResult]:
        """Cattura ed elabora un singolo frame dalla webcam. Ritorna la lista
        dei risultati (uno per mano rilevata) e pubblica gli eventi
        corrispondenti sull'EventBus. Ritorna lista vuota se nessuna mano e'
        rilevata nel frame corrente."""
        if not self._is_open:
            raise RuntimeError("GestureEngine non aperto. Usa 'with GestureEngine() as engine:' oppure .open().")

        ok, frame, timestamp_ms = self._camera.read()
        if not ok:
            return []

        frame = preprocess_frame(frame, self.cfg.preprocessing, self._contrast_enhancer)
        detections = self._landmarker.detect(frame, timestamp_ms)

        results: List[HandFrameResult] = []
        now = time.time()
        self._update_fps(now)

        for i, det in enumerate(detections):
            filt = self._filters[i % len(self._filters)]
            filtered_raw = filt.filter(det.landmarks, timestamp_ms / 1000.0)

            normalized = normalize_landmarks(filtered_raw)
            embedding_input = build_embedding_vector(filtered_raw)

            # --- Livello 1: rule-based ---
            rule_result = self._rule_based.recognize(normalized)

            # --- Livello 2: embedding + k-NN open-set ---
            embedding = self._embedding_engine.embed(embedding_input)
            knn_result = self._knn.predict(embedding)

            # --- Livello 3: FSM temporale + swipe ---
            self._sequence_fsm.update(rule_result.label, now=now)
            self._swipe_detector.update(filtered_raw[0][:2], now=now)

            result = HandFrameResult(
                handedness=det.handedness,
                raw_points=filtered_raw,
                normalized_points=normalized,
                embedding_input=embedding_input,
                rule_label=rule_result.label,
                rule_finger_count=rule_result.finger_count,
                knn_label=knn_result.label,
                knn_confidence=knn_result.confidence,
            )
            results.append(result)

            self._emit_frame_events(i, det.handedness, rule_result, knn_result)

        self._last_frame_bgr = frame
        self._last_detections = detections
        self._last_results = results
        return results

    def _emit_frame_events(self, hand_index: int, handedness: str, rule_result, knn_result) -> None:
        payload = {
            "hand_index": hand_index,
            "handedness": handedness,
            "finger_count": rule_result.finger_count,
            "pinch_strength": rule_result.pinch_strength,
        }
        self.event_bus.emit(f"gesture.{rule_result.label}", payload=payload)

        if knn_result.label != UNKNOWN_LABEL:
            self.event_bus.emit(
                f"custom_gesture.{knn_result.label}",
                payload={**payload, "confidence": knn_result.confidence},
            )

    def _on_sequence_complete(self, sequence_name: str) -> None:
        self.event_bus.emit(f"sequence.{sequence_name}", payload={"name": sequence_name})

    def _on_swipe(self, swipe_event: SwipeEvent) -> None:
        self.event_bus.emit(
            f"swipe.{swipe_event.direction}",
            payload={"direction": swipe_event.direction, "velocity": swipe_event.velocity},
        )

    def _update_fps(self, now: float) -> None:
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self._fps * 0.9 + instant_fps * 0.1 if self._fps > 0 else instant_fps

    # ====================================================================
    # Loop grafico "batteries-included" (facoltativo)
    # ====================================================================
    def run(self, window_name: str = "Gesture Control Engine") -> None:
        """Loop bloccante con finestra OpenCV di preview e overlay di debug.
        Premi 'q' o ESC per uscire. Utile per demo rapide; per integrazioni
        custom preferire .step() dentro il proprio loop applicativo."""
        opened_here = not self._is_open
        if opened_here:
            self.open()
        self._running = True
        try:
            while self._running:
                results = self.step()
                frame = getattr(self, "_last_frame_bgr", None)
                if frame is None:
                    continue

                if self.cfg.visualization.draw_landmarks or self.cfg.visualization.draw_connections:
                    for det in getattr(self, "_last_detections", []):
                        pts_px = landmarks_to_pixel_coords(det.landmarks[:, :2], frame.shape)
                        draw_hand(frame, pts_px, self.cfg.visualization)

                for i, r in enumerate(results):
                    draw_debug_panel(
                        frame,
                        hand_index=i,
                        rule_label=r.rule_label,
                        knn_label=r.knn_label,
                        knn_confidence=r.knn_confidence,
                        fps=self._fps,
                        viz_config=self.cfg.visualization,
                    )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' o ESC
                    break
        finally:
            self._running = False
            cv2.destroyWindow(window_name) if cv2.getWindowProperty(window_name, 0) >= 0 else None
            if opened_here:
                self.close()

    def stop(self) -> None:
        """Segnala l'uscita da un loop .run() in corso (es. da un altro thread)."""
        self._running = False
