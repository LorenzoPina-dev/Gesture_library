"""
config.py
=========
Configurazione centrale del Gesture Control Engine.

Tutti i parametri regolabili dall'utente finale (senza toccare il codice)
vivono qui e vengono anche serializzati/deserializzati da e verso JSON,
cosi' la UI grafica (ui/app.py) puo' leggerli e scriverli a runtime.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "engine_config.json",
)


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    flip_horizontal: bool = True  # effetto "specchio", piu' naturale per l'utente


@dataclass
class PreprocessingConfig:
    enable_clahe: bool = False          # bilanciamento contrasto per scarsa illuminazione
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8


@dataclass
class LandmarkerConfig:
    model_asset_path: str = os.path.join("models", "hand_landmarker.task")
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.70
    min_hand_presence_confidence: float = 0.70
    min_tracking_confidence: float = 0.65


@dataclass
class FilterConfig:
    # Scegli "ema" (semplice, robusto) o "one_euro" (piu' reattivo su movimenti rapidi)
    method: str = "one_euro"
    # --- EMA ---
    ema_alpha: float = 0.35
    # --- One Euro Filter ---
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.3
    one_euro_d_cutoff: float = 1.0


@dataclass
class RuleBasedConfig:
    # soglie normalizzate (frazioni della hand_scale) per considerare un dito "esteso"
    finger_extended_ratio: float = 1.35
    pinch_distance_ratio: float = 0.35  # sotto questa soglia => "pinch"
    fist_curl_ratio: float = 0.55       # sopra questa soglia (curl medio) => "pugno"


@dataclass
class EmbeddingConfig:
    input_dim: int = 63          # 21 landmark * 3 (x, y, z) normalizzati
    embedding_dim: int = 128
    hidden_dims: tuple = (256, 256)
    onnx_model_path: str = os.path.join("models", "gesture_embedding.onnx")
    torch_weights_path: str = os.path.join("models", "gesture_embedding.pt")


@dataclass
class KNNConfig:
    k: int = 5
    max_cosine_distance: float = 0.30   # oltre questa soglia => "UNKNOWN" (open-set rejection)
    database_path: str = os.path.join("data", "enrolled_gestures", "embeddings_db.npz")
    min_samples_per_class_for_enrollment: int = 5


@dataclass
class StateMachineConfig:
    # finestra temporale massima (secondi) per completare una sequenza dinamica
    sequence_timeout_s: float = 1.5
    # velocita' minima (unita normalizzate/secondo) per rilevare uno "swipe"
    swipe_velocity_threshold: float = 4.0


@dataclass
class EventBusConfig:
    default_cooldown_s: float = 0.4


@dataclass
class VisualizationConfig:
    draw_landmarks: bool = True
    draw_connections: bool = True
    show_debug_text: bool = True
    show_fps: bool = True


@dataclass
class EngineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    rule_based: RuleBasedConfig = field(default_factory=RuleBasedConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # --- Persistenza ---
    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str = DEFAULT_CONFIG_PATH) -> "EngineConfig":
        if not os.path.exists(path):
            cfg = EngineConfig()
            cfg.save(path)
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return EngineConfig(
            camera=CameraConfig(**raw.get("camera", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            landmarker=LandmarkerConfig(**raw.get("landmarker", {})),
            filter=FilterConfig(**raw.get("filter", {})),
            rule_based=RuleBasedConfig(**raw.get("rule_based", {})),
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            knn=KNNConfig(**raw.get("knn", {})),
            state_machine=StateMachineConfig(**raw.get("state_machine", {})),
            event_bus=EventBusConfig(**raw.get("event_bus", {})),
            visualization=VisualizationConfig(**raw.get("visualization", {})),
        )
