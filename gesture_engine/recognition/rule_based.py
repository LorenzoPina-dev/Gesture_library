"""
rule_based.py
=============
Livello 1 del motore di riconoscimento: regole euristiche su distanze
euclidee relative, calcolate sui landmark GIA' normalizzati
(gesture_engine.normalization.geometric.normalize_landmarks).

Copre:
  - Conteggio dita estese (0-5)
  - Rilevamento pugno ("fist")
  - Rilevamento pinch adattivo (pollice-indice, normalizzato alla mano)

Questo livello e' intenzionalmente semplice e a bassissima latenza: serve
come base solida sempre disponibile, anche prima/senza il livello 2 (embedding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from gesture_engine.normalization.geometric import (
    finger_extension_ratios,
    pinch_distance,
    average_curl,
)


@dataclass
class RuleBasedResult:
    extended_fingers: Dict[str, bool]
    finger_count: int
    is_fist: bool
    is_pinch: bool
    pinch_strength: float  # 1.0 = massimo pinch (distanza 0), 0.0 = dita lontane
    label: str              # etichetta sintetica, es. "fist", "open_palm", "3_fingers", "pinch"


class RuleBasedRecognizer:
    def __init__(self, rule_based_config):
        self.cfg = rule_based_config

    def recognize(self, normalized_points: np.ndarray) -> RuleBasedResult:
        ratios = finger_extension_ratios(normalized_points)
        extended = {f: (r >= self.cfg.finger_extended_ratio) for f, r in ratios.items()}
        finger_count = sum(extended.values())

        curl = average_curl(normalized_points)
        is_fist = curl <= self.cfg.fist_curl_ratio and finger_count <= 1

        pinch_d = pinch_distance(normalized_points, "thumb", "index")
        is_pinch = pinch_d <= self.cfg.pinch_distance_ratio
        # forza del pinch: 1.0 quando distanza=0, 0.0 quando distanza>=soglia*2
        pinch_strength = float(np.clip(1.0 - pinch_d / (self.cfg.pinch_distance_ratio * 2), 0.0, 1.0))

        if is_fist:
            label = "fist"
        elif is_pinch:
            label = "pinch"
        elif finger_count == 5:
            label = "open_palm"
        elif finger_count == 0:
            label = "fist"
        else:
            label = f"{finger_count}_fingers"

        return RuleBasedResult(
            extended_fingers=extended,
            finger_count=finger_count,
            is_fist=is_fist,
            is_pinch=is_pinch,
            pinch_strength=pinch_strength,
            label=label,
        )
