"""
gesture_engine/control/stabilizer.py
======================================
Risoluzione gesture "confermata" per il controllo diretto (mouse, dual-hand,
ecc.): combina k-NN (Livello 2) + fallback rule-based (Livello 1) + conferma
geometrica esplicita per pinch/middle_pinch + debounce temporale.

Estratta da run_mouse_control.py cosi' da essere riusabile anche da
run_dual_hand_control.py (una istanza per mano).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from gesture_engine import GestureEngine
from gesture_engine.normalization import pinch_distance


class GestureStabilizer:
    MIN_KNN_CONFIDENCE = 0.55       # sotto questa confidenza non ci si fida del k-NN
    STABILITY_FRAMES = 3            # frame consecutivi uguali richiesti per confermare un cambio
    PINCH_AMBIGUITY_MARGIN = 0.85   # una distanza deve essere < 85% dell'altra per non essere ambigua

    def __init__(self, engine: GestureEngine):
        self._engine = engine
        self._history: deque[str] = deque(maxlen=self.STABILITY_FRAMES)
        self.confirmed_gesture: str = "UNKNOWN"
        # Ultima diagnosi calcolata (per il logging esterno, es. run_dual_hand_control.py):
        # cosa hanno detto knn/rule-based/geometria PRIMA del debounce.
        self.last_knn_label: str = "UNKNOWN"
        self.last_knn_confidence: float = 0.0
        self.last_rule_label: str = "UNKNOWN"
        self.last_geo_label: Optional[str] = None
        self.last_raw_label: str = "UNKNOWN"

    def reset(self) -> None:
        self._history.clear()
        self.confirmed_gesture = "UNKNOWN"

    def _classify_pinch_family(self, normalized_points) -> Optional[str]:
        """Determina geometricamente se e' davvero un pinch indice-pollice,
        medio-pollice, o nessuno dei due (dita non abbastanza vicine, o
        distanza ambigua tra le due candidate)."""
        threshold = self._engine.cfg.rule_based.pinch_distance_ratio

        d_index = pinch_distance(normalized_points, "thumb", "index")
        d_middle = pinch_distance(normalized_points, "thumb", "middle")

        index_closed = d_index <= threshold
        middle_closed = d_middle <= threshold

        if index_closed and not middle_closed:
            return "pinch"
        if middle_closed and not index_closed:
            return "middle_pinch"
        if index_closed and middle_closed:
            if d_index <= d_middle * self.PINCH_AMBIGUITY_MARGIN:
                return "pinch"
            if d_middle <= d_index * self.PINCH_AMBIGUITY_MARGIN:
                return "middle_pinch"
            return None
        return None

    def _resolve_raw_label(self, hand) -> str:
        # Il k-NN e' ora la fonte PRIMARIA della decisione: se e' sicuro (nota
        # e confidenza >= soglia), la sua etichetta e' quella di partenza e
        # rule-based/geometria possono solo CONFERMARLA o disambiguarla
        # all'interno della stessa famiglia (pinch vs middle_pinch), mai
        # scavalcarla del tutto. Prima invece la geometria aveva sempre
        # l'ultima parola sul pinch anche quando il k-NN diceva con sicurezza
        # qualcos'altro (es. 'puntatore'): risultato, chiudendo la mano per
        # fare un pinch indice-pollice il pollice si avvicinava per errore
        # anche al medio e la geometria dichiarava 'middle_pinch' anche se il
        # k-NN aveva gia' correttamente riconosciuto un'altra gesture.
        knn_confident = hand.knn_label != "UNKNOWN" and hand.knn_confidence >= self.MIN_KNN_CONFIDENCE
        label = hand.knn_label if knn_confident else hand.rule_label

        geo_label = self._classify_pinch_family(hand.normalized_points)

        if knn_confident:
            if label in ("pinch", "middle_pinch"):
                # Il k-NN ha gia' deciso che e' un pinch: la geometria (distanza
                # REALE pollice-indice/medio) puo' correggere QUALE dei due e'
                # (pinch vs middle_pinch), perche' li' e' piu' affidabile del
                # k-NN sulla sotto-categoria. Se pero' in questo frame la
                # geometria non conferma nessun pinch (dita non ancora
                # chiuse/misurazione ambigua), ci si fida comunque del k-NN.
                if geo_label is not None:
                    label = geo_label
            # Se il k-NN dice qualcos'altro (es. 'puntatore', 'fist', ...) la
            # geometria NON lo scavalca piu': evita i falsi 'middle_pinch'
            # durante la chiusura della mano per un pinch normale.
        else:
            # k-NN non sicuro: si ricade sul comportamento precedente, dove
            # rule-based/geometria hanno l'ultima parola sul pinch.
            if geo_label is not None and hand.rule_label != "fist":
                label = geo_label
            elif label in ("pinch", "middle_pinch"):
                # Il rule-based pensava fosse un pinch ma la geometria dice di
                # no (non ancora chiuso, o ambiguo): niente click a vuoto.
                label = "puntatore" if hand.rule_label == "1_fingers" else hand.rule_label

        # Salva la diagnosi per il logging esterno (vedi run_dual_hand_control.py)
        self.last_knn_label = hand.knn_label
        self.last_knn_confidence = hand.knn_confidence
        self.last_rule_label = hand.rule_label
        self.last_geo_label = geo_label
        self.last_raw_label = label

        return label

    def update(self, hand) -> str:
        raw_label = self._resolve_raw_label(hand)
        self._history.append(raw_label)

        if len(self._history) == self._history.maxlen and len(set(self._history)) == 1:
            self.confirmed_gesture = raw_label

        return self.confirmed_gesture
