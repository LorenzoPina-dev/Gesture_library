"""
state_machine.py
=================
Livello 3 del motore di riconoscimento: Finite State Machine (FSM) guidata
dal tempo per gesture dinamiche/sequenziali, che compongono in sequenza
le etichette prodotte dai Livelli 1/2 (es. rule-based "fist" -> "open_palm")
oppure pattern di velocita' (es. "swipe" rilevato da un salto rapido di
posizione del polso).

Due tipologie di sequenza supportate "out of the box":

1. GestureSequence: sequenza discreta di etichette (es. ["fist", "open_palm"])
   da completare entro `sequence_timeout_s`. Esempio: "Boom/Explosion".

2. SwipeDetector: rileva movimenti rapidi del polso lungo un asse (velocita'
   normalizzata sopra soglia) e li classifica in direzioni
   (left/right/up/down). Esempio: swipe di navigazione o "schiocco".

Entrambe emettono eventi tramite callback fornita dall'EventBus (Livello 4),
disaccoppiando completamente la logica di stato dalla gestione eventi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class SequenceDefinition:
    name: str                       # nome evento emesso, es. "boom_explosion"
    steps: Sequence[str]            # sequenza di label attese, es. ("fist", "open_palm")
    timeout_s: Optional[float] = None  # override del timeout globale, se serve


class GestureSequenceFSM:
    """
    Tiene traccia, per ciascuna SequenceDefinition registrata, di quanti step
    consecutivi sono gia' stati osservati e del tempo trascorso dal primo step.
    Se la sequenza si completa entro il timeout, emette l'evento; se scade il
    tempo o arriva una label che rompe la sequenza, si resetta.
    """

    def __init__(self, state_machine_config, on_sequence_complete: Callable[[str], None]):
        self.cfg = state_machine_config
        self._on_complete = on_sequence_complete
        self._definitions: List[SequenceDefinition] = []
        # stato per-definizione: (step_index_corrente, tempo_inizio)
        self._progress = {}

    def register_sequence(self, definition: SequenceDefinition) -> None:
        self._definitions.append(definition)
        self._progress[definition.name] = (0, None)

    def update(self, current_label: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()

        for d in self._definitions:
            step_idx, start_t = self._progress[d.name]
            timeout = d.timeout_s or self.cfg.sequence_timeout_s

            if step_idx > 0 and start_t is not None and (now - start_t) > timeout:
                step_idx, start_t = 0, None  # timeout: reset

            expected_label = d.steps[step_idx]

            if current_label == expected_label:
                if step_idx == 0:
                    start_t = now
                step_idx += 1
                if step_idx >= len(d.steps):
                    self._on_complete(d.name)
                    step_idx, start_t = 0, None  # pronta per la prossima volta
            elif step_idx > 0 and current_label == d.steps[step_idx - 1]:
                # l'utente e' ancora "fermo" sullo step precedente: nessuna penalita'
                pass
            else:
                # label inattesa: reset della sequenza (solo se non e' lo step 0 stesso)
                if current_label != d.steps[0]:
                    step_idx, start_t = 0, None

            self._progress[d.name] = (step_idx, start_t)


@dataclass
class SwipeEvent:
    direction: str    # "left", "right", "up", "down"
    velocity: float


class SwipeDetector:
    """
    Rileva swipe/schiocchi rapidi osservando la velocita' del polso (landmark 0,
    in coordinate normalizzate immagine) tra frame consecutivi. Quando la
    velocita' supera `swipe_velocity_threshold`, classifica la direzione
    dominante ed emette l'evento (con cooldown gestito a monte dall'EventBus).
    """

    def __init__(self, state_machine_config, on_swipe: Callable[[SwipeEvent], None]):
        self.cfg = state_machine_config
        self._on_swipe = on_swipe
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    def update(self, wrist_xy: np.ndarray, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()

        if self._prev_pos is None:
            self._prev_pos, self._prev_t = wrist_xy.copy(), now
            return

        dt = max(now - self._prev_t, 1e-4)
        delta = wrist_xy - self._prev_pos
        velocity_vec = delta / dt
        speed = float(np.linalg.norm(velocity_vec))

        if speed >= self.cfg.swipe_velocity_threshold:
            dx, dy = velocity_vec
            if abs(dx) >= abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
            self._on_swipe(SwipeEvent(direction=direction, velocity=speed))
            # evita di "ri-innescare" sullo stesso movimento nei frame immediatamente successivi
            self._prev_pos, self._prev_t = wrist_xy.copy(), now
            return

        self._prev_pos, self._prev_t = wrist_xy.copy(), now
