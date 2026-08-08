"""
gesture_engine/control/filters.py
===================================
Filtro One Euro applicato direttamente in coordinate schermo (px), usato dai
controller che pilotano il mouse/cursore (run_mouse_control.py,
run_dual_hand_control.py). E' concettualmente lo stesso filtro usato dal
motore sui landmark grezzi (gesture_engine.normalization.filters), qui
riapplicato al livello di output per una UX di puntamento ottimizzata
(stabile da fermo, reattivo in movimento) indipendente dallo smoothing
interno del motore.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import pyautogui


class _OneEuroScalar:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(t - self.t_prev, 1e-6)

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class CursorController:
    """Muove il mouse in modo fluido e reattivo tramite One Euro Filter."""

    MIN_CUTOFF = 0.9
    BETA = 1.2
    D_CUTOFF = 1.0

    # Moltiplicatore applicato al delta del polso durante il drag: un valore
    # > 1 rende il drag piu' sensibile (basta muovere la mano di meno per
    # spostare il cursore della stessa distanza sullo schermo).
    DRAG_SENSITIVITY = 1.8

    # L'asse orizzontale del cursore va invertito rispetto alla coordinata X
    # grezza del landmark: il frame della webcam viene gia' specchiato da
    # CameraStream (flip_horizontal, vedi engine_config.json) per la preview
    # e per la corretta etichettatura Left/Right di MediaPipe, ma su molti
    # setup Windows il driver della webcam (backend DirectShow) consegna un
    # frame GIA' specchiato di suo: il flip del codice lo specchia una
    # SECONDA volta, annullando l'effetto e lasciando l'asse X invertito
    # rispetto al movimento fisico della mano (muovi la mano a destra, il
    # cursore va a sinistra). Invertire qui, a valle, corregge il mapping
    # cursore<->schermo senza toccare la detection/l'etichettatura Left/Right
    # di MediaPipe (che dipende dal frame specchiato per essere corretta).
    INVERT_X = True

    def __init__(self):
        self._fx = _OneEuroScalar(self.MIN_CUTOFF, self.BETA, self.D_CUTOFF)
        self._fy = _OneEuroScalar(self.MIN_CUTOFF, self.BETA, self.D_CUTOFF)
        self._wrist_fx = _OneEuroScalar(self.MIN_CUTOFF, self.BETA, self.D_CUTOFF)
        self._wrist_fy = _OneEuroScalar(self.MIN_CUTOFF, self.BETA, self.D_CUTOFF)
        self._last_wrist_x: Optional[float] = None
        self._last_wrist_y: Optional[float] = None
        self._screen_w, self._screen_h = pyautogui.size()

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._wrist_fx.reset()
        self._wrist_fy.reset()
        self._last_wrist_x = None
        self._last_wrist_y = None

    def move_to(self, raw_x: float, raw_y: float) -> None:
        """Movimento ASSOLUTO (puntatore): mappa la posizione 1:1 sullo
        schermo. raw_x/raw_y in [0,1] (coordinate immagine normalizzate)."""
        now = time.time()
        mapped_x = (1.0 - raw_x) if self.INVERT_X else raw_x
        target_x = mapped_x * self._screen_w
        target_y = raw_y * self._screen_h

        smooth_x = self._fx.filter(target_x, now)
        smooth_y = self._fy.filter(target_y, now)

        pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)

    def begin_relative_tracking(self, wrist_x: float, wrist_y: float) -> None:
        """Da chiamare all'inizio di un drag: fissa il riferimento del polso
        SENZA spostare il cursore (niente salto al primo frame)."""
        now = time.time()
        self._wrist_fx.reset()
        self._wrist_fy.reset()
        self._last_wrist_x = self._wrist_fx.filter(wrist_x, now)
        self._last_wrist_y = self._wrist_fy.filter(wrist_y, now)

    def move_by_wrist(self, wrist_x: float, wrist_y: float) -> None:
        """Movimento RELATIVO (drag): il cursore si sposta del DELTA del
        polso rispetto al frame precedente, invece di seguire la posizione
        assoluta della punta del dito (che si muove anche solo chiudendo il
        pinch, causando drag involontari se usata direttamente)."""
        now = time.time()
        fx = self._wrist_fx.filter(wrist_x, now)
        fy = self._wrist_fy.filter(wrist_y, now)

        if self._last_wrist_x is None:
            self._last_wrist_x, self._last_wrist_y = fx, fy
            return

        raw_dx = (fx - self._last_wrist_x) * self._screen_w * self.DRAG_SENSITIVITY
        dy = (fy - self._last_wrist_y) * self._screen_h * self.DRAG_SENSITIVITY
        dx = -raw_dx if self.INVERT_X else raw_dx
        self._last_wrist_x, self._last_wrist_y = fx, fy

        if dx == 0 and dy == 0:
            return

        curr_x, curr_y = pyautogui.position()
        pyautogui.moveTo(int(curr_x + dx), int(curr_y + dy), _pause=False)
