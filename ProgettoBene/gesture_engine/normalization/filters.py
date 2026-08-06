"""
filters.py
==========
Filtri temporali anti-sfarfallio (anti-jitter) applicati ai landmark grezzi
PRIMA della normalizzazione geometrica, cosi' da eliminare micro-salti dovuti
a rumore del sensore o occlusioni parziali senza introdurre troppa latenza.

Implementa due strategie selezionabili da config (FilterConfig.method):
  - "ema":       Exponential Moving Average, semplice e stabile.
  - "one_euro":  One Euro Filter (Casiez et al. 2012), si adatta alla velocita'
                 del movimento: molto smoothing quando fermo, poco smoothing
                 (bassa latenza) quando il movimento e' rapido.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np


class EMAFilter:
    """Filtro esponenziale semplice su un array (N, 3) di landmark."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def filter(self, points: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        # timestamp ignorato: l'EMA non e' adattivo alla velocita', accettato
        # solo per uniformita' di interfaccia con OneEuroFilter.
        if self._state is None:
            self._state = points.copy()
        else:
            self._state = self.alpha * points + (1.0 - self.alpha) * self._state
        return self._state.copy()


class _OneEuroScalar:
    """Implementazione base del One Euro Filter per un singolo scalare."""

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

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(t - self.t_prev, 1e-6)

        # Stima derivata (velocita') filtrata
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Cutoff adattivo: piu' veloce il movimento, meno smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilter:
    """
    One Euro Filter applicato indipendentemente a ciascuna delle 63 componenti
    (21 landmark x 3 assi) di un frame di landmark grezzi.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters: Optional[list] = None

    def reset(self) -> None:
        self._filters = None

    def filter(self, points: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        t = timestamp if timestamp is not None else time.time()
        flat = points.reshape(-1)

        if self._filters is None:
            self._filters = [
                _OneEuroScalar(self.min_cutoff, self.beta, self.d_cutoff)
                for _ in range(flat.shape[0])
            ]

        out = np.empty_like(flat)
        for i, v in enumerate(flat):
            out[i] = self._filters[i].filter(float(v), t)
        return out.reshape(points.shape)


def build_filter(filter_config):
    """Factory: crea il filtro configurato in FilterConfig."""
    if filter_config.method == "ema":
        return EMAFilter(alpha=filter_config.ema_alpha)
    elif filter_config.method == "one_euro":
        return OneEuroFilter(
            min_cutoff=filter_config.one_euro_min_cutoff,
            beta=filter_config.one_euro_beta,
            d_cutoff=filter_config.one_euro_d_cutoff,
        )
    else:
        raise ValueError(f"Metodo di filtro sconosciuto: {filter_config.method}")
