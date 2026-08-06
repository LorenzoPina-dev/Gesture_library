"""
event_bus.py
============
Livello 4: Event Bus basato su pattern Observer/Pub-Sub.

Caratteristiche:
  - Multi-Callback Support: piu' funzioni possono sottoscrivere lo stesso
    evento; vengono tutte invocate in ordine di registrazione.
  - Hysteresis & Cooldown: ogni tipo di evento ha un cooldown minimo tra
    due attivazioni consecutive (default: EventBusConfig.default_cooldown_s,
    sovrascrivibile per singolo evento), per evitare trigger multipli
    indesiderati su eventi continui (es. "pinch" mantenuto per piu' frame).
  - Disaccoppiamento totale: i livelli di riconoscimento (1/2/3) non
    conoscono i consumatori finali; pubblicano solo eventi con nome e payload.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Event:
    name: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)


class EventBus:
    def __init__(self, event_bus_config):
        self.cfg = event_bus_config
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._last_emit_time: Dict[str, float] = {}
        self._custom_cooldowns: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Subscription (Observer pattern)
    # ------------------------------------------------------------------ #
    def subscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        """Registra `callback` per l'evento `event_name`. Piu' callback per lo
        stesso evento sono supportati e vengono chiamati tutti."""
        self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[[Event], None]) -> bool:
        try:
            self._subscribers[event_name].remove(callback)
            return True
        except ValueError:
            return False

    def set_cooldown(self, event_name: str, cooldown_s: float) -> None:
        """Sovrascrive il cooldown di default per uno specifico evento."""
        self._custom_cooldowns[event_name] = cooldown_s

    # ------------------------------------------------------------------ #
    # Pubblicazione con isteresi/cooldown
    # ------------------------------------------------------------------ #
    def emit(self, event_name: str, payload: Any = None, now: Optional[float] = None) -> bool:
        """Pubblica un evento se il cooldown lo consente. Ritorna True se e'
        stato effettivamente emesso (e quindi le callback sono state invocate)."""
        now = now if now is not None else time.time()
        cooldown = self._custom_cooldowns.get(event_name, self.cfg.default_cooldown_s)
        last = self._last_emit_time.get(event_name, -1e9)

        if (now - last) < cooldown:
            return False

        self._last_emit_time[event_name] = now
        event = Event(name=event_name, payload=payload, timestamp=now)

        for callback in list(self._subscribers.get(event_name, [])):
            try:
                callback(event)
            except Exception:
                # Un errore in un singolo subscriber non deve interrompere
                # l'intera pipeline real-time.
                traceback.print_exc()
        return True

    def emit_immediate(self, event_name: str, payload: Any = None) -> None:
        """Emette ignorando il cooldown (usato tipicamente per eventi di
        sistema come 'engine_started', 'engine_stopped', 'error')."""
        event = Event(name=event_name, payload=payload, timestamp=time.time())
        self._last_emit_time[event_name] = event.timestamp
        for callback in list(self._subscribers.get(event_name, [])):
            try:
                callback(event)
            except Exception:
                traceback.print_exc()
