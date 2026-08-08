"""
gesture_engine/app_context/profiles.py
=========================================
Profili di gesture per-app: ogni app puo' avere un proprio set di azioni
(quante e quali gesture servono varia molto tra, es., Unity e un
visualizzatore di foto), caricati da file JSON in data/gesture_profiles/ e
selezionati automaticamente in base alla finestra in primo piano.

Schema di un profilo (vedi anche i file .json di esempio nella cartella):

    {
      "name": "Unity",
      "match_process": ["unity.exe"],        // substring, case-insensitive
      "match_title": ["Unity"],               // substring sul titolo finestra
      "primary_hand": "Right",                // mano che controlla cursore/click
      "secondary_modifiers": {                // gesture della mano SECONDARIA -> tasto modificatore tenuto premuto
        "pugno": "ctrl",
        "aperta": "alt"
      },
      "primary_actions": {                    // gesture della mano PRIMARIA -> azione
        "puntatore": {"type": "move_cursor"},
        "pinch":     {"type": "left_drag"},
        "middle_pinch": {"type": "right_click"},
        "pugno":     {"type": "scroll"}
      },
      "extra_gestures": {                     // eventi aggiuntivi del motore (swipe/sequence) -> azione
        "swipe.left":  {"type": "key", "keys": ["ctrl", "z"]},
        "swipe.right": {"type": "key", "keys": ["ctrl", "y"]}
      }
    }

Tipi di azione supportati da ActionExecutor (gesture_engine.control):
    move_cursor, left_drag, right_click, scroll, click, key, none
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .active_window import get_active_window_info

_DEFAULT_PROFILE_DICT = {
    "name": "default",
    "match_process": [],
    "match_title": [],
    "primary_hand": "Right",
    "secondary_modifiers": {                
            "pugno": "ctrl",
            "aperta": "alt"
          },
    "primary_actions": {
        "puntatore": {"type": "move_cursor"},
        "pinch": {"type": "left_drag"},
        "middle_pinch": {"type": "right_click"},
        "pugno": {"type": "scroll"},
    },
    "extra_gestures": {                    
        "swipe.left":  {"type": "key", "keys": ["ctrl", "z"]},
        "swipe.right": {"type": "key", "keys": ["ctrl", "y"]}
    }
}


@dataclass
class GestureProfile:
    name: str
    match_process: List[str] = field(default_factory=list)
    match_title: List[str] = field(default_factory=list)
    primary_hand: str = "Right"
    secondary_modifiers: Dict[str, str] = field(default_factory=dict)
    primary_actions: Dict[str, dict] = field(default_factory=dict)
    extra_gestures: Dict[str, dict] = field(default_factory=dict)

    @property
    def secondary_hand(self) -> str:
        return "Left" if self.primary_hand == "Right" else "Right"

    @classmethod
    def from_dict(cls, d: dict) -> "GestureProfile":
        return cls(
            name=d.get("name", "unnamed"),
            match_process=[s.lower() for s in d.get("match_process", [])],
            match_title=[s.lower() for s in d.get("match_title", [])],
            primary_hand=d.get("primary_hand", "Right"),
            secondary_modifiers=d.get("secondary_modifiers", {}),
            primary_actions=d.get("primary_actions", {}),
            extra_gestures=d.get("extra_gestures", {}),
        )

    def matches(self, process_name: str, window_title: str) -> bool:
        process_name = process_name.lower()
        window_title = window_title.lower()
        if any(m in process_name for m in self.match_process):
            return True
        if any(m in window_title for m in self.match_title):
            return True
        return False


class ProfileManager:
    """Carica tutti i profili da una cartella e seleziona quello attivo in
    base alla finestra in primo piano, con polling limitato (POLL_INTERVAL_S)
    per non interrogare il sistema operativo ad ogni frame."""

    POLL_INTERVAL_S = 0.5

    def __init__(self, profiles_dir: str):
        self.profiles_dir = profiles_dir
        self.default_profile = GestureProfile.from_dict(_DEFAULT_PROFILE_DICT)
        self.profiles: List[GestureProfile] = []
        self._current: GestureProfile = self.default_profile
        self._last_poll_t = 0.0
        self._last_logged_info: Optional[Tuple[str, str]] = None

    def load(self) -> None:
        self.profiles = []
        if not os.path.isdir(self.profiles_dir):
            return
        for path in sorted(glob.glob(os.path.join(self.profiles_dir, "*.json"))):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = GestureProfile.from_dict(data)
                if profile.name.lower() == "default":
                    self.default_profile = profile
                else:
                    self.profiles.append(profile)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[ProfileManager] Impossibile caricare '{path}': {exc}")

    def get_active_profile(self, force: bool = False) -> GestureProfile:
        """Ritorna il profilo che corrisponde alla finestra in primo piano
        (throttled a POLL_INTERVAL_S). Se nessun profilo corrisponde, ritorna
        il profilo di default."""
        now = time.time()
        if not force and (now - self._last_poll_t) < self.POLL_INTERVAL_S:
            return self._current
        self._last_poll_t = now

        info = get_active_window_info()
        if info is None:
            # Backend non disponibile (pywin32/psutil mancanti) O finestra non
            # risolvibile: stampa il motivo UNA VOLTA (non ad ogni poll) cosi'
            # non si confonde con un vero mismatch di match_process/match_title.
            if self._last_logged_info is not None:
                from .active_window import backend_available
                reason = (
                    "pywin32/psutil non installati"
                    if not backend_available()
                    else "finestra in primo piano non risolvibile"
                )
                print(f"[ProfileManager] Nessuna info finestra ({reason}) -> uso 'default'.")
            self._last_logged_info = None
            self._current = self.default_profile
            return self._current

        process_name, window_title = info
        # Logga SOLO quando la finestra rilevata cambia (non ad ogni poll,
        # altrimenti con POLL_INTERVAL_S=0.5 sarebbero 2 righe/secondo anche
        # restando sulla stessa finestra): questo mostra sempre, la prima
        # volta che porti Unity in primo piano, esattamente cosa il sistema
        # ha visto - utile per capire perche' NON scatta il profilo giusto.
        if info != self._last_logged_info:
            self._last_logged_info = info
            matched = next((p for p in self.profiles if p.matches(process_name, window_title)), None)
            if matched is not None:
                print(
                    f"[ProfileManager] Finestra: processo='{process_name}' titolo='{window_title}' "
                    f"-> profilo '{matched.name}' combacia."
                )
            else:
                print(
                    f"[ProfileManager] Finestra: processo='{process_name}' titolo='{window_title}' "
                    f"-> NESSUN profilo combacia (uso 'default'). Se questa e' la finestra di Unity, "
                    f"aggiungi il testo sopra (processo o titolo) a 'match_process'/'match_title' in "
                    f"data/gesture_profiles/unity.json."
                )

        for profile in self.profiles:
            if profile.matches(process_name, window_title):
                self._current = profile
                return self._current

        self._current = self.default_profile
        return self._current
