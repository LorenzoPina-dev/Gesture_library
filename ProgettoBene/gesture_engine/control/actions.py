"""
gesture_engine/control/actions.py
====================================
Esecutore di azioni data-driven: prende un dizionario "azione" (definito nei
profili JSON di gesture_engine.app_context) e lo traduce in chiamate
pyautogui reali. Cosi' lo script di controllo non ha bisogno di un
if/elif per ogni gesture: aggiungere/rimappare gesture -> azione e' solo
questione di modificare un file JSON, non il codice.

Tipi di azione supportati (campo "type" nel dizionario azione):
    move_cursor   -> muove il cursore in modo assoluto (usa la punta dell'indice)
    left_drag     -> tiene premuto il tasto sinistro e trascina seguendo il polso
    right_drag    -> tiene premuto il tasto destro e trascina seguendo il polso
    middle_drag   -> tiene premuto il tasto centrale e trascina seguendo il polso (es. pan vista)
    right_click   -> click destro singolo, edge-triggered (un click per attivazione gesture)
    click         -> click singolo del bottone indicato in "button" (default "left")
    scroll        -> rotellina proporzionale allo spostamento del polso
    key           -> combinazione di tasti (campo "keys": lista, es. ["ctrl","z"])
    none / null   -> nessuna azione (ferma/riposa il cursore)

ModifierController gestisce invece i tasti modificatore (ctrl/alt/shift)
tenuti premuti in base alla gesture della mano SECONDARIA: se il modificatore
e' attivo mentre l'azione "scroll" viene eseguita, il sistema operativo vede
esattamente un CTRL+ROTELLINA / ALT+ROTELLINA nativo, senza bisogno di
gestirlo esplicitamente qui.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import pyautogui

from .filters import CursorController

_MODIFIER_KEY_NAMES = {"ctrl": "ctrl", "alt": "alt", "shift": "shift", "win": "win"}


class ModifierController:
    """Tiene premuto (pyautogui.keyDown) al massimo un tasto modificatore alla
    volta, corrispondente alla gesture corrente della mano secondaria."""

    def __init__(self):
        self._held: Optional[str] = None

    @property
    def held(self) -> Optional[str]:
        return self._held

    def update(self, desired: Optional[str]) -> None:
        if desired == self._held:
            return
        if self._held is not None:
            pyautogui.keyUp(_MODIFIER_KEY_NAMES.get(self._held, self._held))
        if desired is not None:
            pyautogui.keyDown(_MODIFIER_KEY_NAMES.get(desired, desired))
        print(f"[MODIFICATORE] {self._held or '(nessuno)'} -> {desired or '(nessuno)'}")
        self._held = desired

    def release_all(self) -> None:
        self.update(None)


class ActionExecutor:
    """Esegue le azioni della mano PRIMARIA (cursore/click/scroll/drag) e
    delle extra_gestures (combinazioni di tasti su eventi swipe/sequence)."""

    RIGHT_CLICK_COOLDOWN_S = 0.45
    SCROLL_GAIN = 40
    DEADZONE = 0.02

    # Mappa tipo azione di drag -> bottone pyautogui reale.
    _DRAG_BUTTONS = {"left_drag": "left", "right_drag": "right", "middle_drag": "middle"}
    _DRAG_LABELS = {"left_drag": "sinistro", "right_drag": "destro", "middle_drag": "centrale"}

    def __init__(self):
        self.cursor = CursorController()
        # Bottone del mouse attualmente tenuto premuto per un drag in corso
        # ("left"/"right"/"middle"), oppure None se nessun drag e' attivo.
        # Puo' essere premuto UN SOLO bottone alla volta (una gesture primaria
        # per volta), quindi basta un singolo stato invece di un bool per
        # bottone.
        self._drag_button: Optional[str] = None
        self.is_scroll_mode = False
        self._anchor_y: Optional[float] = None
        self._last_right_click_t = 0.0
        self._last_click_t: Dict[str, float] = {}

    def reset(self) -> None:
        """Da chiamare quando la mano primaria esce dal frame: rilascia ogni
        stato 'tenuto premuto' per sicurezza."""
        self._release_drag()
        self.is_scroll_mode = False
        self._anchor_y = None
        self.cursor.reset()

    def _release_drag(self) -> None:
        if self._drag_button is not None:
            pyautogui.mouseUp(button=self._drag_button)
            label = self._DRAG_LABELS.get(self._drag_button, self._drag_button)
            print(f"[AZIONE] Drag {label}: rilasciato")
            self._drag_button = None

    def _proportional_scroll(self, curr_y: float) -> None:
        if self._anchor_y is None:
            self._anchor_y = curr_y
            return
        offset = curr_y - self._anchor_y
        if abs(offset) > self.DEADZONE:
            distance = offset - (self.DEADZONE if offset > 0 else -self.DEADZONE)
            scroll_speed = int(-distance * self.SCROLL_GAIN * abs(distance) * 1000)
            scroll_speed = max(min(scroll_speed, 1200), -1200)
            if scroll_speed != 0:
                pyautogui.scroll(scroll_speed)

    def run_primary(
        self,
        action: Optional[dict],
        gesture_just_changed: bool,
        index_xy: Tuple[float, float],
        wrist_xy: Tuple[float, float],
    ) -> None:
        action_type = (action or {}).get("type", "none")
        index_x, index_y = index_xy
        wrist_x, wrist_y = wrist_xy

        # Chiudi stati "tenuti premuti" incompatibili con l'azione corrente,
        # cosi' cambiare gesture rilascia sempre in modo pulito. Questo
        # scatta anche appena si LASCIA una gesture di drag (l'azione non e'
        # piu' quel tipo di drag -> action_type cambia -> mouseUp), quindi
        # non serve nessun click a parte: rilasciare la gesture non genera
        # mai un click, solo il rilascio del bottone tenuto premuto.
        if action_type != "scroll" and self.is_scroll_mode:
            self.is_scroll_mode = False
            self._anchor_y = None
            print("[AZIONE] Rotellina: disattivata")
        current_drag_type = next((t for t, b in self._DRAG_BUTTONS.items() if b == self._drag_button), None)
        if self._drag_button is not None and action_type != current_drag_type:
            self._release_drag()

        if action_type == "move_cursor":
            self.cursor.move_to(index_x, index_y)

        elif action_type in self._DRAG_BUTTONS:
            button = self._DRAG_BUTTONS[action_type]
            label = self._DRAG_LABELS[action_type]
            if self._drag_button != button:
                pyautogui.mouseDown(button=button)
                self._drag_button = button
                self.cursor.begin_relative_tracking(wrist_x, wrist_y)
                print(f"[AZIONE] Drag {label}: premuto")
            else:
                self.cursor.move_by_wrist(wrist_x, wrist_y)

        elif action_type == "right_click":
            if gesture_just_changed:
                now = time.time()
                if (now - self._last_right_click_t) >= self.RIGHT_CLICK_COOLDOWN_S:
                    pyautogui.click(button="right")
                    self._last_right_click_t = now
                    print("[AZIONE] Click destro")

        elif action_type == "click":
            if gesture_just_changed:
                button = (action or {}).get("button", "left")
                pyautogui.click(button=button)
                print(f"[AZIONE] Click {button}")

        elif action_type == "scroll":
            if not self.is_scroll_mode:
                self.is_scroll_mode = True
                self._anchor_y = wrist_y
                print("[AZIONE] Rotellina: attivata")
            else:
                self._proportional_scroll(wrist_y)

        elif action_type == "key":
            if gesture_just_changed:
                keys = (action or {}).get("keys", [])
                print(f"[AZIONE] Combinazione tasti: {'+'.join(keys)}")
                self._press_combo(keys)

        else:  # "none" o azione non riconosciuta: nessun input, cursore a riposo
            self.cursor.reset()

    def run_event_action(self, action: dict, cooldown_key: str, cooldown_s: float = 0.4) -> None:
        """Esegue un'azione collegata a un evento discreto del motore (swipe,
        sequence). Ha un proprio cooldown per evitare ripetizioni accidentali."""
        now = time.time()
        last = self._last_click_t.get(cooldown_key, 0.0)
        if (now - last) < cooldown_s:
            return
        self._last_click_t[cooldown_key] = now

        action_type = action.get("type", "none")
        if action_type == "key":
            self._press_combo(action.get("keys", []))
        elif action_type == "click":
            pyautogui.click(button=action.get("button", "left"))
        elif action_type == "right_click":
            pyautogui.click(button="right")
        # "scroll"/"move_cursor"/"left_drag" non hanno senso su un evento
        # discreto: vengono ignorati se specificati per errore in un profilo.

    @staticmethod
    def _press_combo(keys: Sequence[str]) -> None:
        keys = [k for k in keys if k]
        if not keys:
            return
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
