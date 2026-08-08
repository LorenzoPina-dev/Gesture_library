"""
scripts/run_dual_hand_control.py
==================================
Controllo Mouse Dual-Hand con profili gesture per-app.

Idea di fondo:
  - la mano PRIMARIA (default: destra) fa quello che gia' fa
    run_mouse_control.py: muove il cursore, click sinistro/drag, click
    destro, rotellina proporzionale;
  - la mano SECONDARIA (default: sinistra) NON tocca il mouse: la sua
    gesture confermata seleziona un tasto modificatore (Ctrl/Alt/Shift) che
    resta PREMUTO finche' la gesture resta quella. Se in quel momento la
    mano primaria sta scrollando, il sistema operativo vede esattamente un
    CTRL+ROTELLINA o ALT+ROTELLINA nativo (zoom, cambio strumento, ecc. a
    seconda dell'app) - nessuna logica speciale necessaria, e' solo un vero
    tasto tenuto premuto mentre arriva un vero evento di scroll.
  - quale gesture fa cosa NON e' hardcoded: viene letto da un "profilo"
    JSON (vedi data/gesture_profiles/*.json) selezionato automaticamente in
    base alla finestra applicativa in primo piano (Unity, un
    visualizzatore immagini, ...). Un'app come Unity puo' avere un profilo
    con molte piu' gesture/scorciatoie di una semplice galleria immagini,
    senza toccare il codice - solo il file JSON.

Se 'pywin32'/'psutil' non sono installati, il rilevamento automatico della
finestra attiva e' disattivato e viene sempre usato il profilo 'default'
(nessun errore, solo niente cambio automatico di profilo). Installa con:
    pip install pywin32 psutil

Uso:
    python scripts/run_dual_hand_control.py

Per creare un nuovo profilo: copia uno dei file in data/gesture_profiles/,
rinominalo, cambia "match_process"/"match_title" per la tua app, e modifica
"primary_actions"/"secondary_modifiers"/"extra_gestures" a piacere - vedi il
docstring di gesture_engine/app_context/profiles.py per lo schema completo.

LOG IN CONSOLE (per debug, es. "il pinch mi viene letto come pugno")
----------------------------------------------------------------------
Ogni ~0.25s viene stampata una riga [STATO] con, per ciascuna mano:
    knn=<etichetta>(<confidenza>)   -> cosa dice il classificatore k-NN
    rule=<etichetta>                -> cosa dice il rule-based (Livello 1)
    geo=<pinch|middle_pinch|None>   -> cosa dice il controllo geometrico
                                        pollice-indice/medio (ha sempre
                                        l'ultima parola sul pinch, tranne
                                        quando rule=fist)
    -> confermata=<etichetta>       -> il risultato finale dopo il debounce
                                        a 3 frame, quello che guida le azioni
In piu' [AZIONE]/[MODIFICATORE] loggano ogni click/drag/scroll/tasto e ogni
cambio di tasto modificatore tenuto premuto, e [PROFILO] logga ogni cambio
di app rilevata.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List, Optional

import cv2
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine import GestureEngine
from gesture_engine.control import GestureStabilizer, ActionExecutor, ModifierController
from gesture_engine.app_context import ProfileManager, GestureProfile, backend_available

# Ogni quanti secondi stampare lo stato diagnostico completo (profilo attivo +
# cosa vede knn/rule-based/geometria/confermata per ciascuna mano). Un valore
# troppo basso rende la console illeggibile: 4 righe al secondo sono un buon
# compromesso per seguire a occhio cosa succede senza sommergere il terminale.
DEBUG_LOG_INTERVAL_S = 0.25


class DualHandGestureController:
    def __init__(self, engine: GestureEngine, profiles_dir: str):
        self._engine = engine
        self._stabilizers: Dict[str, GestureStabilizer] = {
            "Left": GestureStabilizer(engine),
            "Right": GestureStabilizer(engine),
        }
        self._executor = ActionExecutor()
        self._modifiers = ModifierController()

        self._profile_manager = ProfileManager(profiles_dir)
        self._profile_manager.load()
        self._current_profile: Optional[GestureProfile] = None
        self._extra_subs: Dict[str, Callable] = {}
        self._prev_primary_gesture = "UNKNOWN"
        self._last_debug_print_t = 0.0

        self._switch_profile(self._profile_manager.get_active_profile(force=True))

    # -- Cambio di profilo (app in primo piano cambiata) --------------------
    def _switch_profile(self, profile: GestureProfile) -> None:
        if profile is self._current_profile:
            return

        # Rimuove le sottoscrizioni evento del profilo precedente
        for event_name, callback in self._extra_subs.items():
            self._engine.event_bus.unsubscribe(event_name, callback)
        self._extra_subs.clear()

        # Sicurezza: rilascia mouse/tasti/filtri quando cambia il contesto,
        # cosi' non resta nulla "premuto" passando da un'app all'altra.
        self._executor.reset()
        self._modifiers.release_all()
        for stabilizer in self._stabilizers.values():
            stabilizer.reset()

        # Registra le extra_gestures (swipe/sequence -> azione) del nuovo profilo
        for event_name, action in profile.extra_gestures.items():
            def _callback(_event, _action=action, _key=event_name):
                self._executor.run_event_action(_action, cooldown_key=_key)

            self._engine.event_bus.subscribe(event_name, _callback)
            self._extra_subs[event_name] = _callback

        self._current_profile = profile
        print(f"[PROFILO] Attivo: '{profile.name}' (mano primaria: {profile.primary_hand})")

    # -- Elaborazione di un frame --------------------------------------------
    def handle(self, results: List) -> None:
        profile = self._profile_manager.get_active_profile()
        if profile is not self._current_profile:
            self._switch_profile(profile)

        by_hand: Dict[str, Optional[object]] = {"Left": None, "Right": None}
        for r in results:
            if r.handedness in by_hand and by_hand[r.handedness] is None:
                by_hand[r.handedness] = r

        primary_hand_result = by_hand.get(profile.primary_hand)
        secondary_hand_result = by_hand.get(profile.secondary_hand)

        # --- Mano secondaria: solo tasto modificatore, nessuna azione diretta ---
        if secondary_hand_result is not None:
            secondary_gesture = self._stabilizers[profile.secondary_hand].update(secondary_hand_result)
        else:
            self._stabilizers[profile.secondary_hand].reset()
            secondary_gesture = "UNKNOWN"
        self._modifiers.update(profile.secondary_modifiers.get(secondary_gesture))

        # --- Mano primaria: cursore / click / drag / scroll ---
        if primary_hand_result is None:
            self._executor.reset()
            self._stabilizers[profile.primary_hand].reset()
            self._prev_primary_gesture = "UNKNOWN"
            self._log_debug_status(profile, None, secondary_hand_result)
            return

        gesture = self._stabilizers[profile.primary_hand].update(primary_hand_result)
        gesture_just_changed = gesture != self._prev_primary_gesture

        index_xy = tuple(primary_hand_result.raw_points[8][:2])
        wrist_xy = tuple(primary_hand_result.raw_points[0][:2])

        action = profile.primary_actions.get(gesture)
        self._executor.run_primary(action, gesture_just_changed, index_xy, wrist_xy)

        self._prev_primary_gesture = gesture

        self._log_debug_status(profile, primary_hand_result, secondary_hand_result)

    # -- Log diagnostico periodico: app attiva + cosa riconosce ciascuna mano --
    def _log_debug_status(self, profile: GestureProfile, primary_hand_result, secondary_hand_result) -> None:
        now = time.time()
        if (now - self._last_debug_print_t) < DEBUG_LOG_INTERVAL_S:
            return
        self._last_debug_print_t = now

        def _hand_diag(side: str, hand_result) -> str:
            if hand_result is None:
                return f"{side}: (non rilevata)"
            st = self._stabilizers[side]
            return (
                f"{side}: knn={st.last_knn_label}({st.last_knn_confidence:.2f}) "
                f"rule={st.last_rule_label} geo={st.last_geo_label} "
                f"-> confermata={st.confirmed_gesture}"
            )

        primary_side = profile.primary_hand
        secondary_side = profile.secondary_hand
        print(
            f"[STATO] profilo={profile.name} | "
            f"{_hand_diag(primary_side, primary_hand_result)} (PRIMARIA) | "
            f"{_hand_diag(secondary_side, secondary_hand_result)} (SECONDARIA) | "
            f"modificatore={self._modifiers.held or '(nessuno)'}"
        )

    def shutdown(self) -> None:
        self._executor.reset()
        self._modifiers.release_all()


def main():
    engine = GestureEngine()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profiles_dir = os.path.join(project_root, "data", "gesture_profiles")

    controller = DualHandGestureController(engine, profiles_dir)

    print("Controllo Dual-Hand Attivo:")
    print(" - Mano PRIMARIA: cursore / click sinistro-drag / click destro / rotellina")
    print(" - Mano SECONDARIA: gesture -> tasto modificatore (Ctrl/Alt/Shift) tenuto premuto")
    print(" - Il set di gesture/azioni dipende dall'app in primo piano (vedi data/gesture_profiles/)")
    if not backend_available():
        print(" NOTA: 'pywin32'/'psutil' non installati -> sempre profilo 'default'.")
        print("       Installa con: pip install pywin32 psutil")
    print(" - Premi 'q' o ESC sulla finestra video per uscire.")

    with engine:
        engine._running = True
        while engine._running:
            results = engine.step()
            controller.handle(results)

            frame = getattr(engine, "_last_frame_bgr", None)
            if frame is not None:
                cv2.imshow("Dual-Hand Gesture Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        controller.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
