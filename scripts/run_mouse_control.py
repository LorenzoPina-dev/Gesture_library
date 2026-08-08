"""
scripts/run_mouse_control.py
============================
Controllo Mouse basato su Custom Gestures:
- puntatore: Muove il cursore
- aperta: Ferma il cursore (Pausa)
- pinch: Click Sinistro e Drag & Drop / Selezione testo
- middle_pinch: Click Destro
- pugno: Modalita' Rotellina Proporzionale

Rispetto alla prima versione, questo script risolve due problemi tipici del
controllo mouse "diretto frame-per-frame":

  1. Confusione pinch / middle_pinch: il classificatore k-NN (Livello 2)
     lavora su un embedding e puo' confondere le due gesture quando il pinch
     e' ancora in fase di chiusura (le due forme sono molto simili a meta'
     movimento). Qui la scelta finale tra "pinch" e "middle_pinch" viene
     sempre CONFERMATA geometricamente confrontando la distanza reale
     pollice-indice e pollice-medio (Livello 1, gia' presente nel motore),
     usando la stessa soglia 'pinch_distance_ratio' configurata
     nell'Impostazioni della UI. Se nessuna delle due distanze e' abbastanza
     chiusa, o sono ambigue (quasi uguali), la gesture non viene accettata:
     niente click "a vuoto" per un pinch non ancora chiuso del tutto.

  2. Reattivita' vs. stabilita' del cursore: invece di un semplice EMA a
     alpha fisso, il cursore usa un One Euro Filter (stesso principio usato
     dal motore per i landmark, qui applicato direttamente in coordinate
     schermo): quando la mano e' quasi ferma il cursore resta stabile
     (niente jitter), quando la mano si muove velocemente il filtro riduce
     automaticamente lo smoothing per seguire il movimento senza percepibile
     ritardo.

  3. Debounce/isteresi sulla gesture confermata: ogni gesture (inclusi
     pinch/middle_pinch/pugno/puntatore/aperta) deve rimanere stabile per
     alcuni frame consecutivi prima di essere "confermata" e generare
     un'azione. Questo elimina i click/scroll spuri causati da un singolo
     frame di misclassificazione, con una latenza aggiuntiva minima
     (~3 frame, circa 60-100 ms a 30-50 FPS) impercettibile all'uso.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine import GestureEngine
from gesture_engine.control import CursorController, GestureStabilizer


# ============================================================================
# Controller principale del mouse: stato di drag, scroll, click destro
# ============================================================================
class MouseGestureController:
    DEADZONE = 0.02
    SCROLL_GAIN = 40  # dimezzato rispetto al valore precedente (80): scroll piu' lento/controllato
    RIGHT_CLICK_COOLDOWN_S = 0.45

    def __init__(self, engine: GestureEngine):
        self._engine = engine
        self._stabilizer = GestureStabilizer(engine)
        self._cursor = CursorController()

        self.is_left_mouse_down = False
        self.is_scroll_mode = False
        self._anchor_y: Optional[float] = None
        self._last_right_click_t = 0.0
        self._prev_confirmed = "UNKNOWN"

    # -- Reset di sicurezza quando la mano esce dal frame ------------------
    def _reset_all(self) -> None:
        if self.is_left_mouse_down:
            pyautogui.mouseUp(button="left")
            self.is_left_mouse_down = False
        self.is_scroll_mode = False
        self._anchor_y = None
        self._cursor.reset()
        self._stabilizer.reset()
        self._prev_confirmed = "UNKNOWN"

    def _handle_proportional_scroll(self, curr_y: float) -> None:
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

    def handle(self, results) -> None:
        if not results:
            self._reset_all()
            return

        first_hand = results[0]
        index_x, index_y = first_hand.raw_points[8][:2]   # punta dell'indice (puntatore)
        wrist_x, wrist_y = first_hand.raw_points[0][:2]    # polso (drag/scroll)

        gesture = self._stabilizer.update(first_hand)
        gesture_just_changed = gesture != self._prev_confirmed

        # 1. ROTELLINA PROPORZIONALE (pugno) - il delta si basa sul polso:
        # durante il pugno le dita sono chiuse e la punta dell'indice non e'
        # un riferimento stabile, il polso invece si muove in modo pulito e
        # prevedibile con il gesto "su/giu'" del braccio.
        if gesture == "pugno":
            if not self.is_scroll_mode:
                self.is_scroll_mode = True
                self._anchor_y = wrist_y
                print("[STATO] Rotellina Attivata (Pugno)")
            self._handle_proportional_scroll(wrist_y)
            self._prev_confirmed = gesture
            return
        elif self.is_scroll_mode:
            self.is_scroll_mode = False
            self._anchor_y = None
            print("[STATO] Rotellina Disattivata")

        # 2. CLICK DESTRO (middle_pinch) - edge-triggered, con cooldown
        if gesture == "middle_pinch" and gesture_just_changed:
            now = time.time()
            if not self.is_left_mouse_down and (now - self._last_right_click_t) >= self.RIGHT_CLICK_COOLDOWN_S:
                pyautogui.click(button="right")
                self._last_right_click_t = now
                print("[STATO] Click Destro (Middle Pinch)")

        # 3. CLICK SINISTRO E DRAG & DROP (pinch)
        is_pinching = (gesture == "pinch")

        if is_pinching and not self.is_left_mouse_down:
            pyautogui.mouseDown(button="left")
            self.is_left_mouse_down = True
            # Ancora il tracking sul polso SENZA muovere il cursore: da qui
            # in poi solo lo spostamento del polso conta, cosi' il gesto di
            # chiusura delle dita (che sposta la punta dell'indice verso il
            # pollice) non trasforma piu' ogni click in un drag involontario.
            self._cursor.begin_relative_tracking(wrist_x, wrist_y)
            print("[STATO] Click Sinistro Premuto (Pinch)")
        elif not is_pinching and self.is_left_mouse_down:
            pyautogui.mouseUp(button="left")
            self.is_left_mouse_down = False
            print("[STATO] Click Sinistro Rilasciato")

        # 4. MOVIMENTO O FERMO DEL CURSORE
        if gesture == "puntatore":
            self._cursor.move_to(index_x, index_y)
        elif is_pinching and self.is_left_mouse_down:
            # Durante il drag il cursore segue il DELTA del polso, non la
            # posizione assoluta dell'indice (vedi move_by_wrist).
            self._cursor.move_by_wrist(wrist_x, wrist_y)
        elif gesture == "aperta":
            # Mano aperta = il cursore resta fermo dove si trova
            self._cursor.reset()

        self._prev_confirmed = gesture


def main():
    engine = GestureEngine()
    controller = MouseGestureController(engine)

    print("Controllo Mouse basato su Custom Gestures Attivo:")
    print(" - 'puntatore': Muove il cursore")
    print(" - 'aperta': Ferma il cursore (Pausa)")
    print(" - 'pinch': Click Sinistro (Tieni premuto per Evidenziare / Trascinare)")
    print(" - 'middle_pinch': Click Destro")
    print(" - 'pugno': Rotellina Proporzionale (sposta in Alto / Basso)")
    print(" - Premi 'q' o ESC sulla finestra video per uscire.")

    with engine:
        engine._running = True
        while engine._running:
            results = engine.step()
            controller.handle(results)

            frame = getattr(engine, "_last_frame_bgr", None)
            if frame is not None:
                cv2.imshow("Mouse Gesture Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if controller.is_left_mouse_down:
                    pyautogui.mouseUp(button="left")
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
