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

import math
import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine import GestureEngine
from gesture_engine.normalization import pinch_distance


# ============================================================================
# One Euro Filter: smoothing adattivo alla velocita', per un cursore
# stabile da fermo e reattivo in movimento (vedi gesture_engine.normalization
# .filters per la stessa tecnica applicata ai landmark grezzi).
# ============================================================================
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
    """Muove il mouse in modo fluido e reattivo tramite One Euro Filter
    applicato direttamente in coordinate schermo (px)."""

    # min_cutoff basso = molta stabilita' da fermo (niente jitter);
    # beta alto = il filtro "si apre" velocemente sui movimenti rapidi,
    # riducendo il ritardo percepito quasi a zero.
    MIN_CUTOFF = 0.9
    BETA = 1.2
    D_CUTOFF = 1.0

    # Moltiplicatore applicato al delta del polso durante il drag: un valore
    # > 1 rende il drag piu' sensibile (basta muovere la mano di meno per
    # spostare il cursore della stessa distanza sullo schermo).
    DRAG_SENSITIVITY = 1.8

    def __init__(self):
        self._fx = _OneEuroScalar(self.MIN_CUTOFF, self.BETA, self.D_CUTOFF)
        self._fy = _OneEuroScalar(self.MIN_CUTOFF, self.BETA, self.D_CUTOFF)
        # Filtri separati per il tracking relativo del polso (drag/scroll):
        # devono restare indipendenti da quelli del puntatore assoluto,
        # altrimenti passare da un modo all'altro causerebbe un salto.
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
        """Movimento ASSOLUTO (usato dal puntatore): mappa la posizione
        dell'indice 1:1 sullo schermo, con smoothing One Euro.
        raw_x/raw_y in [0,1] (coordinate immagine normalizzate)."""
        now = time.time()
        target_x = raw_x * self._screen_w
        target_y = raw_y * self._screen_h

        smooth_x = self._fx.filter(target_x, now)
        smooth_y = self._fy.filter(target_y, now)

        pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)

    def begin_relative_tracking(self, wrist_x: float, wrist_y: float) -> None:
        """Da chiamare all'inizio di un drag/scroll: fissa il riferimento del
        polso SENZA spostare il cursore, cosi' il primo frame non causa
        nessun salto (il cursore resta esattamente dov'era al click)."""
        now = time.time()
        self._wrist_fx.reset()
        self._wrist_fy.reset()
        self._last_wrist_x = self._wrist_fx.filter(wrist_x, now)
        self._last_wrist_y = self._wrist_fy.filter(wrist_y, now)

    def move_by_wrist(self, wrist_x: float, wrist_y: float) -> None:
        """Movimento RELATIVO (usato durante drag col pinch): il cursore si
        sposta del DELTA del polso rispetto al frame precedente, invece di
        seguire la posizione assoluta della punta del dito. Cosi' chiudere
        le dita per fare il pinch (che sposta la punta dell'indice verso il
        pollice) non genera piu' movimento del cursore: solo un vero
        spostamento della mano (polso) muove/trascina."""
        now = time.time()
        fx = self._wrist_fx.filter(wrist_x, now)
        fy = self._wrist_fy.filter(wrist_y, now)

        if self._last_wrist_x is None:
            self._last_wrist_x, self._last_wrist_y = fx, fy
            return

        dx = (fx - self._last_wrist_x) * self._screen_w * self.DRAG_SENSITIVITY
        dy = (fy - self._last_wrist_y) * self._screen_h * self.DRAG_SENSITIVITY
        self._last_wrist_x, self._last_wrist_y = fx, fy

        if dx == 0 and dy == 0:
            return

        curr_x, curr_y = pyautogui.position()
        pyautogui.moveTo(int(curr_x + dx), int(curr_y + dy), _pause=False)


# ============================================================================
# Risoluzione gesture: combina k-NN (Livello 2) + fallback rule-based
# (Livello 1) + conferma geometrica esplicita per pinch/middle_pinch +
# debounce temporale, cosi' da ottenere un'unica "gesture confermata" stabile
# su cui basare le azioni del mouse.
# ============================================================================
class GestureStabilizer:
    MIN_KNN_CONFIDENCE = 0.55       # sotto questa confidenza non ci si fida del k-NN
    STABILITY_FRAMES = 3            # frame consecutivi uguali richiesti per confermare un cambio
    PINCH_AMBIGUITY_MARGIN = 0.85   # una distanza deve essere < 85% dell'altra per non essere ambigua

    def __init__(self, engine: GestureEngine):
        self._engine = engine
        self._history: deque[str] = deque(maxlen=self.STABILITY_FRAMES)
        self.confirmed_gesture: str = "UNKNOWN"

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
            # Entrambe le dita sono abbastanza vicine al pollice: serve un
            # margine netto per decidere, altrimenti e' una posizione
            # ambigua (es. transizione) e non va accettata come nessuna
            # delle due, per evitare di sfarfallare tra pinch/middle_pinch.
            if d_index <= d_middle * self.PINCH_AMBIGUITY_MARGIN:
                return "pinch"
            if d_middle <= d_index * self.PINCH_AMBIGUITY_MARGIN:
                return "middle_pinch"
            return None
        return None  # nessuna delle due dita e' sufficientemente chiusa

    def _resolve_raw_label(self, hand) -> str:
        label = hand.knn_label
        if label == "UNKNOWN" or hand.knn_confidence < self.MIN_KNN_CONFIDENCE:
            label = hand.rule_label

        if label in ("pinch", "middle_pinch"):
            geo_label = self._classify_pinch_family(hand.normalized_points)
            if geo_label is not None:
                label = geo_label
            else:
                # Pinch non ancora chiuso del tutto o ambiguo: non e' un
                # click valido. Ricadi sulla gesture di base (indice teso =
                # puntatore, altrimenti nessuna azione).
                label = "puntatore" if hand.rule_label == "1_fingers" else hand.rule_label

        return label

    def update(self, hand) -> str:
        raw_label = self._resolve_raw_label(hand)
        self._history.append(raw_label)

        # Conferma il cambio di stato solo se le ultime N letture concordano:
        # elimina i falsi trigger di 1 frame senza introdurre lag percepibile.
        if len(self._history) == self._history.maxlen and len(set(self._history)) == 1:
            self.confirmed_gesture = raw_label

        return self.confirmed_gesture


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
