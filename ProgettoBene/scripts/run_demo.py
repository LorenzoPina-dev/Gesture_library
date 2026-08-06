"""
scripts/run_demo.py
====================
Demo rapida da linea di comando: apre la webcam, mostra la finestra di
preview con overlay di debug ed esempi di sottoscrizione ad eventi.

Uso:
    python scripts/run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine import GestureEngine


def main():
    engine = GestureEngine()

    # Esempi di sottoscrizione multi-callback allo stesso evento
    engine.on("gesture.fist", lambda e: print(f"[EVENTO] Pugno rilevato su mano {e.payload['hand_index']}"))
    engine.on("gesture.open_palm", lambda e: print("[EVENTO] Palmo aperto"))
    engine.on("gesture.pinch", lambda e: print(f"[EVENTO] Pinch (forza={e.payload['pinch_strength']:.2f})"))
    engine.on("swipe.left", lambda e: print("[EVENTO] Swipe verso sinistra!"))
    engine.on("swipe.right", lambda e: print("[EVENTO] Swipe verso destra!"))

    # Esempio di gesture sequenziale di Livello 3
    engine.register_sequence("boom_explosion", ["fist", "open_palm"])
    engine.on("sequence.boom_explosion", lambda e: print("[EVENTO] BOOM! Sequenza pugno->palmo completata"))

    print("Avvio del Gesture Control Engine. Premi 'q' o ESC nella finestra video per uscire.")
    with engine:
        engine.run()


if __name__ == "__main__":
    main()
