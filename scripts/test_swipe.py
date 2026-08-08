"""
scripts/test_swipe.py
======================
Script di debug per testare la sensibilità e il funzionamento degli swipe.
"""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine import GestureEngine


def main():
    engine = GestureEngine()

    # Per il test abbassiamo/impostiamo il cooldown sull'EventBus
    engine.event_bus.set_cooldown("swipe.up", 0.15)
    engine.event_bus.set_cooldown("swipe.down", 0.15)

    # Callback di debug
    engine.on("swipe.up", lambda e: print(f" [SWIPE UP DETECTED!] Velocità: {e.payload['velocity']:.2f}"))
    engine.on("swipe.down", lambda e: print(f" [SWIPE DOWN DETECTED!] Velocità: {e.payload['velocity']:.2f}"))
    engine.on("swipe.left", lambda e: print(f" [SWIPE LEFT DETECTED!] Velocità: {e.payload['velocity']:.2f}"))
    engine.on("swipe.right", lambda e: print(f" [SWIPE RIGHT DETECTED!] Velocità: {e.payload['velocity']:.2f}"))

    print("Test Swipe Attivo. Fai scatti rapidi con la mano in alto/basso/destra/sinistra.")
    print("Premi 'q' o ESC per uscire.")

    with engine:
        engine._running = True
        while engine._running:
            engine.step()

            frame = getattr(engine, "_last_frame_bgr", None)
            if frame is not None:
                cv2.imshow("Test Swipe", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()