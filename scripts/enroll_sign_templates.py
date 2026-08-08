"""
scripts/enroll_sign_templates.py
=================================
Registra dal vivo (webcam) i template di segni usati da
run_sign_language_translate.py per il riconoscimento.

A differenza dell'enrollment delle custom gesture (Livello 2, un singolo
embedding statico per campione, vedi ui/app.py tab "Enrollment"), qui ogni
campione e' l'INTERA sequenza temporale di un segno (forma delle mani nel
tempo + traiettoria dei polsi), perche' la maggior parte dei segni reali e'
dinamica e non riconoscibile da una singola posa.

Uso:
    python scripts/enroll_sign_templates.py

Controlli a schermo:
    SPAZIO  inizia / termina la registrazione della ripetizione corrente
    n       passa a una nuova etichetta (chiede il nome da terminale)
    q/ESC   salva ed esce

Consiglio: registra 3-5 ripetizioni per ogni segno, variando leggermente
velocita' e posizione in modo che il matching DTW sia piu' robusto.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.ingestion import CameraStream, HandLandmarkerEngine
from gesture_engine.visualization import draw_hand, landmarks_to_pixel_coords
from gesture_engine.sign_language import (
    build_frame_feature,
    split_detections_by_hand,
    SignTemplateStore,
)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sign_templates", "templates_db.npz"
)
MIN_SAMPLE_FRAMES = 5  # sotto questa lunghezza una ripetizione viene scartata (probabile tocco accidentale)


def parse_args():
    p = argparse.ArgumentParser(description="Enrollment di template per il riconoscimento della lingua dei segni.")
    p.add_argument("--database", default=DEFAULT_DB_PATH, help="Percorso del database dei template (.npz)")
    p.add_argument("--camera", type=int, default=0, help="Indice della webcam")
    p.add_argument("--reps", type=int, default=3, help="Numero di ripetizioni suggerite per segno")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = EngineConfig.load()
    cfg.camera.device_index = args.camera
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    store = SignTemplateStore(args.database)
    store.load()

    print("Enrollment segni - database:", args.database)
    if store.templates:
        print("Etichette gia' presenti:", store.known_labels())

    with CameraStream(cfg.camera) as camera, HandLandmarkerEngine(cfg.landmarker, project_root) as landmarker:
        while True:
            label = input("\nNome del segno da registrare (invio vuoto per uscire): ").strip()
            if not label:
                break

            rep_count = 0
            print(f"Segno '{label}': premi SPAZIO per iniziare/terminare ogni ripetizione "
                  f"({args.reps} consigliate). 'n' per passare al segno successivo, 'q' per uscire e salvare.")

            recording = False
            buffer = []
            quit_all = False

            while True:
                ok, frame, ts_ms = camera.read()
                if not ok:
                    continue

                detections = landmarker.detect(frame, ts_ms)
                by_hand = split_detections_by_hand(detections)

                for det in detections:
                    pts_px = landmarks_to_pixel_coords(det.landmarks[:, :2], frame.shape)
                    draw_hand(frame, pts_px, cfg.visualization)

                if recording:
                    feat = build_frame_feature(by_hand)
                    buffer.append(feat)
                    status = f"REC '{label}' - frame {len(buffer)}"
                    color = (0, 0, 255)
                else:
                    status = f"'{label}': {rep_count}/{args.reps} ripetizioni - SPAZIO per registrare"
                    color = (0, 200, 0)

                cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, "SPAZIO=rec  n=nuovo segno  q=esci", (12, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("Enrollment Lingua dei Segni", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if not recording:
                        recording = True
                        buffer = []
                    else:
                        recording = False
                        if len(buffer) >= MIN_SAMPLE_FRAMES:
                            store.add(label, np.stack(buffer, axis=0))
                            rep_count += 1
                            print(f"  Ripetizione {rep_count} salvata ({len(buffer)} frame).")
                        else:
                            print(f"  Scartata: solo {len(buffer)} frame (minimo {MIN_SAMPLE_FRAMES}).")
                        buffer = []
                elif key == ord("n"):
                    break
                elif key in (ord("q"), 27):
                    quit_all = True
                    break

            if quit_all:
                break

    store.save()
    print(f"\nDatabase salvato in '{args.database}'. Etichette totali: {store.known_labels()}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
