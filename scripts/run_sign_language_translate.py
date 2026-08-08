"""
scripts/run_sign_language_translate.py
========================================
Converte un video in lingua dei segni in testo, riconoscendo segni
ISOLATI tramite confronto con dei template registrati (vedi
enroll_sign_templates.py) con Dynamic Time Warping (DTW).

COSA FA REALMENTE (leggi prima di usarlo)
------------------------------------------
Questo NON e' un traduttore linguistico generale: e' un riconoscitore di
segni a vocabolario chiuso, template-matching. Concretamente:

  1. Estrae, per ogni frame del video, la forma delle mani (invariante a
     posizione/scala/rotazione) + posizione grezza dei polsi + distanza tra
     le mani (vedi gesture_engine/sign_language/features.py).
  2. Segmenta automaticamente il video in "segni candidati" individuando le
     brevi pause tra un segno e il successivo (assume quindi che il
     segnante faccia una piccola pausa tra un segno e l'altro - vero per il
     fingerspelling e per molti sistemi di riconoscimento isolato, MA NON
     per l'eloquio segnato continuo naturale, dove i segni si susseguono
     senza pause nette).
  3. Confronta ogni segmento con i template registrati via DTW e restituisce
     l'etichetta del template piu' vicino, solo se la distanza e' sotto la
     soglia (rigetto open-set: se nessun template assomiglia abbastanza, il
     segmento viene ignorato invece di essere forzato su un'etichetta a
     caso).
  4. Concatena le etichette riconosciute (glosse) con uno spazio: il
     risultato e' una TRASCRIZIONE in ordine di segnazione, non una
     traduzione grammaticalmente corretta nella lingua parlata target.
     L'ordine dei segni in LIS/ASL/altre lingue dei segni NON coincide con
     l'ordine delle parole in italiano/inglese, e questo script non fa
     alcuna riordinazione sintattica.

LIMITI NOTI
-----------
  - Nessun marcatore NON manuale: espressioni del viso, movimento delle
    labbra, direzione dello sguardo, inclinazioni della testa non sono
    catturati (serve MediaPipe Holistic + landmark del volto, non incluso
    in questo progetto). Molti segni che si distinguono SOLO per questi
    marcatori (es. affermazione vs domanda con la stessa configurazione
    manuale) risulteranno indistinguibili.
  - Vocabolario chiuso: riconosce solo i segni che hai registrato con
    enroll_sign_templates.py. Non e' un sistema pre-addestrato su LIS/ASL.
  - La segmentazione a pausa fallisce su eloquio segnato fluente/continuo
    (co-articolazione tra segni): per quello servirebbe un modello
    sequence-to-sequence con CTC addestrato su un grande corpus annotato
    (es. architetture come le usate per il dataset RWTH-PHOENIX), un
    progetto di ricerca a se', non uno script.

Uso:
    python scripts/run_sign_language_translate.py --video percorso/video.mp4
    python scripts/run_sign_language_translate.py --camera 0      (webcam live)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.ingestion import HandLandmarkerEngine
from gesture_engine.visualization import draw_hand, landmarks_to_pixel_coords
from gesture_engine.sign_language import (
    build_frame_feature,
    split_detections_by_hand,
    motion_energy,
    hands_present,
    best_match,
    SignTemplateStore,
)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sign_templates", "templates_db.npz"
)

# --- Parametri di segmentazione (tarabili da CLI) --------------------------
MOTION_THRESHOLD = 0.015      # spostamento minimo (coord. normalizzate) per considerare un frame "in movimento"
PAUSE_SECONDS = 0.35          # pausa (in secondi) che chiude un segmento
MIN_SEGMENT_SECONDS = 0.25    # sotto questa durata un segmento e' scartato come rumore
MAX_DTW_DISTANCE = 0.09       # oltre questa distanza il segmento e' "sconosciuto" (rigetto open-set)


def parse_args():
    p = argparse.ArgumentParser(description="Riconoscimento di segni isolati da video -> testo (template matching + DTW).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Percorso del file video da elaborare")
    src.add_argument("--camera", type=int, help="Indice webcam per elaborazione live invece che da file")
    p.add_argument("--database", default=DEFAULT_DB_PATH, help="Percorso del database dei template (.npz)")
    p.add_argument("--max-distance", type=float, default=MAX_DTW_DISTANCE, help="Soglia di rigetto DTW (piu' basso = piu' severo)")
    p.add_argument("--output", help="Se indicato, salva la trascrizione anche su questo file .txt")
    p.add_argument("--no-display", action="store_true", help="Non mostrare la finestra video (elaborazione piu' veloce)")
    p.add_argument("--flip", action="store_true", help="Specchia orizzontalmente (utile solo per webcam live)")
    return p.parse_args()


class SignSegmenter:
    """Accumula le feature per-frame e le suddivide in segmenti (segni
    candidati) individuando le pause, secondo la logica descritta nel
    docstring del modulo."""

    def __init__(self, fps: float):
        self.pause_frames = max(1, int(round(PAUSE_SECONDS * fps)))
        self.min_segment_frames = max(1, int(round(MIN_SEGMENT_SECONDS * fps)))
        self._buffer: List[np.ndarray] = []
        self._idle_run = 0
        self._prev_feat: Optional[np.ndarray] = None

    def push(self, feat: np.ndarray) -> Optional[np.ndarray]:
        """Aggiunge un frame. Ritorna un segmento (np.ndarray (T,D)) quando se
        ne chiude uno abbastanza lungo da essere considerato un segno valido,
        altrimenti None."""
        present = hands_present(feat)
        moving = motion_energy(self._prev_feat, feat) > MOTION_THRESHOLD
        self._prev_feat = feat

        if present:
            self._buffer.append(feat)
            self._idle_run = 0 if moving else self._idle_run + 1
        else:
            self._idle_run += 1

        if self._idle_run >= self.pause_frames and self._buffer:
            return self._flush()
        return None

    def flush_final(self) -> Optional[np.ndarray]:
        """Da chiamare a fine video per non perdere l'ultimo segno in corso."""
        if self._buffer:
            return self._flush()
        return None

    def _flush(self) -> Optional[np.ndarray]:
        segment = self._buffer
        self._buffer = []
        self._idle_run = 0
        if len(segment) < self.min_segment_frames:
            return None
        return np.stack(segment, axis=0)


def main():
    args = parse_args()

    cfg = EngineConfig.load()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    store = SignTemplateStore(args.database)
    store.load()
    if not store.templates:
        print(f"ATTENZIONE: nessun template trovato in '{args.database}'.")
        print("Registra prima dei segni con: python scripts/enroll_sign_templates.py")
        return
    print(f"Template caricati: {store.known_labels()}")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Impossibile aprire il video: {args.video}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        flip = args.flip
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera)
        fps = cfg.camera.target_fps or 30.0
        flip = True if not args.flip else args.flip  # default a specchio per webcam live

    segmenter = SignSegmenter(fps)
    recognized: List[str] = []
    frame_idx = 0
    start_t = time.time()

    with HandLandmarkerEngine(cfg.landmarker, project_root) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            ts_ms = int(frame_idx * (1000.0 / fps))
            detections = landmarker.detect(frame, ts_ms)
            by_hand = split_detections_by_hand(detections)
            feat = build_frame_feature(by_hand)

            segment = segmenter.push(feat)
            if segment is not None:
                label, dist = best_match(segment, store.templates)
                if label is not None and dist <= args.max_distance:
                    recognized.append(label)
                    print(f"[{frame_idx / fps:6.2f}s] Riconosciuto: '{label}'  (distanza DTW {dist:.4f})")
                else:
                    print(f"[{frame_idx / fps:6.2f}s] Segmento non riconosciuto (distanza minima {dist:.4f})")

            if not args.no_display:
                for det in detections:
                    pts_px = landmarks_to_pixel_coords(det.landmarks[:, :2], frame.shape)
                    draw_hand(frame, pts_px, cfg.visualization)
                cv2.putText(frame, " ".join(recognized[-8:]), (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
                cv2.imshow("Sign Language Translate", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_idx += 1

        # non perdere un segno rimasto "in corso" alla fine del video
        tail = segmenter.flush_final()
        if tail is not None:
            label, dist = best_match(tail, store.templates)
            if label is not None and dist <= args.max_distance:
                recognized.append(label)
                print(f"[fine video] Riconosciuto: '{label}'  (distanza DTW {dist:.4f})")

    cap.release()
    cv2.destroyAllWindows()

    text = " ".join(recognized)
    elapsed = time.time() - start_t
    print("\n" + "=" * 60)
    print(f"Trascrizione (glosse, {len(recognized)} segni, {elapsed:.1f}s di elaborazione):")
    print(text if text else "(nessun segno riconosciuto)")
    print("=" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Salvato in '{args.output}'")


if __name__ == "__main__":
    main()
