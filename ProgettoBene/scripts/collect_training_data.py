"""
scripts/collect_training_data.py
==================================
Raccoglie un dataset etichettato di vettori di input della embedding network
(69-d: 63 di shape normalizzata + 6 di orientamento del polso, vedi
gesture_engine.normalization.geometric.build_embedding_vector) dalla webcam,
da usare per addestrare GestureEmbeddingNet con Triplet Loss (vedi
scripts/train_embedding_net.py).

Uso interattivo:
    python scripts/collect_training_data.py

Comandi a schermo:
    - digita il nome della classe e premi INVIO nel terminale per iniziare
      a registrare quella classe
    - premi 'c' nella finestra video per catturare un campione
    - premi 'n' per passare a una nuova classe (richiede nome nel terminale)
    - premi 'q' o ESC per terminare e salvare il dataset

Il dataset viene salvato in data/training_dataset.npz con array:
    - "vectors": (N, 69) float32
    - "labels":  (N,) stringhe
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.ingestion import CameraStream, HandLandmarkerEngine
from gesture_engine.normalization import build_embedding_vector, build_filter
from gesture_engine.visualization import draw_hand, landmarks_to_pixel_coords

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training_dataset.npz"
)


def main():
    cfg = EngineConfig.load()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    camera = CameraStream(cfg.camera)
    landmarker = HandLandmarkerEngine(cfg.landmarker, project_root=project_root)
    filt = build_filter(cfg.filter)

    vectors, labels = [], []
    if os.path.exists(OUTPUT_PATH):
        data = np.load(OUTPUT_PATH, allow_pickle=True)
        vectors = list(data["vectors"])
        labels = list(data["labels"])
        if vectors and vectors[0].shape[0] != cfg.embedding.input_dim:
            print(
                f"ERRORE: il dataset esistente contiene vettori a {vectors[0].shape[0]}-d, ma la pipeline "
                f"attuale (build_embedding_vector) produce vettori a {cfg.embedding.input_dim}-d. E' stato "
                f"raccolto con una versione precedente della pipeline di feature. Rinomina o elimina "
                f"'{OUTPUT_PATH}' e ricomincia la raccolta da zero."
            )
            return
        print(f"Dataset esistente caricato: {len(vectors)} campioni.")

    current_label = input("Nome della prima classe da registrare: ").strip()

    camera.open()
    landmarker.open()
    print("Premi 'c' per catturare un campione, 'n' per nuova classe, 'q'/ESC per uscire e salvare.")

    try:
        while True:
            ok, frame, ts = camera.read()
            if not ok:
                continue
            detections = landmarker.detect(frame, ts)

            if detections:
                pts_px = landmarks_to_pixel_coords(detections[0].landmarks[:, :2], frame.shape)
                draw_hand(frame, pts_px, cfg.visualization)

            cv2.putText(
                frame, f"Classe: {current_label} | campioni totali: {len(vectors)}",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Raccolta Dataset - Gesture Control Engine", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and detections:
                filtered = filt.filter(detections[0].landmarks)
                vectors.append(build_embedding_vector(filtered))
                labels.append(current_label)
                print(f"Campione catturato per '{current_label}' (totale: {len(vectors)})")

            elif key == ord("n"):
                new_label = input("Nome della nuova classe: ").strip()
                if new_label:
                    current_label = new_label
                    filt.reset()

            elif key in (ord("q"), 27):
                break
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()

    if vectors:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        np.savez(
            OUTPUT_PATH,
            vectors=np.stack(vectors).astype(np.float32),
            labels=np.array(labels, dtype=object),
        )
        print(f"Dataset salvato in {OUTPUT_PATH} ({len(vectors)} campioni totali).")
    else:
        print("Nessun campione raccolto, dataset non salvato.")


if __name__ == "__main__":
    main()
