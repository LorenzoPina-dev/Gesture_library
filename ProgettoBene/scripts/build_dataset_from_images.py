"""
scripts/build_dataset_from_images.py
======================================
Costruisce (o estende) data/training_dataset.npz a partire da un dataset
PUBBLICO di immagini organizzato "a cartelle per classe":

    dataset_root/
        classe_1/
            img001.jpg
            img002.jpg
            ...
        classe_2/
            ...

Per ogni immagine rileva la mano con MediaPipe ed esegue esattamente la
stessa normalizzazione geometrica usata a runtime da GestureEngine
(fondamentale per coerenza), salvando vettori 63-d con la stessa identica
struttura prodotta da scripts/collect_training_data.py. Cosi'
scripts/train_embedding_net.py funziona invariato, ed e' anche possibile
FONDERE un dataset pubblico con campioni raccolti a mano dalla webcam
(--merge).

ATTENZIONE - IMPORTANTE:
Questo script produce dati per ADDESTRARE il backbone (la rete di
embedding condivisa). Ogni volta che riaddestri il backbone, TUTTI gli
embedding gia' calcolati (comprese le tue gesture custom enrollate nel
tab "Enrollment" della UI) diventano incompatibili con la nuova rete e
vanno ricalcolati da capo (ri-enrollali dopo il training).
Se invece vuoi solo AGGIUNGERE nuove classi riconoscibili senza toccare
quelle gia' enrollate, usa scripts/enroll_from_images.py (additivo, non
richiede retraining e non invalida nulla).

Uso:
    python scripts/build_dataset_from_images.py --dataset_root D:\\percorso\\hagrid
    python scripts/build_dataset_from_images.py --dataset_root D:\\percorso\\hagrid --merge
    python scripts/build_dataset_from_images.py --dataset_root D:\\percorso\\hagrid --max_per_class 300
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.ingestion import HandLandmarkerEngine
from gesture_engine.normalization import normalize_landmarks, flatten

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training_dataset.npz"
)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(dataset_root: str):
    for class_name in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTENSIONS:
                yield class_name, os.path.join(class_dir, fname)


def main():
    parser = argparse.ArgumentParser(
        description="Costruisce training_dataset.npz da un dataset pubblico di immagini (cartella per classe)."
    )
    parser.add_argument("--dataset_root", required=True, help="Cartella radice, con una sottocartella per classe.")
    parser.add_argument("--merge", action="store_true", help="Aggiunge ai campioni gia' presenti invece di sovrascrivere.")
    parser.add_argument("--max_per_class", type=int, default=None, help="Limite immagini per classe (utile su dataset enormi).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = EngineConfig.load()
    landmarker = HandLandmarkerEngine(cfg.landmarker, project_root=project_root)
    landmarker.open()

    vectors, labels = [], []
    if args.merge and os.path.exists(OUTPUT_PATH):
        data = np.load(OUTPUT_PATH, allow_pickle=True)
        vectors = list(data["vectors"])
        labels = list(data["labels"])
        print(f"Dataset esistente caricato: {len(vectors)} campioni (verranno estesi).")

    per_class_count = {}
    skipped_no_hand = 0
    processed = 0
    ts_ms = 0

    try:
        for class_name, img_path in iter_images(args.dataset_root):
            if args.max_per_class is not None and per_class_count.get(class_name, 0) >= args.max_per_class:
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            ts_ms += 33  # timestamp fittizio incrementale (ms), richiesto in modalita' VIDEO
            detections = landmarker.detect(frame, ts_ms)
            if not detections:
                skipped_no_hand += 1
                continue

            normalized = normalize_landmarks(detections[0].landmarks)
            vectors.append(flatten(normalized))
            labels.append(class_name)
            per_class_count[class_name] = per_class_count.get(class_name, 0) + 1

            processed += 1
            if processed % 200 == 0:
                print(f"  ...{processed} campioni estratti finora ({len(per_class_count)} classi)")
    finally:
        landmarker.close()

    if not vectors:
        print("Nessun campione valido estratto. Controlla --dataset_root e la struttura a cartelle.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        vectors=np.stack(vectors).astype(np.float32),
        labels=np.array(labels, dtype=object),
    )
    print(f"Salvato {OUTPUT_PATH}: {len(vectors)} campioni totali, {len(set(labels))} classi.")
    print(f"Immagini scartate (mano non rilevata): {skipped_no_hand}")
    print("Ora esegui: python scripts/train_embedding_net.py")


if __name__ == "__main__":
    main()
