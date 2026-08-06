"""
scripts/enroll_from_images.py
===============================
Aggiunge nuove classi di gesture al database k-NN
(data/enrolled_gestures/embeddings_db.npz) partendo da un dataset di
immagini organizzato a cartelle per classe, usando la rete di embedding
GIA' ADDESTRATA E CONGELATA (models/gesture_embedding.pt o .onnx).

Differenza rispetto a build_dataset_from_images.py:
  - build_dataset_from_images.py prepara dati per ADDESTRARE il backbone
    (cambia i pesi della rete -> invalida tutti gli embedding esistenti).
  - questo script e' invece PURAMENTE ADDITIVO: calcola nuovi embedding
    con la rete attuale (che NON viene modificata) e li accoda al
    database k-NN gia' esistente, senza toccare o invalidare nulla di
    cio' che hai gia' enrollato (dalla webcam o da esecuzioni precedenti
    di questo stesso script).

IMPORTANTE: esegui questo script solo DOPO aver addestrato/congelato la
rete di embedding almeno una volta (models/gesture_embedding.pt o .onnx
presenti). Se in futuro riaddestri il backbone, tutti gli embedding
(vecchi e nuovi) diventano incompatibili tra loro e vanno ricalcolati.

Uso:
    python scripts/enroll_from_images.py --dataset_root D:\\percorso\\dataset
    python scripts/enroll_from_images.py --dataset_root D:\\percorso\\dataset --classes fist,peace
    python scripts/enroll_from_images.py --dataset_root D:\\percorso\\dataset --max_per_class 30
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
from gesture_engine.recognition import EmbeddingInferenceEngine, EmbeddingKNNClassifier

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(dataset_root: str, only_classes):
    for class_name in sorted(os.listdir(dataset_root)):
        if only_classes is not None and class_name not in only_classes:
            continue
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTENSIONS:
                yield class_name, os.path.join(class_dir, fname)


def main():
    parser = argparse.ArgumentParser(
        description="Enrolla nuove classi nel database k-NN partendo da immagini (additivo: non invalida nulla di gia' presente)."
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--classes", default=None, help="Lista separata da virgole di classi da enrollare (default: tutte le sottocartelle).")
    parser.add_argument("--max_per_class", type=int, default=30, help="Numero massimo di campioni da enrollare per classe.")
    args = parser.parse_args()

    only_classes = set(args.classes.split(",")) if args.classes else None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = EngineConfig.load()

    landmarker = HandLandmarkerEngine(cfg.landmarker, project_root=project_root)
    embedding_engine = EmbeddingInferenceEngine(cfg.embedding, project_root=project_root)
    knn = EmbeddingKNNClassifier(cfg.knn)

    landmarker.open()
    embedding_engine.open()
    knn.load()  # carica il database ESISTENTE: tutto cio' che c'e' gia' resta intatto

    if embedding_engine._backend != "onnx" and not os.path.exists(
        embedding_engine._resolve(cfg.embedding.torch_weights_path)
    ):
        print(
            "ATTENZIONE: nessun modello addestrato trovato (ne' .onnx ne' .pt): "
            "la rete userebbe pesi casuali e gli embedding calcolati ora NON "
            "sarebbero compatibili con quelli futuri. Addestra prima il backbone "
            "con scripts/train_embedding_net.py."
        )
        landmarker.close()
        embedding_engine.close()
        return

    per_class_embeddings = {}
    ts_ms = 0
    skipped_no_hand = 0
    processed = 0

    try:
        for class_name, img_path in iter_images(args.dataset_root, only_classes):
            if len(per_class_embeddings.get(class_name, [])) >= args.max_per_class:
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            ts_ms += 33
            detections = landmarker.detect(frame, ts_ms)
            if not detections:
                skipped_no_hand += 1
                continue

            normalized = normalize_landmarks(detections[0].landmarks)
            flat = flatten(normalized)
            embedding = embedding_engine.embed(flat)
            per_class_embeddings.setdefault(class_name, []).append(embedding)

            processed += 1
            if processed % 50 == 0:
                print(f"  ...{processed} campioni elaborati")
    finally:
        landmarker.close()
        embedding_engine.close()

    if not per_class_embeddings:
        print("Nessun campione valido estratto. Controlla --dataset_root e la struttura a cartelle.")
        return

    for class_name, embeddings in per_class_embeddings.items():
        knn.enroll(class_name, np.stack(embeddings, axis=0))
        print(f"Enrollata classe '{class_name}': {len(embeddings)} campioni aggiunti.")

    knn.save()  # append-only: le classi/gli embedding gia' presenti restano invariati
    totals = knn.known_classes()
    print(f"Database k-NN aggiornato: {sum(totals.values())} campioni totali, {len(totals)} classi.")
    print(f"Immagini scartate (mano non rilevata): {skipped_no_hand}")


if __name__ == "__main__":
    main()
