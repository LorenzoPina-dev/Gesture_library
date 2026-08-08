"""
scripts/train_embedding_net.py
================================
Addestra GestureEmbeddingNet con Triplet Loss sul dataset etichettato
prodotto da scripts/collect_training_data.py (data/training_dataset.npz).

Strategia di mining delle triplette: "batch-hard semi-random" — per ogni
epoca, per ogni ancora si campiona un positivo casuale della stessa classe
e un negativo casuale di classe diversa. Con dataset piccoli (few-shot,
tipicamente 5-30 campioni per classe) questo e' sufficiente e molto piu'
stabile del batch-hard mining "vero" che richiederebbe batch grandi.

Uso:
    python scripts/train_embedding_net.py --epochs 200 --lr 1e-3

Al termine, salva i pesi in models/gesture_embedding.pt. Esporta poi in
ONNX con scripts/export_onnx.py.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.recognition.embedding_net import GestureEmbeddingNet, TripletLoss

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training_dataset.npz"
)


class TripletDataset(Dataset):
    """Genera triplette (anchor, positive, negative) campionando dinamicamente
    dal dataset ad ogni __getitem__, cosi' da massimizzare la varieta' delle
    combinazioni anche con pochi campioni per classe."""

    def __init__(self, vectors: np.ndarray, labels: np.ndarray, epoch_size: int = 2000):
        self.vectors = vectors
        self.labels = labels
        self.epoch_size = epoch_size
        self.by_class = {}
        for i, lbl in enumerate(labels):
            self.by_class.setdefault(lbl, []).append(i)
        self.classes = list(self.by_class.keys())
        if len(self.classes) < 2:
            raise ValueError(
                "Servono almeno 2 classi diverse nel dataset per la Triplet Loss. "
                "Usa scripts/collect_training_data.py per raccoglierne di piu'."
            )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx + np.random.randint(0, 1_000_000))
        anchor_class = rng.choice(self.classes)
        anchor_idx, positive_idx = rng.choice(self.by_class[anchor_class], size=2, replace=True)

        negative_class = rng.choice([c for c in self.classes if c != anchor_class])
        negative_idx = rng.choice(self.by_class[negative_class])

        return (
            torch.from_numpy(self.vectors[anchor_idx]).float(),
            torch.from_numpy(self.vectors[positive_idx]).float(),
            torch.from_numpy(self.vectors[negative_idx]).float(),
        )


def train(epochs: int, lr: float, batch_size: int, margin: float) -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset non trovato in {DATASET_PATH}. Esegui prima "
            f"scripts/collect_training_data.py per raccogliere campioni etichettati."
        )

    data = np.load(DATASET_PATH, allow_pickle=True)
    vectors, labels = data["vectors"], data["labels"]
    print(f"Dataset caricato: {len(vectors)} campioni, {len(set(labels))} classi.")

    cfg = EngineConfig.load()
    dataset = TripletDataset(vectors, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GestureEmbeddingNet(
        input_dim=cfg.embedding.input_dim,
        hidden_dims=cfg.embedding.hidden_dims,
        embedding_dim=cfg.embedding.embedding_dim,
    )
    if os.path.exists(cfg.embedding.torch_weights_path):
        checkpoint = torch.load(cfg.embedding.torch_weights_path, map_location="cpu")
        checkpoint_input_dim = checkpoint.get("backbone.0.weight", None)
        checkpoint_input_dim = checkpoint_input_dim.shape[1] if checkpoint_input_dim is not None else None

        if checkpoint_input_dim is not None and checkpoint_input_dim != cfg.embedding.input_dim:
            # Il checkpoint su disco e' stato addestrato con una versione precedente
            # della pipeline di feature (es. senza le 6 feature di orientamento del
            # polso: 63-d invece delle attuali 69-d). I pesi del primo layer non sono
            # compatibili con la shape di input corrente: caricarli comunque farebbe
            # fallire load_state_dict con un size mismatch. In questo caso e'
            # necessario ripartire da zero (i pesi non sono ri-tarabili tra input_dim
            # diversi) invece di far crashare il training.
            print(
                f"ATTENZIONE: il checkpoint esistente in '{cfg.embedding.torch_weights_path}' si aspetta "
                f"input_dim={checkpoint_input_dim}, ma la configurazione attuale usa "
                f"input_dim={cfg.embedding.input_dim} (probabilmente la pipeline di feature e' cambiata, "
                f"es. aggiunta delle feature di orientamento del polso). Il checkpoint NON verra' caricato: "
                f"si riparte da pesi inizializzati casualmente. Al termine del training ricorda di rieseguire "
                f"'scripts/export_onnx.py' e di ri-enrollare tutte le gesture custom (i vecchi embedding non "
                f"sono piu' compatibili con la rete riaddestrata)."
            )
        else:
            model.load_state_dict(checkpoint)
            print("Checkpoint esistente caricato, si continua il training (fine-tuning).")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = TripletLoss(margin=margin)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            loss = loss_fn(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoca {epoch}/{epochs} — loss media: {total_loss / max(n_batches,1):.4f}")

    os.makedirs(os.path.dirname(cfg.embedding.torch_weights_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.embedding.torch_weights_path)
    print(f"Training completato. Pesi salvati in {cfg.embedding.torch_weights_path}")
    print("Esegui ora 'python scripts/export_onnx.py' per generare la versione ONNX ad alte prestazioni.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GestureEmbeddingNet con Triplet Loss")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, margin=args.margin)
