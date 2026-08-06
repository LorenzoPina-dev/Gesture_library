"""
embedding_net.py
=================
Livello 2 (parte 1/2): GestureEmbeddingNet, una piccola MLP che proietta il
vettore di input (default 69-d: 63 di shape normalizzata invariante a
rotazione, 21 punti x xyz, + 6 di orientamento del polso, vedi
gesture_engine.normalization.geometric.build_embedding_vector) in uno spazio
di embedding a 128 dimensioni, L2-normalizzato, adatto a classificazione
k-NN basata su distanza del coseno con Triplet Loss / Contrastive Loss.

Nota di design: la rete e' volutamente leggera (poche centinaia di migliaia
di parametri) per garantire inferenza <1ms su CPU una volta esportata in ONNX
(vedi onnx_export.py). L'addestramento con Triplet Loss e' implementato in
scripts/train_embedding_net.py e richiede un dataset etichettato raccolto
dall'utente (vedi README "Addestramento del Livello 2").

Se non viene fornito un checkpoint addestrato, la rete puo' comunque essere
usata "as-is" (pesi inizializzati) in combinazione con il Few-Shot Dynamic
Enrollment: gli embedding non saranno semanticamente ottimali quanto quelli
di una rete addestrata con triplet loss, ma il classificatore k-NN con
soglia di rigetto open-set rimane funzionante per gesture enrollate
dall'utente stesso, perche' la normalizzazione geometrica a monte fa gia'
gran parte del lavoro di invarianza.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureEmbeddingNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 69,
        hidden_dims: Sequence[int] = (256, 256),
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> ritorna embedding L2-normalizzato (B, embedding_dim)."""
        h = self.backbone(x)
        emb = self.head(h)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def embed_single(self, flat_vector) -> "torch.Tensor":
        """Comoda per inferenza singola: accetta un array (input_dim,) e ritorna (embedding_dim,)."""
        self.eval()
        x = torch.as_tensor(flat_vector, dtype=torch.float32).unsqueeze(0)
        return self.forward(x).squeeze(0)


class TripletLoss(nn.Module):
    """Triplet Loss standard su embedding L2-normalizzati (distanza euclidea == funzione monotona
    della distanza del coseno per vettori unitari, quindi coerente con il k-NN a coseno usato
    in fase di inferenza)."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(d_pos - d_neg + self.margin)
        return losses.mean()
