"""
onnx_inference.py
==================
Runtime di inferenza per la embedding network esportata in ONNX
(onnxruntime, CPUExecutionProvider). Fallback automatico su PyTorch
"eager mode" se il file .onnx non e' ancora stato generato, cosi' il
sistema resta utilizzabile anche prima dell'export esplicito.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class EmbeddingInferenceEngine:
    """
    Interfaccia unica per ottenere l'embedding a 128-d di un vettore di
    landmark normalizzati (63,), indipendentemente dal backend (ONNX o
    PyTorch) effettivamente disponibile.
    """

    def __init__(self, embedding_config, project_root: Optional[str] = None):
        self.cfg = embedding_config
        self._project_root = project_root or os.getcwd()
        self._backend = None
        self._onnx_session = None
        self._torch_model = None

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path) and os.path.exists(path):
            return path
        candidate = os.path.join(self._project_root, path)
        return candidate if os.path.exists(candidate) else path

    def open(self) -> None:
        onnx_path = self._resolve(self.cfg.onnx_model_path)
        if os.path.exists(onnx_path):
            import onnxruntime as ort

            self._onnx_session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._backend = "onnx"
            print(f"[EmbeddingInferenceEngine] Backend ONNX attivo ({onnx_path})")
            return

        # Fallback: PyTorch eager (utile in sviluppo, prima dell'export)
        import torch
        from gesture_engine.recognition.embedding_net import GestureEmbeddingNet

        model = GestureEmbeddingNet(
            input_dim=self.cfg.input_dim,
            hidden_dims=self.cfg.hidden_dims,
            embedding_dim=self.cfg.embedding_dim,
        )
        weights_path = self._resolve(self.cfg.torch_weights_path)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        self._torch_model = model
        self._backend = "torch"
        print(
            "[EmbeddingInferenceEngine] Nessun modello ONNX trovato: uso il "
            "fallback PyTorch (eseguire scripts/export_onnx.py per performance ottimali)."
        )

    def embed(self, flat_vector: np.ndarray) -> np.ndarray:
        """flat_vector: (63,) float32 -> ritorna embedding (128,) float32, L2-normalizzato."""
        x = flat_vector.astype(np.float32).reshape(1, -1)

        if self._backend == "onnx":
            outputs = self._onnx_session.run(None, {"landmarks_flat": x})
            return outputs[0][0]

        elif self._backend == "torch":
            import torch

            with torch.no_grad():
                t = torch.from_numpy(x)
                out = self._torch_model(t)
            return out.numpy()[0]

        raise RuntimeError("EmbeddingInferenceEngine non aperto. Chiama .open() prima di .embed().")

    def close(self) -> None:
        self._onnx_session = None
        self._torch_model = None
        self._backend = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
