"""
onnx_export.py
===============
Esporta GestureEmbeddingNet (PyTorch) in formato ONNX per inferenza
ad altissima velocita' su CPU (<1ms per forward pass su hardware moderno,
grazie alla rete leggera e al runtime ottimizzato onnxruntime).

Uso tipico:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import os

import torch

from gesture_engine.recognition.embedding_net import GestureEmbeddingNet


def export_to_onnx(
    embedding_config,
    torch_weights_path: str = None,
    onnx_output_path: str = None,
    opset: int = 17,
) -> str:
    torch_weights_path = torch_weights_path or embedding_config.torch_weights_path
    onnx_output_path = onnx_output_path or embedding_config.onnx_model_path

    model = GestureEmbeddingNet(
        input_dim=embedding_config.input_dim,
        hidden_dims=embedding_config.hidden_dims,
        embedding_dim=embedding_config.embedding_dim,
    )

    if os.path.exists(torch_weights_path):
        state = torch.load(torch_weights_path, map_location="cpu")
        checkpoint_input_dim = state.get("backbone.0.weight", None)
        checkpoint_input_dim = checkpoint_input_dim.shape[1] if checkpoint_input_dim is not None else None
        if checkpoint_input_dim is not None and checkpoint_input_dim != embedding_config.input_dim:
            raise RuntimeError(
                f"Il checkpoint '{torch_weights_path}' si aspetta input_dim={checkpoint_input_dim}, ma "
                f"EmbeddingConfig.input_dim={embedding_config.input_dim}. La pipeline di feature e' cambiata "
                f"da quando questo checkpoint e' stato salvato: riaddestra prima con "
                f"'scripts/train_embedding_net.py' (che riparte da zero automaticamente in questo caso) e poi "
                f"riesporta."
            )
        model.load_state_dict(state)
        print(f"[onnx_export] Pesi caricati da {torch_weights_path}")
    else:
        print(
            f"[onnx_export] ATTENZIONE: nessun checkpoint trovato in "
            f"'{torch_weights_path}'. Esporto la rete con pesi inizializzati "
            f"casualmente (utile solo per testare la pipeline end-to-end)."
        )

    model.eval()

    dummy_input = torch.randn(1, embedding_config.input_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    export_kwargs = dict(
        input_names=["landmarks_flat"],
        output_names=["embedding"],
        dynamic_axes={"landmarks_flat": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
    )
    try:
        # PyTorch >= 2.x preferisce l'esportatore "dynamo"; richiede il pacchetto
        # opzionale 'onnxscript'. Se non presente, ricadiamo sull'esportatore
        # legacy basato su TorchScript tracing, comunque pienamente valido.
        torch.onnx.export(model, dummy_input, onnx_output_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Versioni piu' vecchie di PyTorch non conoscono l'argomento 'dynamo'
        torch.onnx.export(model, dummy_input, onnx_output_path, **export_kwargs)
    print(f"[onnx_export] Modello esportato in {onnx_output_path}")
    return onnx_output_path


if __name__ == "__main__":
    from gesture_engine.config import EngineConfig

    cfg = EngineConfig.load()
    export_to_onnx(cfg.embedding)
