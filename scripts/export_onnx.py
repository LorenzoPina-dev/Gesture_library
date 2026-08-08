"""
scripts/export_onnx.py
=======================
Esporta la GestureEmbeddingNet (pesi in models/gesture_embedding.pt, se
presenti) in formato ONNX per inferenza ad alta velocita' su CPU.

Uso:
    python scripts/export_onnx.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.recognition.onnx_export import export_to_onnx


def main():
    cfg = EngineConfig.load()
    export_to_onnx(cfg.embedding)


if __name__ == "__main__":
    main()
