"""
diagnostics/projection.py
==========================
Wrapper unico per le tecniche di riduzione dimensionale usate per
visualizzare lo spazio di embedding a 128-d in 2D:

  - PCA: sempre disponibile (scikit-learn), preserva la varianza globale
    e lineare. Prima cosa da guardare: dice quanto lo spazio a 128-d e'
    "davvero" usato (vedi anche diagnostics.metrics.effective_rank, che
    e' la versione numerica dello stesso concetto).
  - UMAP: se installato (pip install umap-learn), standard de facto per
    preservare sia struttura locale sia un po' di struttura globale.
  - t-SNE: fallback se UMAP non e' disponibile (preserva bene solo la
    struttura locale, NON fidarsi delle distanze tra cluster lontani).

fit_transform_2d ritorna anche il "reducer" fittato quando possibile
(PCA e UMAP lo supportano nativamente; t-SNE no, non ha un vero .transform
per nuovi punti — per proiettare nuovi campioni con t-SNE bisogna rifare
il fit su old+new, gestito automaticamente da project_new_points).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def pca_2d(embeddings: np.ndarray):
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    return coords, reducer


def pca_variance_explained(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Quota di varianza spiegata dalle prime n_components componenti principali.
    Se le prime ~5-10 spiegano gia' >90-95%, lo spazio effettivo usato dalla
    rete e' molto piu' piccolo della dimensione nominale (128) — stesso
    fenomeno misurato numericamente da diagnostics.metrics.effective_rank."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    reducer = PCA(n_components=n_components, random_state=42)
    reducer.fit(embeddings)
    return reducer.explained_variance_ratio_


def best_available_2d(embeddings: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, str, Optional[object]]:
    """Prova UMAP (metrica coseno, coerente col k-NN usato in produzione),
    fallback su t-SNE se umap-learn non e' installato. Ritorna
    (coords (N,2), nome_tecnica, reducer_o_None)."""
    try:
        import umap

        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed)
        coords = reducer.fit_transform(embeddings)
        return coords, "UMAP", reducer
    except ImportError:
        from sklearn.manifold import TSNE

        n = embeddings.shape[0]
        perplexity = max(5, min(30, n // 4))
        reducer = TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplexity, random_state=seed)
        coords = reducer.fit_transform(embeddings)
        return coords, "t-SNE", None


def project_new_points(reducer, technique: str, base_embeddings: np.ndarray, new_embeddings: np.ndarray) -> np.ndarray:
    """
    Proietta nuovi embedding (es. una gesture candidata non ancora nel
    training set) nello STESSO spazio 2D gia' calcolato per il dataset
    esistente, cosi' da poterli sovrapporre visivamente. Con PCA/UMAP si usa
    .transform() (nessun refit, confrontabile). Con t-SNE non esiste una
    vera trasformazione per nuovi punti: si rifà il fit su base+nuovi e si
    ritornano solo le coordinate dei nuovi punti (le coordinate della base
    cambieranno leggermente rispetto al plot originale — e' un limite noto
    di t-SNE, per questo UMAP e' preferibile quando serve confrontare
    dataset esistente vs candidato).
    """
    if technique in ("PCA", "UMAP") and reducer is not None:
        return reducer.transform(new_embeddings)

    from sklearn.manifold import TSNE

    combined = np.concatenate([base_embeddings, new_embeddings], axis=0)
    n = combined.shape[0]
    perplexity = max(5, min(30, n // 4))
    reducer2 = TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplexity, random_state=42)
    coords_all = reducer2.fit_transform(combined)
    return coords_all[base_embeddings.shape[0]:]
