"""
scripts/analyze_embeddings.py
==============================
Strumento diagnostico per lo spazio di embedding a 128-d prodotto da
GestureEmbeddingNet (Livello 2). Risponde a tre domande:

  1. VISUALIZZAZIONE: come e' fatto lo spazio a 128-d? Le classi formano
     cluster separati o si sovrappongono? Quanto e' "usato" davvero lo
     spazio disponibile (collasso dimensionale)?
  2. INTERPRETABILITA' PER-DIMENSIONE: cosa rappresenta ogni singola
     dimensione dell'embedding? E' possibile collegarla a feature note e
     interpretabili (angoli di orientamento, curl, pinch, estensione dita)?
     Se un domani vuoi separare due gesture in base a UNA feature precisa,
     questo report ti dice se quella feature e' gia' isolata su un asse
     (separabile linearmente) o "attorcigliata" con altre (rischio di
     confusione nel k-NN).
  3. METRICHE QUANTITATIVE: Recall@K, Triplet Accuracy, Silhouette,
     Alignment/Uniformity (Wang & Isola 2020), Effective Rank, rapporto
     distanza intra/inter-classe, matrice di confondibilita' tra classi.

Le feature geometriche "note" contro cui si confronta l'embedding sono
ricalcolate dagli stessi 69 valori di input della rete (non servono i
landmark grezzi): 63 di shape normalizzata -> finger_extension_ratios,
pinch_distance, average_curl (vedi gesture_engine.normalization.geometric);
6 di orientamento -> roll/pitch/yaw ricostruiti da sin/cos.

Uso:
    python scripts/analyze_embeddings.py
    python scripts/analyze_embeddings.py --dataset data/training_dataset.npz --k 5 --open
    python scripts/analyze_embeddings.py --umap-neighbors 8 --n-triplets 8000

Dipendenze aggiuntive (non necessarie per l'uso runtime della libreria):
    pip install -r requirements-analysis.txt

Genera un report HTML interattivo (Plotly, autonomo, apribile offline) in
reports/embedding_analysis.html.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import os
import sys
import webbrowser
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_engine.config import EngineConfig
from gesture_engine.normalization import average_curl, finger_extension_ratios, pinch_distance
from gesture_engine.recognition.embedding_net import GestureEmbeddingNet

import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "training_dataset.npz")
DEFAULT_ENROLLED_DB_PATH = os.path.join(PROJECT_ROOT, "data", "enrolled_gestures", "embeddings_db.npz")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "reports", "embedding_analysis.html")

FEATURE_NAMES = [
    "thumb_ext", "index_ext", "middle_ext", "ring_ext", "pinky_ext",
    "finger_count", "pinch_thumb_index", "avg_curl", "roll_deg", "pitch_deg", "yaw_deg",
]


# ---------------------------------------------------------------------- #
# Caricamento dati e modello
# ---------------------------------------------------------------------- #

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset non trovato in '{path}'. Esegui prima scripts/collect_training_data.py."
        )
    data = np.load(path, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    labels = np.array([str(l) for l in data["labels"]])
    return vectors, labels


def load_enrolled_db(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = np.array([str(l) for l in data["labels"]])
    if embeddings.shape[0] == 0:
        return None
    return embeddings, labels


def load_model(cfg: EngineConfig) -> GestureEmbeddingNet:
    weights_path = cfg.embedding.torch_weights_path
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(PROJECT_ROOT, weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Nessun checkpoint trovato in '{weights_path}'. Esegui prima "
            f"'python scripts/train_embedding_net.py' per addestrare la rete."
        )
    model = GestureEmbeddingNet(
        input_dim=cfg.embedding.input_dim,
        hidden_dims=cfg.embedding.hidden_dims,
        embedding_dim=cfg.embedding.embedding_dim,
    )
    state = torch.load(weights_path, map_location="cpu")
    ckpt_input_dim = state.get("backbone.0.weight", None)
    ckpt_input_dim = ckpt_input_dim.shape[1] if ckpt_input_dim is not None else None
    if ckpt_input_dim is not None and ckpt_input_dim != cfg.embedding.input_dim:
        raise RuntimeError(
            f"Il checkpoint '{weights_path}' si aspetta input_dim={ckpt_input_dim}, ma la config "
            f"attuale usa input_dim={cfg.embedding.input_dim}. Riaddestra con "
            f"scripts/train_embedding_net.py prima di analizzare."
        )
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def embed_all(model: GestureEmbeddingNet, vectors: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(vectors).float()
    return model(x).numpy()


# ---------------------------------------------------------------------- #
# Feature geometriche note (per l'analisi di interpretabilita')
# ---------------------------------------------------------------------- #

def compute_known_features(vectors: np.ndarray, finger_extended_ratio: float) -> Dict[str, np.ndarray]:
    """Ricostruisce, dai 69 valori di input della rete, un set di feature
    geometriche note e interpretabili, per confrontarle con lo spazio di
    embedding (correlazione, linear probing)."""
    n = vectors.shape[0]
    shape_part = vectors[:, :63].reshape(n, 21, 3)
    orient_part = vectors[:, 63:69]  # [sin_r, sin_p, sin_y, cos_r, cos_p, cos_y]

    out = {name: np.zeros(n, dtype=np.float64) for name in FEATURE_NAMES}
    for i in range(n):
        pts = shape_part[i]
        ratios = finger_extension_ratios(pts)
        out["thumb_ext"][i] = ratios["thumb"]
        out["index_ext"][i] = ratios["index"]
        out["middle_ext"][i] = ratios["middle"]
        out["ring_ext"][i] = ratios["ring"]
        out["pinky_ext"][i] = ratios["pinky"]
        out["finger_count"][i] = sum(1 for r in ratios.values() if r > finger_extended_ratio)
        out["pinch_thumb_index"][i] = pinch_distance(pts, "thumb", "index")
        out["avg_curl"][i] = average_curl(pts)

    sin_r, sin_p, sin_y = orient_part[:, 0], orient_part[:, 1], orient_part[:, 2]
    cos_r, cos_p, cos_y = orient_part[:, 3], orient_part[:, 4], orient_part[:, 5]
    out["roll_deg"] = np.degrees(np.arctan2(sin_r, cos_r))
    out["pitch_deg"] = np.degrees(np.arctan2(sin_p, cos_p))
    out["yaw_deg"] = np.degrees(np.arctan2(sin_y, cos_y))
    return out


# ---------------------------------------------------------------------- #
# Metriche quantitative
# ---------------------------------------------------------------------- #

def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = e @ e.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> Tuple[float, Dict[str, float]]:
    """Per ogni campione, frazione dei k vicini piu' prossimi (esclusa se' stesso)
    che condividono la stessa classe. Ritorna (macro-media, per-classe)."""
    dist = cosine_distance_matrix(embeddings)
    np.fill_diagonal(dist, np.inf)
    n = embeddings.shape[0]
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return 0.0, {}

    per_sample = np.zeros(n)
    nearest_idx = np.argpartition(dist, k_eff, axis=1)[:, :k_eff]
    for i in range(n):
        neighbor_labels = labels[nearest_idx[i]]
        per_sample[i] = np.mean(neighbor_labels == labels[i])

    per_class: Dict[str, float] = {}
    for cls in np.unique(labels):
        per_class[cls] = float(np.mean(per_sample[labels == cls]))
    return float(np.mean(per_sample)), per_class


def triplet_accuracy(embeddings: np.ndarray, labels: np.ndarray, n_samples: int, rng: np.random.Generator) -> Optional[float]:
    """Frazione di triplette (anchor, positive, negative) campionate a caso per cui
    d(a,p) < d(a,n), cioe' esattamente cio' che la Triplet Loss ottimizza in training."""
    by_class: Dict[str, np.ndarray] = {cls: np.where(labels == cls)[0] for cls in np.unique(labels)}
    classes = [c for c, idx in by_class.items() if len(idx) >= 2]
    if len(classes) < 2:
        return None

    correct = 0
    for _ in range(n_samples):
        anchor_class = rng.choice(classes)
        a_idx, p_idx = rng.choice(by_class[anchor_class], size=2, replace=False)
        negative_class = rng.choice([c for c in classes if c != anchor_class] or list(by_class.keys()))
        n_idx = rng.choice(by_class[negative_class])

        d_ap = np.linalg.norm(embeddings[a_idx] - embeddings[p_idx])
        d_an = np.linalg.norm(embeddings[a_idx] - embeddings[n_idx])
        if d_ap < d_an:
            correct += 1
    return correct / n_samples


def alignment_uniformity(embeddings: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                          t: float = 2.0, max_pairs: int = 20000) -> Tuple[Optional[float], float]:
    """Wang & Isola (2020) 'Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere'. Alignment basso = i positivi (stessa
    classe) sono vicini tra loro. Uniformity molto negativa = lo spazio e' ben distribuito
    sull'ipersfera (niente collasso in una zona piccola)."""
    n = embeddings.shape[0]

    # Alignment: media di ||f(x)-f(y)||^2 su coppie positive (stessa classe)
    pos_pairs = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < 2:
            continue
        pairs = list(itertools.combinations(idx, 2))
        pos_pairs.extend(pairs)
    alignment = None
    if pos_pairs:
        if len(pos_pairs) > max_pairs:
            chosen = rng.choice(len(pos_pairs), size=max_pairs, replace=False)
            pos_pairs = [pos_pairs[i] for i in chosen]
        diffs = np.array([embeddings[i] - embeddings[j] for i, j in pos_pairs])
        alignment = float(np.mean(np.sum(diffs ** 2, axis=1)))

    # Uniformity: log( media di exp(-t * ||f(x)-f(y)||^2) ) su coppie casuali qualsiasi
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    diffs = embeddings[i_idx] - embeddings[j_idx]
    sq_dists = np.sum(diffs ** 2, axis=1)
    uniformity = float(np.log(np.mean(np.exp(-t * sq_dists)) + 1e-12))
    return alignment, uniformity


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank (Roy & Vetterli, 2007): entropia dei valori singolari normalizzati,
    esponenziata. Se e' molto minore di embedding_dim, la rete sta usando molte meno
    dimensioni di quante ne ha a disposizione (collasso dimensionale)."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def intra_inter_class_ratio(embeddings: np.ndarray, labels: np.ndarray) -> Optional[float]:
    dist = cosine_distance_matrix(embeddings)
    n = embeddings.shape[0]
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(n, k=1)
    same_pairs = same[iu]
    d_pairs = dist[iu]
    if not same_pairs.any() or same_pairs.all():
        return None
    intra = d_pairs[same_pairs].mean()
    inter = d_pairs[~same_pairs].mean()
    return float(intra / (inter + 1e-8))


def class_centroid_distance_matrix(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Centroide 'sferico' per classe (media poi ri-normalizzata) e matrice di distanza
    del coseno tra centroidi: dice quali classi rischiano di essere confuse dal k-NN."""
    classes = sorted(np.unique(labels).tolist())
    centroids = []
    for cls in classes:
        c = embeddings[labels == cls].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids.append(c)
    centroids = np.array(centroids)
    dist = cosine_distance_matrix(centroids)
    return classes, dist


# ---------------------------------------------------------------------- #
# Interpretabilita' per-dimensione
# ---------------------------------------------------------------------- #

def dimension_feature_correlation(embeddings: np.ndarray, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Matrice (embedding_dim, n_features) di correlazione di Pearson tra ciascuna
    dimensione dell'embedding e ciascuna feature geometrica nota."""
    feat_names = list(features.keys())
    feat_matrix = np.stack([features[name] for name in feat_names], axis=1)
    combined = np.concatenate([embeddings, feat_matrix], axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(combined, rowvar=False)
    n_dims = embeddings.shape[1]
    block = corr[:n_dims, n_dims:]
    block = np.nan_to_num(block, nan=0.0)
    return block, feat_names


def linear_probe_r2(embeddings: np.ndarray, features: Dict[str, np.ndarray],
                     corr_block: np.ndarray, feat_names: List[str], n_folds: int = 5) -> Dict[str, Dict[str, float]]:
    """Per ogni feature nota: R^2 di una Ridge regression 128-d -> feature (cross-validata,
    'quanto e' decodificabile linearmente usando TUTTO lo spazio') confrontato con l'R^2
    ottenibile usando SOLO la singola dimensione piu' correlata ('e' concentrata in un
    asse o distribuita su piu' dimensioni?')."""
    n = embeddings.shape[0]
    n_folds_eff = max(2, min(n_folds, n // 2)) if n >= 4 else 0
    results: Dict[str, Dict[str, float]] = {}

    for f_i, name in enumerate(feat_names):
        y = features[name]
        if np.std(y) < 1e-8 or n_folds_eff == 0:
            results[name] = {"r2_full": 0.0, "r2_best_dim": 0.0, "best_dim": -1}
            continue

        kf = KFold(n_splits=n_folds_eff, shuffle=True, random_state=42)
        try:
            r2_full = float(np.mean(cross_val_score(Ridge(alpha=1.0), embeddings, y, cv=kf, scoring="r2")))
        except ValueError:
            r2_full = 0.0

        best_dim = int(np.argmax(np.abs(corr_block[:, f_i])))
        x_single = embeddings[:, [best_dim]]
        try:
            r2_single = float(np.mean(cross_val_score(Ridge(alpha=1.0), x_single, y, cv=kf, scoring="r2")))
        except ValueError:
            r2_single = 0.0

        results[name] = {"r2_full": r2_full, "r2_best_dim": r2_single, "best_dim": best_dim}
    return results


# ---------------------------------------------------------------------- #
# Proiezioni 2D/3D
# ---------------------------------------------------------------------- #

def compute_projections(embeddings: np.ndarray, umap_neighbors: int, umap_min_dist: float,
                         seed: int) -> Dict[str, np.ndarray]:
    n = embeddings.shape[0]
    projections: Dict[str, np.ndarray] = {}

    pca = PCA(n_components=min(2, embeddings.shape[1]), random_state=seed)
    projections["PCA_2D"] = pca.fit_transform(embeddings)
    projections["_pca_explained_var"] = pca.explained_variance_ratio_

    pca_full = PCA(n_components=min(n, embeddings.shape[1]), random_state=seed)
    pca_full.fit(embeddings)
    projections["_pca_full_explained_var"] = pca_full.explained_variance_ratio_

    if HAS_UMAP and n >= 5:
        n_neighbors = max(2, min(umap_neighbors, n - 1))
        reducer_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=umap_min_dist,
                                metric="cosine", random_state=seed)
        projections["MAIN_2D"] = reducer_2d.fit_transform(embeddings)
        projections["main_method"] = "UMAP"

        reducer_3d = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=umap_min_dist,
                                metric="cosine", random_state=seed)
        projections["MAIN_3D"] = reducer_3d.fit_transform(embeddings)
    elif n >= 5:
        perplexity = max(2, min(30, n - 1))
        projections["MAIN_2D"] = TSNE(n_components=2, perplexity=perplexity, metric="cosine",
                                       init="pca", random_state=seed).fit_transform(embeddings)
        projections["main_method"] = "t-SNE (UMAP non installato)"
        pca3 = PCA(n_components=min(3, embeddings.shape[1]), random_state=seed)
        projections["MAIN_3D"] = pca3.fit_transform(embeddings)
    else:
        projections["MAIN_2D"] = projections["PCA_2D"]
        projections["main_method"] = "PCA (troppo pochi campioni per UMAP/t-SNE)"
        pca3 = PCA(n_components=min(3, embeddings.shape[1]), random_state=seed)
        projections["MAIN_3D"] = pca3.fit_transform(embeddings)

    return projections


# ---------------------------------------------------------------------- #
# Costruzione figure Plotly
# ---------------------------------------------------------------------- #

DISCRETE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d",
]


def fig_scatter_by_label(coords: np.ndarray, labels: np.ndarray, title: str, is_3d: bool = False) -> go.Figure:
    fig = go.Figure()
    classes = sorted(np.unique(labels).tolist())
    for i, cls in enumerate(classes):
        mask = labels == cls
        color = DISCRETE_COLORS[i % len(DISCRETE_COLORS)]
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers", name=cls, marker=dict(size=4, color=color, opacity=0.85),
                text=[cls] * int(mask.sum()), hovertemplate="%{text}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=coords[mask, 0], y=coords[mask, 1],
                mode="markers", name=cls, marker=dict(size=7, color=color, opacity=0.85, line=dict(width=0.5, color="white")),
                text=[cls] * int(mask.sum()), hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(title=title, template="plotly_white", legend_title="classe",
                       height=650 if is_3d else 550)
    return fig


def fig_scatter_by_source(coords: np.ndarray, source: np.ndarray, labels: np.ndarray, title: str) -> go.Figure:
    """Confronta dataset di training ed embedding enrollati nella stessa proiezione:
    utile per capire se il few-shot enrollment 'cade' nella nuvola giusta."""
    fig = go.Figure()
    for src, color, symbol in [("train", "#1f77b4", "circle"), ("enrolled", "#d62728", "diamond")]:
        mask = source == src
        if not mask.any():
            continue
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=src,
            marker=dict(size=8 if src == "enrolled" else 6, color=color, symbol=symbol, opacity=0.85),
            text=labels[mask], hovertemplate="%{text} (%{fullData.name})<extra></extra>",
        ))
    fig.update_layout(title=title, template="plotly_white", height=550)
    return fig


def fig_scatter_by_continuous(coords: np.ndarray, values: np.ndarray, labels: np.ndarray,
                               feature_name: str) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=coords[:, 0], y=coords[:, 1], mode="markers",
        marker=dict(size=7, color=values, colorscale="Viridis", showscale=True,
                    colorbar=dict(title=feature_name), line=dict(width=0.5, color="white")),
        text=[f"{lbl}<br>{feature_name}={v:.2f}" for lbl, v in zip(labels, values)],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(title=f"Proiezione colorata per '{feature_name}'", template="plotly_white", height=480)
    return fig


def fig_explained_variance(explained_var: np.ndarray) -> go.Figure:
    cum = np.cumsum(explained_var)
    x = list(range(1, len(explained_var) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=explained_var, name="varianza per componente", marker_color="#9467bd", opacity=0.6))
    fig.add_trace(go.Scatter(x=x, y=cum, name="varianza cumulativa", yaxis="y2", line=dict(color="#d62728", width=2)))
    fig.update_layout(
        title="PCA: varianza spiegata (quante dimensioni 'servono' davvero?)",
        template="plotly_white", height=450,
        xaxis_title="componente principale",
        yaxis=dict(title="varianza spiegata"),
        yaxis2=dict(title="cumulativa", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fig_correlation_heatmap(corr_block: np.ndarray, feat_names: List[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_block.T, x=[f"dim_{i}" for i in range(corr_block.shape[0])], y=feat_names,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1, colorbar=dict(title="corr. Pearson"),
    ))
    fig.update_layout(
        title="Correlazione dimensione-embedding &harr; feature geometrica nota",
        template="plotly_white", height=350 + 20 * len(feat_names),
        xaxis=dict(showticklabels=False, title=f"128 dimensioni dell'embedding"),
    )
    return fig


def fig_r2_probe(probe_results: Dict[str, Dict[str, float]]) -> go.Figure:
    names = list(probe_results.keys())
    r2_full = [probe_results[n]["r2_full"] for n in names]
    r2_best = [probe_results[n]["r2_best_dim"] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=r2_full, name="R² con tutte le 128 dim (Ridge, CV)", marker_color="#2ca02c"))
    fig.add_trace(go.Bar(x=names, y=r2_best, name="R² con la SOLA dim piu' correlata", marker_color="#ff7f0e"))
    fig.update_layout(
        title="Quanto ogni feature nota e' decodificabile linearmente dall'embedding",
        template="plotly_white", barmode="group", height=450,
        yaxis=dict(title="R² (cross-validato)", range=[min(-0.1, min(r2_full + r2_best) - 0.05), 1.0]),
    )
    return fig


def fig_centroid_distance(classes: List[str], dist: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=dist, x=classes, y=classes, colorscale="Reds", zmin=0,
        colorbar=dict(title="dist. coseno"),
        text=np.round(dist, 3), texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig.update_layout(
        title="Distanza tra centroidi di classe (valori bassi = classi a rischio di confusione)",
        template="plotly_white", height=350 + 25 * len(classes),
    )
    return fig


def fig_recall_per_class(per_class: Dict[str, float], k: int) -> go.Figure:
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    names = [it[0] for it in items]
    values = [it[1] for it in items]
    colors = ["#d62728" if v < 0.7 else "#2ca02c" for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(
        title=f"Recall@{k} per classe (frazione dei {k} vicini piu' prossimi della classe corretta)",
        template="plotly_white", height=300 + 25 * len(names),
        xaxis=dict(title="recall", range=[0, 1.02]),
    )
    return fig


def fig_metrics_table(metrics: Dict[str, str]) -> go.Figure:
    fig = go.Figure(go.Table(
        header=dict(values=["Metrica", "Valore"], fill_color="#2c3e50", font=dict(color="white", size=13), align="left"),
        cells=dict(values=[list(metrics.keys()), list(metrics.values())], align="left", height=28),
    ))
    fig.update_layout(height=40 + 32 * len(metrics), margin=dict(t=10, b=10))
    return fig


# ---------------------------------------------------------------------- #
# Report HTML
# ---------------------------------------------------------------------- #

REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fafafa; }
h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }
h2 { margin-top: 48px; border-left: 5px solid #2c3e50; padding-left: 12px; }
.note { background: #eef3f8; border-left: 4px solid #2980b9; padding: 10px 16px; margin: 12px 0; font-size: 14px; }
.warn { background: #fdf3e7; border-left: 4px solid #e67e22; padding: 10px 16px; margin: 12px 0; font-size: 14px; }
.meta { color: #666; font-size: 13px; }
.section { background: white; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
"""


def build_report(sections: List[Tuple[str, str, str]], meta: Dict[str, str], output_path: str) -> None:
    """sections: lista di (titolo_h2, nota_html_opzionale, html_figura)."""
    parts = [
        "<html><head><meta charset='utf-8'>",
        "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>",
        f"<style>{REPORT_CSS}</style></head><body>",
        "<h1>Analisi dello spazio di embedding &mdash; Gesture Control Engine</h1>",
        f"<p class='meta'>Generato il {meta['timestamp']} &middot; dataset: {meta['dataset']} "
        f"&middot; {meta['n_samples']} campioni, {meta['n_classes']} classi &middot; "
        f"checkpoint: {meta['checkpoint']} &middot; proiezione principale: {meta['main_method']}</p>",
    ]
    for title, note, fig_html in sections:
        parts.append(f"<div class='section'><h2>{title}</h2>")
        if note:
            parts.append(note)
        parts.append(fig_html)
        parts.append("</div>")
    parts.append("</body></html>")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Analisi diagnostica dello spazio di embedding (report Plotly).")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--enrolled-db", type=str, default=DEFAULT_ENROLLED_DB_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--k", type=int, default=5, help="k per Recall@K (default: allineato a KNNConfig.k)")
    parser.add_argument("--umap-neighbors", type=int, default=10)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--n-triplets", type=int, default=5000, help="numero di triplette campionate per Triplet Accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open", action="store_true", help="apre il report nel browser al termine")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = EngineConfig.load()

    print(f"Carico dataset da {args.dataset} ...")
    vectors, labels = load_dataset(args.dataset)
    print(f"{len(vectors)} campioni, {len(set(labels.tolist()))} classi.")

    if vectors.shape[1] != cfg.embedding.input_dim:
        raise ValueError(
            f"Il dataset contiene vettori a {vectors.shape[1]}-d ma EmbeddingConfig.input_dim="
            f"{cfg.embedding.input_dim}. Dataset raccolto con una pipeline di feature diversa: "
            f"non e' analizzabile con la rete attuale."
        )

    print("Carico il modello addestrato ...")
    model = load_model(cfg)
    embeddings = embed_all(model, vectors)

    print("Ricalcolo le feature geometriche note per l'analisi di interpretabilita' ...")
    features = compute_known_features(vectors, cfg.rule_based.finger_extended_ratio)

    print("Calcolo le metriche quantitative ...")
    recall_macro, recall_per_class = recall_at_k(embeddings, labels, args.k)
    trip_acc = triplet_accuracy(embeddings, labels, args.n_triplets, rng)
    alignment, uniformity = alignment_uniformity(embeddings, labels, rng)
    eff_rank = effective_rank(embeddings)
    intra_inter = intra_inter_class_ratio(embeddings, labels)
    try:
        sil = silhouette_score(embeddings, labels, metric="cosine") if len(set(labels.tolist())) >= 2 else None
    except ValueError:
        sil = None
    classes_sorted, centroid_dist = class_centroid_distance_matrix(embeddings, labels)

    print("Calcolo correlazioni dimensione-feature e linear probing ...")
    corr_block, feat_names = dimension_feature_correlation(embeddings, features)
    probe_results = linear_probe_r2(embeddings, features, corr_block, feat_names)

    print("Calcolo proiezioni 2D/3D (puo' richiedere qualche secondo) ..." + ("" if HAS_UMAP else " [UMAP non installato, uso t-SNE/PCA]"))
    projections = compute_projections(embeddings, args.umap_neighbors, args.umap_min_dist, args.seed)

    enrolled = load_enrolled_db(args.enrolled_db)

    # ------------------------------------------------------------------ #
    # Costruzione sezioni del report
    # ------------------------------------------------------------------ #
    sections: List[Tuple[str, str, str]] = []

    metrics_display = {
        "Campioni / classi": f"{len(vectors)} / {len(classes_sorted)}",
        "Recall@%d (macro, media su tutte le classi)" % args.k: f"{recall_macro:.3f}",
        "Triplet Accuracy (d(a,p) < d(a,n) su triplette casuali)": f"{trip_acc:.3f}" if trip_acc is not None else "n/d (< 2 classi)",
        "Silhouette score (distanza coseno)": f"{sil:.3f}" if sil is not None else "n/d",
        "Alignment (positivi vicini; piu' basso = meglio)": f"{alignment:.4f}" if alignment is not None else "n/d",
        "Uniformity (spazio ben distribuito; piu' negativo = meglio)": f"{uniformity:.4f}",
        "Effective rank (Roy-Vetterli, max = %d)" % embeddings.shape[1]: f"{eff_rank:.1f}",
        "Rapporto distanza intra/inter classe (< 1 = classi separate)": f"{intra_inter:.3f}" if intra_inter is not None else "n/d",
    }
    sections.append((
        "1. Cruscotto metriche",
        "<div class='note'>Recall@K e Triplet Accuracy dicono direttamente \"quanto bene funzionerebbe il "
        "k-NN a runtime\". Alignment/Uniformity (Wang &amp; Isola, 2020) diagnosticano la qualita' dello spazio "
        "ipersferico L2-normalizzato che questa rete produce. L'Effective Rank confrontato con embedding_dim=128 "
        "dice quante dimensioni la rete sta davvero usando: se e' molto basso (es. &lt;20) con pochi campioni per "
        "classe e' normale (collasso dimensionale tipico del few-shot), ma se resta basso anche aumentando i dati "
        "vale la pena rivedere l'architettura o l'aggiunta di piu' classi/varieta'.</div>",
        fig_metrics_table(metrics_display).to_html(full_html=False, include_plotlyjs=False),
    ))

    # Proiezione principale colorata per classe
    main_note = (
        f"<div class='note'>Proiezione calcolata con <b>{projections['main_method']}</b>. "
        "Cluster ben separati e compatti per colore = la rete distingue bene quelle classi. "
        "Cluster sovrapposti o allungati che si toccano = rischio di confusione reale nel k-NN "
        "(controlla anche la matrice di distanza tra centroidi piu' sotto).</div>"
    )
    sections.append((
        "2. Proiezione 2D colorata per classe",
        main_note,
        fig_scatter_by_label(projections["MAIN_2D"], labels, "Spazio di embedding (2D) per classe").to_html(
            full_html=False, include_plotlyjs=False),
    ))
    sections.append((
        "2b. Proiezione 3D colorata per classe (trascina per ruotare)",
        "",
        fig_scatter_by_label(projections["MAIN_3D"], labels, "Spazio di embedding (3D) per classe", is_3d=True).to_html(
            full_html=False, include_plotlyjs=False),
    ))

    # Proiezione per feature continue note: capire COSA identifica la rete
    continuous_feats = ["roll_deg", "pitch_deg", "yaw_deg", "avg_curl", "pinch_thumb_index", "finger_count"]
    feat_figs_html = "".join(
        fig_scatter_by_continuous(projections["MAIN_2D"], features[name], labels, name).to_html(
            full_html=False, include_plotlyjs=False)
        for name in continuous_feats
    )
    sections.append((
        "3. Stessa proiezione, colorata per feature geometrica nota",
        "<div class='note'>Se una nuvola di punti forma un <b>gradiente pulito</b> di colore lungo un asse "
        "(indipendentemente dalla forma dei cluster di classe), la rete ha isolato quella feature come "
        "direzione quasi indipendente nello spazio &mdash; separarla in futuro con una nuova classe binaria "
        "dovrebbe funzionare bene. Se il colore invece si mescola in modo casuale dentro ogni cluster di "
        "classe, quella feature <b>non</b> e' ben rappresentata nello spazio (o e' ridondante con la "
        "forma/label stessa). Confronta con la sezione 5 (correlazione per-dimensione) per numeri precisi.</div>",
        feat_figs_html,
    ))

    # Varianza spiegata / effective dimensionality
    sections.append((
        "4. Dimensionalita' effettiva dello spazio (PCA)",
        "<div class='note'>Se poche componenti principali spiegano gia' la maggior parte della varianza, "
        "la rete sta usando uno spazio effettivo molto piu' piccolo di 128-d: non e' necessariamente un "
        "problema (anzi e' atteso con pochi campioni/classi), ma indica quanto margine hai prima di "
        "\"esaurire\" la capacita' rappresentativa attuale aggiungendo nuove classi molto simili tra loro.</div>",
        fig_explained_variance(projections["_pca_full_explained_var"]).to_html(full_html=False, include_plotlyjs=False),
    ))

    # Correlazione dimensione <-> feature nota
    sections.append((
        "5. Cosa rappresenta ogni dimensione: correlazione con feature note",
        "<div class='note'>Ogni riga e' una feature geometrica nota (angoli di orientamento in gradi, "
        "curl medio, distanza di pinch, conteggio dita estese); ogni colonna e' una delle 128 dimensioni "
        "dell'embedding. Rosso/blu intenso = quella dimensione codifica fortemente (positivamente/"
        "negativamente) quella feature. Se una feature non ha NESSUNA cella intensa, non e' concentrata "
        "in un asse singolo: guarda la sezione 6 per capire se e' comunque decodificabile in modo "
        "distribuito su piu' dimensioni.</div>",
        fig_correlation_heatmap(corr_block, feat_names).to_html(full_html=False, include_plotlyjs=False),
    ))

    # Linear probing R^2
    probe_table_rows = "".join(
        f"<tr><td>{name}</td><td>{res['r2_full']:.3f}</td><td>{res['r2_best_dim']:.3f}</td>"
        f"<td>dim_{res['best_dim']}</td></tr>"
        for name, res in probe_results.items()
    )
    probe_note = (
        "<div class='note'>R² vicino a 1 con <b>tutte le 128 dim</b> ma basso con la <b>sola dimensione "
        "migliore</b> = feature codificata in modo <i>distribuito</i> (normale e sano). R² alto anche con "
        "una sola dimensione = feature isolata quasi su un asse dedicato (facilissima da separare in futuro). "
        "R² basso in entrambi i casi = la rete non ha ancora imparato bene quella feature: se e' importante "
        "per te, considera di aggiungere campioni/classi che la variano esplicitamente in training.</div>"
        f"<table style='width:100%;border-collapse:collapse' border='1' cellpadding='6'>"
        f"<tr><th>Feature</th><th>R² (128 dim, CV)</th><th>R² (1 dim migliore)</th><th>Dimensione migliore</th></tr>"
        f"{probe_table_rows}</table>"
    )
    sections.append((
        "6. Linear probing: la feature e' decodificabile linearmente?",
        probe_note,
        fig_r2_probe(probe_results).to_html(full_html=False, include_plotlyjs=False),
    ))

    # Confusione tra classi
    sections.append((
        "7. Rischio di confusione tra classi",
        "<div class='note'>Distanza del coseno tra i centroidi di ciascuna classe. Valori bassi (celle "
        "chiare) tra due classi diverse indicano che i loro embedding medi sono vicini: sono le coppie "
        "piu' a rischio se aggiungi varianti o rumore. La diagonale e' sempre 0 (una classe rispetto a se' "
        "stessa).</div>",
        fig_centroid_distance(classes_sorted, centroid_dist).to_html(full_html=False, include_plotlyjs=False),
    ))

    # Recall per classe
    if recall_per_class:
        sections.append((
            f"8. Recall@{args.k} per singola classe",
            "<div class='note'>Classi con recall basso (rosso, sotto 0.70) sono quelle dove i vicini piu' "
            "prossimi nel k-NN spesso NON sono della classe corretta: candidate primarie per raccogliere "
            "piu' campioni o rivedere se si confondono con una classe geometricamente simile (vedi sezione "
            "7).</div>",
            fig_recall_per_class(recall_per_class, args.k).to_html(full_html=False, include_plotlyjs=False),
        ))

    # Confronto con il database enrollato (se presente)
    if enrolled is not None:
        enrolled_emb, enrolled_labels = enrolled
        combined_emb = np.concatenate([embeddings, enrolled_emb], axis=0)
        combined_labels = np.concatenate([labels, enrolled_labels])
        source = np.array(["train"] * len(labels) + ["enrolled"] * len(enrolled_labels))
        combined_proj = compute_projections(combined_emb, args.umap_neighbors, args.umap_min_dist, args.seed)

        sections.append((
            "9. Confronto con le gesture enrollate (few-shot)",
            "<div class='note'>I diamanti rossi sono campioni enrollati dinamicamente dall'utente (senza "
            "retraining della rete); i cerchi blu sono i campioni di training. Se un diamante rosso cade "
            "chiaramente FUORI da ogni nuvola blu, quella gesture enrollata rischia di essere instabile o "
            "poco rappresentativa (pochi campioni, angolazione anomala) — considera di ri-enrollarla con "
            "piu' varieta'.</div>"
            + fig_scatter_by_source(combined_proj["MAIN_2D"], source, combined_labels,
                                     "Training vs Enrollment, stessa proiezione").to_html(
                full_html=False, include_plotlyjs=False)
            + fig_scatter_by_label(combined_proj["MAIN_2D"], combined_labels,
                                    "Training + Enrollment colorati per classe").to_html(
                full_html=False, include_plotlyjs=False),
            ""
        ))
    else:
        sections.append((
            "9. Confronto con le gesture enrollate (few-shot)",
            f"<div class='warn'>Nessun database enrollato trovato in '{args.enrolled_db}'. Questa sezione "
            "confronta gli embedding raccolti dinamicamente dalla UI con quelli di training, per capire se "
            "il few-shot enrollment cade nella nuvola giusta: sara' popolata non appena enrolli almeno una "
            "gesture custom.</div>",
            "",
        ))

    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": os.path.relpath(args.dataset, PROJECT_ROOT),
        "n_samples": str(len(vectors)),
        "n_classes": str(len(classes_sorted)),
        "checkpoint": cfg.embedding.torch_weights_path,
        "main_method": projections["main_method"],
    }
    build_report(sections, meta, args.output)
    print(f"\nReport salvato in: {args.output}")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
