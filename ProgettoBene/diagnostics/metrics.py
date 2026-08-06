"""
diagnostics/metrics.py
=======================
Metriche quantitative per valutare la qualita' dello spazio di embedding
prodotto da GestureEmbeddingNet (128-d, L2-normalizzato, distanza del
coseno). Tutte le funzioni lavorano su array numpy semplici, cosi' da poter
essere usate sia da scripts/analyze_embeddings.py sia da
scripts/probe_new_gesture.py, sia da eventuali test.

Riferimenti allo stato dell'arte:
  - Recall@K / Precision@K: standard nel deep metric learning (FaceNet e derivati).
  - Alignment & Uniformity: Wang & Isola 2020, "Understanding Contrastive
    Representation Learning through Alignment and Uniformity on the
    Hypersphere". Pensata esattamente per embedding L2-normalizzati come i
    nostri.
  - Effective rank / participation ratio: diagnosi del "collasso dimensionale",
    fenomeno ben documentato nel contrastive/metric learning con dataset
    piccoli (few-shot), che e' esattamente il nostro caso.
  - Linear probing: Alain & Bengio 2016, "Understanding intermediate layers
    using linear classifier probes".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Utility di base
# --------------------------------------------------------------------------- #
def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """(N, D) L2-normalizzati -> (N, N) matrice di distanza del coseno (0=identici, 2=opposti)."""
    emb = embeddings.astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    emb = emb / norms
    sim = emb @ emb.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


# --------------------------------------------------------------------------- #
# Recall@K
# --------------------------------------------------------------------------- #
def recall_at_k(embeddings: np.ndarray, labels: Sequence[str], k_values: Sequence[int] = (1, 3, 5)) -> Dict[int, float]:
    """
    Per ogni campione (leave-one-out), guarda i k vicini piu' prossimi (esclude
    se stesso) e verifica se la classe di maggioranza tra essi coincide con la
    classe vera. E' la metrica piu' diretta di "quanto bene funzionerebbe il
    k-NN in produzione" sul dataset corrente.
    """
    labels = np.asarray(labels)
    dist = cosine_distance_matrix(embeddings)
    np.fill_diagonal(dist, np.inf)
    n = len(labels)
    results = {}
    for k in k_values:
        correct = 0
        for i in range(n):
            k_eff = min(k, n - 1)
            nn_idx = np.argsort(dist[i])[:k_eff]
            nn_labels = labels[nn_idx]
            vals, counts = np.unique(nn_labels, return_counts=True)
            majority = vals[np.argmax(counts)]
            if majority == labels[i]:
                correct += 1
        results[k] = correct / n
    return results


# --------------------------------------------------------------------------- #
# Triplet accuracy
# --------------------------------------------------------------------------- #
def triplet_accuracy(
    embeddings: np.ndarray,
    labels: Sequence[str],
    n_triplets: int = 5000,
    seed: int = 0,
) -> float:
    """% di triplette casuali (anchor, positive, negative) per cui
    d(a,p) < d(a,n), ossia la condizione che la Triplet Loss ha ottimizzato.
    E' la misura piu' diretta di "la loss di training ha funzionato?"."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    dist = cosine_distance_matrix(embeddings)

    by_class: Dict[str, List[int]] = {}
    for i, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(i)
    classes = [c for c, idxs in by_class.items() if len(idxs) >= 2]
    if len(classes) < 2:
        raise ValueError("Servono almeno 2 classi con >=2 campioni per calcolare la triplet accuracy.")

    correct = 0
    total = 0
    for _ in range(n_triplets):
        c = rng.choice(classes)
        a, p = rng.choice(by_class[c], size=2, replace=False)
        neg_c = rng.choice([x for x in classes if x != c] or [c])
        n = rng.choice(by_class[neg_c])
        if dist[a, p] < dist[a, n]:
            correct += 1
        total += 1
    return correct / total


# --------------------------------------------------------------------------- #
# Intra/inter class distance ratio (Fisher-like)
# --------------------------------------------------------------------------- #
def intra_inter_class_ratio(embeddings: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Rapporto (distanza media intra-classe) / (distanza media inter-classe).
    Piu' basso e' meglio (classi compatte e ben separate). Un valore vicino
    a 1 significa che la rete non sta separando le classi meglio del caso
    a caso; un valore molto sopra 1 indica classi piu' sparse al loro interno
    che rispetto ad altre classi (situazione patologica)."""
    labels = np.asarray(labels)
    dist = cosine_distance_matrix(embeddings)
    n = len(labels)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(dist[i, j])
    mean_intra = float(np.mean(intra)) if intra else float("nan")
    mean_inter = float(np.mean(inter)) if inter else float("nan")
    ratio = mean_intra / mean_inter if mean_inter > 1e-12 else float("nan")
    return {"mean_intra_class_dist": mean_intra, "mean_inter_class_dist": mean_inter, "intra_inter_ratio": ratio}


# --------------------------------------------------------------------------- #
# Silhouette score (nello spazio 128-d, metrica coseno — MAI sulla proiezione 2D)
# --------------------------------------------------------------------------- #
def silhouette(embeddings: np.ndarray, labels: Sequence[str]) -> float:
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(embeddings, labels, metric="cosine"))


def silhouette_per_class(embeddings: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Silhouette score medio per singola classe: utile per capire QUALI
    classi sono confuse con altre, non solo un numero globale."""
    from sklearn.metrics import silhouette_samples

    labels_arr = np.asarray(labels)
    samples = silhouette_samples(embeddings, labels_arr, metric="cosine")
    out = {}
    for c in sorted(set(labels_arr.tolist())):
        out[c] = float(np.mean(samples[labels_arr == c]))
    return out


# --------------------------------------------------------------------------- #
# Alignment & Uniformity (Wang & Isola, 2020)
# --------------------------------------------------------------------------- #
def alignment(embeddings: np.ndarray, labels: Sequence[str], alpha: float = 2.0) -> float:
    """Media, su tutte le coppie positive (stessa classe), di ||e_i - e_j||^alpha.
    Piu' basso = i positivi sono piu' vicini tra loro (bene)."""
    labels = np.asarray(labels)
    vals = []
    for c in set(labels.tolist()):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sub = embeddings[idx]
        for i in range(len(sub)):
            diff = sub[i + 1:] - sub[i]
            vals.append(np.sum(diff ** alpha if alpha == 2 else np.abs(diff) ** alpha, axis=1))
    if not vals:
        return float("nan")
    return float(np.mean(np.concatenate(vals)))


def uniformity(embeddings: np.ndarray, t: float = 2.0) -> float:
    """log( media_su_tutte_le_coppie( exp(-t * ||e_i - e_j||^2) ) ).
    Piu' negativo/basso = embedding piu' uniformemente distribuiti
    sull'ipersfera (spazio usato in modo efficiente, niente collasso in
    una zona piccola). Vicino a 0 = quasi tutti gli embedding sono
    ammassati in un unico punto."""
    n = embeddings.shape[0]
    # a blocchi per non esplodere in memoria con dataset grandi
    block = 512
    total, count = 0.0, 0
    for start in range(0, n, block):
        chunk = embeddings[start:start + block]
        sq = np.sum((chunk[:, None, :] - embeddings[None, :, :]) ** 2, axis=-1)
        # escludi la diagonale globale (i == j) quando il blocco la contiene
        for local_i, global_i in enumerate(range(start, min(start + block, n))):
            sq[local_i, global_i] = np.inf
        vals = np.exp(-t * sq)
        vals = vals[np.isfinite(sq)]
        total += vals.sum()
        count += vals.size
    return float(np.log(total / count))


# --------------------------------------------------------------------------- #
# Effective rank / collasso dimensionale
# --------------------------------------------------------------------------- #
def effective_rank(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Quante delle D dimensioni dell'embedding sono REALMENTE usate.
    Calcola l'entropia di Shannon della distribuzione normalizzata degli
    autovalori della matrice di covarianza (Roy & Vetterli, 2007) e il
    participation ratio (piu' semplice, piu' sensibile agli outlier).
    Un effective_rank molto minore di D (es. 6 su 128) indica un forte
    collasso dimensionale: la rete sta di fatto usando uno spazio molto
    piu' piccolo di quanto la dimensione nominale suggerisca. Non e'
    necessariamente un problema (anzi e' atteso con poche classi/pochi
    campioni), ma va tenuto d'occhio: aggiungere una nuova feature/classe
    che richiede una direzione "nuova" nello spazio potrebbe non trovare
    capacita' libera se il collasso e' troppo estremo.
    """
    cov = np.cov(embeddings.T)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = eigvals.sum()
    if total < 1e-12:
        return {"effective_rank_entropy": 0.0, "participation_ratio": 0.0, "embedding_dim": embeddings.shape[1]}
    p = eigvals / total
    p_nonzero = p[p > 1e-15]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    eff_rank_entropy = float(np.exp(entropy))
    participation_ratio = float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))
    return {
        "effective_rank_entropy": eff_rank_entropy,
        "participation_ratio": participation_ratio,
        "embedding_dim": int(embeddings.shape[1]),
    }


# --------------------------------------------------------------------------- #
# Linear probing: dimensione (o intero embedding) -> feature nota
# --------------------------------------------------------------------------- #
def linear_probe_regression(embeddings: np.ndarray, feature: np.ndarray, cv: int = 5) -> float:
    """R^2 medio (cross-validated) di una regressione lineare
    embedding(128-d) -> feature continua nota (es. yaw, pinch_distance).
    R^2 alto = quella feature e' linearmente ricostruibile dall'embedding
    (la rete l'ha isolata come informazione accessibile linearmente,
    coerente con cio' che poi usera' il k-NN a coseno)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, embeddings, feature, cv=cv, scoring="r2")
    return float(np.mean(scores))


def linear_probe_classification(embeddings: np.ndarray, labels: Sequence[str], cv: int = 5) -> float:
    """Accuracy media (cross-validated) di una regressione logistica lineare
    embedding -> classe. E' un test di sanita': se questo e' basso (< 90%)
    mentre il modello sembra funzionare bene col k-NN, qualcosa non torna
    (di solito vuol dire dataset troppo piccolo/sbilanciato)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model = LogisticRegression(max_iter=3000)
    scores = cross_val_score(model, embeddings, labels, cv=cv)
    return float(np.mean(scores))


def per_dimension_correlation(embeddings: np.ndarray, features: Dict[str, np.ndarray]) -> "np.ndarray":
    """Matrice (n_features, embedding_dim) di correlazione di Spearman tra
    ogni feature nota e ogni singola dimensione dell'embedding. Economica,
    usala come prima passata prima del linear probing (piu' costoso ma
    cattura anche combinazioni lineari di piu' dimensioni)."""
    from scipy.stats import spearmanr

    feat_names = list(features.keys())
    D = embeddings.shape[1]
    corr = np.zeros((len(feat_names), D), dtype=np.float64)
    for fi, fname in enumerate(feat_names):
        fvals = features[fname]
        for d in range(D):
            r, _ = spearmanr(embeddings[:, d], fvals)
            corr[fi, d] = 0.0 if np.isnan(r) else r
    return corr, feat_names


# --------------------------------------------------------------------------- #
# Separabilita' lineare di una feature/classe candidata (per probe_new_gesture.py)
# --------------------------------------------------------------------------- #
def linear_separability(embeddings_a: np.ndarray, embeddings_b: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """
    Dati due gruppi di embedding (es. candidata nuova classe vs classe
    esistente), stima quanto sono separabili con un confine LINEARE
    (SVM lineare, cross-validated). accuracy ~0.5 = per il classificatore
    lineare sono indistinguibili (rischio concreto di confusione nel k-NN
    a coseno, che e' anch'esso in pratica un confine quasi-lineare a livello
    locale). accuracy vicina a 1.0 = ben separabili.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    X = np.concatenate([embeddings_a, embeddings_b], axis=0)
    y = np.concatenate([np.zeros(len(embeddings_a)), np.ones(len(embeddings_b))])
    n_min_class = min(len(embeddings_a), len(embeddings_b))
    cv_eff = max(2, min(cv, n_min_class))
    model = SVC(kernel="linear")
    scores = cross_val_score(model, X, y, cv=cv_eff)
    return {"linear_svm_accuracy": float(np.mean(scores)), "cv_folds_used": cv_eff}


# --------------------------------------------------------------------------- #
# Report aggregato
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingQualityReport:
    n_samples: int
    n_classes: int
    recall_at_k: Dict[int, float]
    triplet_accuracy: float
    intra_inter: Dict[str, float]
    silhouette_global: float
    silhouette_per_class: Dict[str, float]
    alignment: float
    uniformity: float
    effective_rank: Dict[str, float]
    class_linear_probe_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        print("=" * 70)
        print("REPORT QUALITA' EMBEDDING")
        print("=" * 70)
        print(f"Campioni: {self.n_samples}   Classi: {self.n_classes}")
        print()
        for k, v in self.recall_at_k.items():
            print(f"Recall@{k}:                     {v:.4f}")
        print(f"Triplet accuracy:              {self.triplet_accuracy:.4f}")
        print(f"Intra-class dist (media):      {self.intra_inter['mean_intra_class_dist']:.4f}")
        print(f"Inter-class dist (media):      {self.intra_inter['mean_inter_class_dist']:.4f}")
        print(f"Rapporto intra/inter:          {self.intra_inter['intra_inter_ratio']:.4f}  (piu' basso = meglio)")
        print(f"Silhouette globale (coseno):   {self.silhouette_global:.4f}  (range -1..1, piu' alto = meglio)")
        print(f"Alignment (Wang&Isola):        {self.alignment:.4f}  (piu' basso = positivi piu' vicini)")
        print(f"Uniformity (Wang&Isola):       {self.uniformity:.4f}  (piu' negativo = spazio usato meglio)")
        d = self.effective_rank["embedding_dim"]
        er = self.effective_rank["effective_rank_entropy"]
        pr = self.effective_rank["participation_ratio"]
        print(f"Effective rank (entropia):     {er:.2f} / {d}  ({100*er/d:.1f}% delle dimensioni nominali)")
        print(f"Participation ratio:           {pr:.2f} / {d}")
        print(f"Linear-probe class accuracy:   {self.class_linear_probe_accuracy:.4f}")
        print()
        print("Silhouette per classe (classi piu' problematiche in cima):")
        for c, s in sorted(self.silhouette_per_class.items(), key=lambda kv: kv[1]):
            flag = "  <-- attenzione, poco separata" if s < 0.3 else ""
            print(f"  {c:20s} {s:+.4f}{flag}")
        print("=" * 70)


def compute_full_report(embeddings: np.ndarray, labels: Sequence[str], n_triplets: int = 5000) -> EmbeddingQualityReport:
    labels = np.asarray(labels)
    return EmbeddingQualityReport(
        n_samples=int(embeddings.shape[0]),
        n_classes=int(len(set(labels.tolist()))),
        recall_at_k=recall_at_k(embeddings, labels),
        triplet_accuracy=triplet_accuracy(embeddings, labels, n_triplets=n_triplets),
        intra_inter=intra_inter_class_ratio(embeddings, labels),
        silhouette_global=silhouette(embeddings, labels),
        silhouette_per_class=silhouette_per_class(embeddings, labels),
        alignment=alignment(embeddings, labels),
        uniformity=uniformity(embeddings),
        effective_rank=effective_rank(embeddings),
        class_linear_probe_accuracy=linear_probe_classification(embeddings, labels),
    )
