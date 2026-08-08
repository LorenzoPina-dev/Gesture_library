"""
main.py
=======
Script principale per analizzare e visualizzare gli embedding partendo esclusivamente
dai file `.npz` forniti (embeddings_db.npz e training_dataset.npz).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import dei moduli diagnostici forniti
from diagnostics.known_features import extract_known_features
from diagnostics.metrics import compute_full_report, per_dimension_correlation
from diagnostics.projection import best_available_2d, pca_2d, pca_variance_explained


def load_data_from_npz():
    """Carica embedding, vettori di input e relative etichette direttamente
    dai file `.npz` disponibili nella directory di lavoro.
    """
    embeddings, labels_db = None, None
    vectors, labels_train = None, None

    # 1. Caricamento dagli embedding salvati nel database del k-NN (embeddings_db.npz)
    if Path("./data/enrolled_gestures/embeddings_db.npz").exists():
        print("[INFO] Caricamento da 'embeddings_db.npz'...")
        with np.load("./data/enrolled_gestures/embeddings_db.npz", allow_pickle=True) as data:
            embeddings = data["embeddings.npy"]
            labels_db = data["labels.npy"]
        print(f"  -> Trovati {embeddings.shape[0]} embedding normalizzati (dim: {embeddings.shape[1]}).")

    # 2. Caricamento dal dataset di addestramento (training_dataset.npz)
    if Path("./data/training_dataset.npz").exists():
        print("[INFO] Caricamento da 'training_dataset.npz'...")
        with np.load("./data/training_dataset.npz", allow_pickle=True) as data:
            vectors = data["vectors.npy"]
            labels_train = data["labels.npy"]
        print(f"  -> Trovati {vectors.shape[0]} vettori di input raw (dim: {vectors.shape[1]}).")

    if embeddings is None and vectors is None:
        raise FileNotFoundError(
            "Impossibile trovare 'embeddings_db.npz' o 'training_dataset.npz'. "
            "Assicurati che almeno uno dei due file sia presente nella cartella di esecuzione."
        )

    return embeddings, labels_db, vectors, labels_train


def plot_embeddings_2d(coords_2d: np.ndarray, labels: np.ndarray, method_name: str, title_suffix: str = ""):
    """Genera un grafico a dispersione 2D degli embedding colorati per classe."""
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 7))

    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            label=str(label),
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
            s=45,
        )

    plt.title(f"Spazio Embedding 2D - {method_name} {title_suffix}", fontsize=13, fontweight="bold")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend(title="Classi", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pca_variance(embeddings: np.ndarray):
    """Visualizza la varianza spiegata cumulativa tramite PCA[cite: 2]."""
    variance_ratio = pca_variance_explained(embeddings, n_components=10)
    cumulative_variance = np.cumsum(variance_ratio)

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.6, align="center", label="Varianza Singola")
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where="mid", label="Varianza Cumulativa", color="red")
    plt.ylabel("Varianza Spiegata")
    plt.xlabel("Componenti Principali")
    plt.title("Analisi della Varianza Spiegata (PCA)", fontweight="bold")
    plt.xticks(range(1, len(variance_ratio) + 1))
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Caricamento dati direttamente dagli archivi .npz
    embeddings_db, labels_db, vectors_train, labels_train = load_data_from_npz()

    # 2. Valutazione sul Database degli Embedding Enrolled (embeddings_db.npz)
    if embeddings_db is not None:
        print("\n" + "=" * 70)
        print(" ANALISI DATABASE EMBEDDINGS ENROLLED (embeddings_db.npz)")
        print("=" * 70)
        
        # Generazione e stampa del report di qualità numerica
        n_triplets = min(5000, len(embeddings_db) * 10)
        report_db = compute_full_report(embeddings_db, labels_db, n_triplets=n_triplets)
        report_db.print_summary()

        # Grafici di proiezione 2D[cite: 2]
        print("\nGenerazione proiezioni 2D per il database enrolled...")
        plot_pca_variance(embeddings_db)

        coords_best, method_name, _ = best_available_2d(embeddings_db)
        plot_embeddings_2d(coords_best, labels_db, method_name)

        coords_pca, _ = pca_2d(embeddings_db)
        plot_embeddings_2d(coords_pca, labels_db, "PCA")

    # 3. Analisi delle Feature Geometriche e Correlazioni (training_dataset.npz)
    if vectors_train is not None and vectors_train.shape[1] == 69:
        print("\n" + "=" * 70)
        print(" ANALISI FEATURE GEOMETRICHE RAW 69-D (training_dataset.npz)")
        print("=" * 70)
        
        # Estrazione delle feature fisiche note a partire dai 69-d[cite: 3]
        print("Estrazione feature geometriche (roll, pitch, yaw, estensioni dita, pinch)...")
        known_feats = extract_known_features(vectors_train)
        print(f"Estratte {len(known_feats)} feature continue su {vectors_train.shape[0]} campioni.")

        # Se abbiamo anche gli embedding normalizzati a 128-d associati, calcoliamo le correlazioni[cite: 1]
        if embeddings_db is not None and len(embeddings_db) == len(vectors_train):
            corr_matrix, feat_names = per_dimension_correlation(embeddings_db, known_feats)
            max_corrs = np.max(np.abs(corr_matrix), axis=1)

            print("\nCorrelazione di Spearman MAX (|r|) tra feature fisiche e le 128-d dell'embedding:")
            for fname, max_r in zip(feat_names, max_corrs):
                print(f"  - {fname:22s}: |r| max = {max_r:.4f}")
        else:
            print("\n[NOTE] La dimensione dei campioni in training_dataset.npz non coincide direttamente")
            print("       con embeddings_db.npz: il calcolo della correlazione per-dimensione è stato omesso.")


if __name__ == "__main__":
    main()