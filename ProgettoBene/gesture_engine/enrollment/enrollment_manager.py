"""
enrollment_manager.py
======================
Few-Shot Dynamic Enrollment: permette di registrare una nuova gesture
(es. simbolo custom, segno della lingua dei segni) raccogliendo 5-10
campioni dal vivo dalla webcam, calcolandone gli embedding con la rete
gia' addestrata/esportata, e aggiornando l'indice k-NN al volo — senza
alcun retraining della rete neurale.

Flusso tipico (vedi anche ui/app.py per l'interfaccia grafica):

    manager = EnrollmentManager(embedding_engine, knn_classifier, knn_config)
    manager.start_session("mio_simbolo_custom")
    for ogni frame in cui l'utente mostra la gesture:
        manager.capture_sample(embedding_input)  # da build_embedding_vector(), stessa pipeline del training
    manager.finish_session()   # salva su disco automaticamente
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class EnrollmentSession:
    label: str
    collected_embeddings: List[np.ndarray] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.collected_embeddings)


class EnrollmentManager:
    def __init__(self, embedding_engine, knn_classifier, knn_config):
        self._embedding_engine = embedding_engine
        self._knn = knn_classifier
        self.cfg = knn_config
        self._session: Optional[EnrollmentSession] = None

    # ------------------------------------------------------------------ #
    @property
    def is_active(self) -> bool:
        return self._session is not None

    @property
    def current_count(self) -> int:
        return self._session.num_samples if self._session else 0

    @property
    def current_label(self) -> Optional[str]:
        return self._session.label if self._session else None

    def start_session(self, label: str) -> None:
        if not label or not label.strip():
            raise ValueError("L'etichetta della gesture non puo' essere vuota.")
        self._session = EnrollmentSession(label=label.strip())

    def capture_sample(self, embedding_input: np.ndarray) -> int:
        """Calcola l'embedding del campione corrente e lo aggiunge alla sessione.
        `embedding_input` deve provenire da build_embedding_vector() (stessa pipeline
        di feature usata in training/collect_training_data.py), non dai soli landmark
        normalizzati: incoerenze qui producono embedding non comparabili col database.
        Ritorna il numero totale di campioni raccolti finora nella sessione."""
        if self._session is None:
            raise RuntimeError("Nessuna sessione di enrollment attiva. Chiama start_session() prima.")
        embedding = self._embedding_engine.embed(embedding_input)
        self._session.collected_embeddings.append(embedding)
        return self._session.num_samples

    def is_ready_to_finish(self) -> bool:
        return self._session is not None and (
            self._session.num_samples >= self.cfg.min_samples_per_class_for_enrollment
        )

    def cancel_session(self) -> None:
        self._session = None

    def finish_session(self, persist: bool = True) -> str:
        """Aggiunge i campioni raccolti al database k-NN e (opzionalmente) li
        persiste su disco. Ritorna l'etichetta appena enrollata."""
        if self._session is None:
            raise RuntimeError("Nessuna sessione di enrollment attiva.")
        if self._session.num_samples == 0:
            raise RuntimeError("Nessun campione raccolto: impossibile completare l'enrollment.")

        embeddings = np.stack(self._session.collected_embeddings, axis=0)
        label = self._session.label
        self._knn.enroll(label, embeddings)
        if persist:
            self._knn.save()

        self._session = None
        return label
