# Gesture Control Engine

Libreria Python **industry-level** per il tracciamento, il riconoscimento e la
gestione di eventi basati su gesture della mano in tempo reale via webcam.
Architettura modulare a pipeline, motore di riconoscimento a **3 livelli**
(regole geometriche, deep metric learning + k-NN open-set, FSM temporale),
Event Bus Pub-Sub e interfaccia grafica per gestire tutto senza toccare codice.

---

## 1. Installazione

```bash
cd progetto
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Modello MediaPipe HandLandmarker (obbligatorio)

Il modello `.task` non è distribuito via pip: va scaricato una volta sola e
posizionato in `models/hand_landmarker.task`:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Se manca, `HandLandmarkerEngine` lancia un errore esplicito con questo stesso link.

---

## 2. Avvio rapido

**Interfaccia grafica (consigliata, nessun codice richiesto):**

```bash
python ui/app.py
```

Dal pannello puoi: avviare/fermare il riconoscimento con preview live,
registrare nuove gesture custom (Few-Shot Enrollment) in pochi secondi,
gestire il database di gesture e modificare **tutti** i parametri del
sistema (soglie, filtri, cooldown, confidenze) con salvataggio persistente.

**Demo da linea di comando:**

```bash
python scripts/run_demo.py
```

**Uso come libreria in un tuo script:**

```python
from gesture_engine import GestureEngine

engine = GestureEngine()
engine.on("gesture.fist", lambda e: print("Pugno!"))
engine.on("gesture.pinch", lambda e: print("Pinch:", e.payload["pinch_strength"]))
engine.on("swipe.left", lambda e: print("Swipe sinistra"))

engine.register_sequence("boom_explosion", ["fist", "open_palm"])
engine.on("sequence.boom_explosion", lambda e: print("BOOM!"))

with engine:
    engine.run()   # finestra di preview, 'q'/ESC per uscire
```

Per integrare il motore nel *tuo* loop applicativo (es. dentro una tua UI o
un game engine) usa `.step()` invece di `.run()`:

```python
with GestureEngine() as engine:
    while running:
        results = engine.step()   # lista di HandFrameResult, uno per mano rilevata
        for r in results:
            print(r.rule_label, r.knn_label, r.knn_confidence)
```

---

## 3. Architettura della pipeline

```
Webcam (OpenCV)
   -> Preprocessing opzionale (CLAHE su canale L, LAB) — scarsa illuminazione
   -> MediaPipe HandLandmarker (RunningMode.VIDEO, timestamp sincronizzati)
   -> Filtro anti-jitter (EMA o One Euro Filter) sui landmark grezzi
   -> Normalizzazione geometrica (invariante a traslazione/scala/rotazione planare)
   -> Livello 1: regole euristiche (conteggio dita, pugno, pinch adattivo)
   -> Livello 2: GestureEmbeddingNet (ONNX) -> k-NN a distanza coseno
                  con soglia di rigetto open-set ("UNKNOWN")
   -> Livello 3: FSM temporale per sequenze dinamiche + rilevamento swipe
   -> Event Bus (Observer/Pub-Sub, multi-callback, cooldown/isteresi)
```

### Struttura del progetto

```
progetto/
├── gesture_engine/            libreria principale (package pip-installabile)
│   ├── config.py              configurazione centrale (dataclass + JSON persistente)
│   ├── pipeline.py            GestureEngine: orchestratore end-to-end
│   ├── ingestion/              camera, preprocessing CLAHE, wrapper MediaPipe
│   ├── normalization/          normalizzazione geometrica, filtri EMA/One-Euro
│   ├── recognition/            Livelli 1/2/3: rule-based, embedding+kNN, FSM
│   ├── events/                 EventBus Pub-Sub con cooldown
│   ├── enrollment/             Few-Shot Dynamic Enrollment
│   └── visualization/          overlay di debug (landmark, testo, FPS)
├── ui/app.py                  interfaccia grafica Tkinter (nessuna gestione codice)
├── scripts/
│   ├── run_demo.py             demo rapida da CLI
│   ├── collect_training_data.py  raccolta dataset etichettato per il training
│   ├── train_embedding_net.py    training con Triplet Loss
│   └── export_onnx.py            export ONNX per inferenza <1ms su CPU
├── models/                    hand_landmarker.task, gesture_embedding.{pt,onnx}
├── data/                      config persistita + database gesture enrollate
└── tests/                     14 test automatici (pytest) su normalizzazione,
                                riconoscimento, event bus, FSM
```

---

## 4. Il motore di riconoscimento a 3 livelli, in dettaglio

### Livello 1 — Regole euristiche (sempre attivo, latenza minima)
Conteggio dita estese, rilevamento pugno e pinch pollice-indice, tutto
calcolato su distanze euclidee **relative** alla dimensione della mano
(`gesture_engine/normalization/geometric.py`), quindi robusto a distanza
dalla webcam e dimensione della mano dell'utente.

### Livello 2 — Deep Metric Learning + k-NN open-set
`GestureEmbeddingNet` (MLP leggera, PyTorch) proietta il vettore di input a
69-d (63 valori di shape normalizzata, invariante a rotazione, + 6 di
orientamento del polso — vedi `build_embedding_vector` in
`gesture_engine/normalization/geometric.py`) in uno spazio a 128-d
L2-normalizzato, ottimizzato con **Triplet Loss**. Il classificatore k-NN (`EmbeddingKNNClassifier`) usa
distanza del coseno e **rifiuta** come `"UNKNOWN"` qualsiasi predizione la
cui distanza media dai k vicini superi `knn.max_cosine_distance` (default
0.30): questo azzera i falsi positivi su gesture mai viste.

**Il modello nella repo non è pre-addestrato** (nessun dataset è incluso).
Due percorsi possibili:

1. **Enrollment few-shot puro** (nessun training richiesto): registra 5-10
   campioni di una nuova gesture dalla UI (tab "Enrollment"). Funziona da
   subito perché la normalizzazione geometrica a monte fa già gran parte del
   lavoro di invarianza; è la modalità pensata per l'uso quotidiano.
2. **Training esplicito con Triplet Loss** (consigliato se vuoi embedding
   semanticamente più separati tra classi simili):
   ```bash
   python scripts/collect_training_data.py   # raccogli N campioni per M classi
   python scripts/train_embedding_net.py --epochs 200
   python scripts/export_onnx.py             # rigenera l'ONNX con i nuovi pesi
   ```

### Livello 3 — FSM temporale
`GestureSequenceFSM` compone sequenze discrete di etichette del Livello 1
entro una finestra temporale (`register_sequence`), es. pugno→palmo aperto
entro 1.5s = evento `"boom_explosion"`. `SwipeDetector` rileva movimenti
rapidi del polso e li classifica in `left/right/up/down` in base alla
velocità normalizzata.

### Event Bus (Livello 4)
Pattern Observer/Pub-Sub: `engine.on("gesture.fist", callback)` supporta
**più callback per lo stesso evento**. Ogni evento ha un cooldown
configurabile (`event_bus.default_cooldown_s`, default 0.4s) per evitare
trigger multipli su gesture mantenute nel tempo.

---

## 5. Prestazioni

- Inferenza dell'embedding network via **ONNX Runtime (CPU)**: misurata
  \<0.05ms per forward pass su questo ambiente di sviluppo (requisito \<1ms
  soddisfatto con ampio margine).
- Filtro **One Euro** di default: bassa latenza sui movimenti rapidi, alto
  smoothing quando la mano è ferma — elimina lo sfarfallio senza introdurre
  lag percepibile.
- Soglie di confidenza MediaPipe impostate a ≥0.65-0.70 di default per
  minimizzare falsi positivi da occlusioni/rumore.

---

## 6. Test

```bash
pip install pytest
python -m pytest tests/ -v
```

14 test coprono: invarianza geometrica (traslazione/scala/rotazione),
filtri anti-jitter, riconoscimento rule-based, rigetto open-set del k-NN,
Event Bus (multi-callback + cooldown), FSM di sequenza (successo/timeout),
rilevamento swipe.

---

## 7. Estendere il sistema

- **Nuova gesture custom**: tab "Enrollment" nella UI, oppure via codice:
  ```python
  engine.enrollment.start_session("mia_gesture")
  # per ogni frame utile (embedding_input viene da HandFrameResult.embedding_input,
  # cioe' build_embedding_vector(), la stessa pipeline usata in training):
  engine.enrollment.capture_sample(embedding_input)
  engine.enrollment.finish_session()  # salva automaticamente su disco
  ```
- **Nuova gesture sequenziale**: `engine.register_sequence("nome", ["fist", "open_palm"])`
- **Nuovo evento custom nel bus**: `engine.event_bus.emit("mio.evento", payload={...})`
  da qualunque punto della tua applicazione.
- Tutti i parametri sono in `gesture_engine/config.py` (dataclass) e
  persistiti in `data/engine_config.json`: modificabili dalla UI o a mano.
