"""
ui/app.py
=========
Interfaccia grafica (Tkinter, incluso in ogni installazione Python standard,
nessuna dipendenza aggiuntiva) per gestire l'intero Gesture Control Engine
senza dover modificare una sola riga di codice:

  - Tab "Live": avvia/ferma il riconoscimento con preview webcam e overlay
    di debug, mostra il log eventi emessi in tempo reale.
  - Tab "Enrollment": registra nuove gesture custom in pochi secondi
    (Few-Shot Dynamic Enrollment) e gestisce (elenca/elimina) quelle esistenti.
  - Tab "Impostazioni": modifica a runtime tutti i parametri di
    EngineConfig (soglie, filtri, cooldown, ecc.) con salvataggio su JSON.

Avvio:
    python ui/app.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gesture_engine.config import EngineConfig, DEFAULT_CONFIG_PATH
from gesture_engine.pipeline import GestureEngine
from gesture_engine.visualization import draw_hand, draw_debug_panel, landmarks_to_pixel_coords


class GestureEngineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Control Engine — Pannello di Controllo")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.cfg = EngineConfig.load(DEFAULT_CONFIG_PATH)
        self.engine: Optional[GestureEngine] = None
        self._engine_thread: Optional[threading.Thread] = None
        self._running = False
        self._enrollment_capturing = False
        self._enrollment_target_samples = 8

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Costruzione UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.live_tab = ttk.Frame(notebook)
        self.enroll_tab = ttk.Frame(notebook)
        self.settings_tab = ttk.Frame(notebook)

        notebook.add(self.live_tab, text="  Live  ")
        notebook.add(self.enroll_tab, text="  Enrollment Gesture  ")
        notebook.add(self.settings_tab, text="  Impostazioni  ")

        self._build_live_tab()
        self._build_enroll_tab()
        self._build_settings_tab()

    # ---------------------------- LIVE TAB ----------------------------- #
    def _build_live_tab(self) -> None:
        left = ttk.Frame(self.live_tab)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ttk.Label(left, text="Premi 'Avvia' per iniziare", anchor="center")
        self.video_label.pack(fill="both", expand=True)

        controls = ttk.Frame(self.live_tab)
        controls.pack(side="right", fill="y", padx=10, pady=10)

        ttk.Label(controls, text="Controlli", font=("Segoe UI", 12, "bold")).pack(pady=(0, 10))

        self.start_btn = ttk.Button(controls, text="▶  Avvia riconoscimento", command=self.start_engine)
        self.start_btn.pack(fill="x", pady=4)

        self.stop_btn = ttk.Button(controls, text="■  Ferma", command=self.stop_engine, state="disabled")
        self.stop_btn.pack(fill="x", pady=4)

        ttk.Separator(controls).pack(fill="x", pady=10)

        ttk.Label(controls, text="Log eventi", font=("Segoe UI", 10, "bold")).pack()
        self.event_log = tk.Listbox(controls, width=42, height=25)
        self.event_log.pack(fill="both", expand=True, pady=6)

        ttk.Button(controls, text="Svuota log", command=lambda: self.event_log.delete(0, tk.END)).pack(fill="x")

    def _log_event(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.event_log.insert(0, f"[{ts}] {text}")
        if self.event_log.size() > 300:
            self.event_log.delete(299, tk.END)

    # -------------------------- ENROLLMENT TAB -------------------------- #
    def _build_enroll_tab(self) -> None:
        top = ttk.Frame(self.enroll_tab)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Nome nuova gesture:").pack(side="left")
        self.new_gesture_name = tk.StringVar()
        ttk.Entry(top, textvariable=self.new_gesture_name, width=30).pack(side="left", padx=6)

        ttk.Label(top, text="Campioni da raccogliere:").pack(side="left", padx=(20, 4))
        self.sample_count_var = tk.IntVar(value=8)
        ttk.Spinbox(top, from_=5, to=30, textvariable=self.sample_count_var, width=5).pack(side="left")

        self.enroll_start_btn = ttk.Button(
            top, text="🖐  Inizia registrazione", command=self.start_enrollment
        )
        self.enroll_start_btn.pack(side="left", padx=20)

        self.enroll_status = ttk.Label(self.enroll_tab, text="Nessuna sessione attiva.", font=("Segoe UI", 11))
        self.enroll_status.pack(pady=10)

        self.enroll_progress = ttk.Progressbar(self.enroll_tab, length=400, maximum=100)
        self.enroll_progress.pack(pady=6)

        ttk.Separator(self.enroll_tab).pack(fill="x", pady=15)

        ttk.Label(self.enroll_tab, text="Gesture registrate", font=("Segoe UI", 12, "bold")).pack()

        list_frame = ttk.Frame(self.enroll_tab)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.gestures_listbox = tk.Listbox(list_frame, height=12)
        self.gestures_listbox.pack(side="left", fill="both", expand=True)

        btns = ttk.Frame(list_frame)
        btns.pack(side="left", padx=10)
        ttk.Button(btns, text="Aggiorna elenco", command=self.refresh_gesture_list).pack(fill="x", pady=4)
        ttk.Button(btns, text="Elimina selezionata", command=self.delete_selected_gesture).pack(fill="x", pady=4)

        self.refresh_gesture_list()

    def start_enrollment(self) -> None:
        if not self._running or self.engine is None:
            messagebox.showwarning("Motore non avviato", "Avvia prima il riconoscimento dal tab 'Live'.")
            return
        name = self.new_gesture_name.get().strip()
        if not name:
            messagebox.showwarning("Nome mancante", "Inserisci un nome per la nuova gesture.")
            return

        self._enrollment_target_samples = self.sample_count_var.get()
        self.engine.enrollment.start_session(name)
        self._enrollment_capturing = True
        self.enroll_progress["value"] = 0
        self.enroll_status.config(text=f"Registrazione '{name}': mostra la gesture alla webcam...")

    def refresh_gesture_list(self) -> None:
        self.gestures_listbox.delete(0, tk.END)
        try:
            from gesture_engine.recognition.knn_classifier import EmbeddingKNNClassifier
            knn = EmbeddingKNNClassifier(self.cfg.knn)
            knn.load()
            for label, count in knn.known_classes().items():
                self.gestures_listbox.insert(tk.END, f"{label}  ({count} campioni)")
        except Exception as e:
            self.gestures_listbox.insert(tk.END, f"(errore lettura database: {e})")

    def delete_selected_gesture(self) -> None:
        sel = self.gestures_listbox.curselection()
        if not sel:
            return
        label_text = self.gestures_listbox.get(sel[0])
        label = label_text.split("  (")[0]
        if not messagebox.askyesno("Conferma", f"Eliminare tutti i campioni di '{label}'?"):
            return
        from gesture_engine.recognition.knn_classifier import EmbeddingKNNClassifier
        knn = EmbeddingKNNClassifier(self.cfg.knn)
        knn.load()
        knn.remove_class(label)
        knn.save()
        self.refresh_gesture_list()

    # -------------------------- SETTINGS TAB -------------------------- #
    def _build_settings_tab(self) -> None:
        canvas = tk.Canvas(self.settings_tab)
        scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        self._settings_vars = []

        self._add_settings_section(scroll_frame, "Camera", self.cfg.camera, [
            ("device_index", "Indice webcam", int),
            ("width", "Larghezza frame", int),
            ("height", "Altezza frame", int),
            ("target_fps", "FPS target", int),
            ("flip_horizontal", "Effetto specchio", bool),
        ])
        self._add_settings_section(scroll_frame, "Pre-processing", self.cfg.preprocessing, [
            ("enable_clahe", "Abilita CLAHE (scarsa illuminazione)", bool),
            ("clahe_clip_limit", "CLAHE clip limit", float),
        ])
        self._add_settings_section(scroll_frame, "Rilevamento mano (MediaPipe)", self.cfg.landmarker, [
            ("num_hands", "Numero massimo di mani", int),
            ("min_hand_detection_confidence", "Confidenza minima rilevamento", float),
            ("min_hand_presence_confidence", "Confidenza minima presenza", float),
            ("min_tracking_confidence", "Confidenza minima tracking", float),
        ])
        self._add_settings_section(scroll_frame, "Filtro anti-jitter", self.cfg.filter, [
            ("method", "Metodo (ema / one_euro)", str),
            ("one_euro_min_cutoff", "One Euro: min cutoff", float),
            ("one_euro_beta", "One Euro: beta", float),
        ])
        self._add_settings_section(scroll_frame, "Riconoscimento regole (Livello 1)", self.cfg.rule_based, [
            ("finger_extended_ratio", "Soglia dito esteso", float),
            ("pinch_distance_ratio", "Soglia distanza pinch", float),
            ("fist_curl_ratio", "Soglia chiusura pugno", float),
        ])
        self._add_settings_section(scroll_frame, "k-NN Embedding (Livello 2)", self.cfg.knn, [
            ("k", "Numero vicini (k)", int),
            ("max_cosine_distance", "Soglia rigetto open-set", float),
        ])
        self._add_settings_section(scroll_frame, "Sequenze temporali (Livello 3)", self.cfg.state_machine, [
            ("sequence_timeout_s", "Timeout sequenza (s)", float),
            ("swipe_velocity_threshold", "Soglia velocita' swipe", float),
        ])
        self._add_settings_section(scroll_frame, "Event Bus", self.cfg.event_bus, [
            ("default_cooldown_s", "Cooldown di default (s)", float),
        ])
        self._add_settings_section(scroll_frame, "Visualizzazione", self.cfg.visualization, [
            ("draw_landmarks", "Disegna landmark", bool),
            ("draw_connections", "Disegna connessioni", bool),
            ("show_debug_text", "Mostra pannello testuale", bool),
            ("show_fps", "Mostra FPS", bool),
        ])

        ttk.Button(scroll_frame, text="💾  Salva impostazioni", command=self.save_settings).pack(pady=20)

    def _add_settings_section(self, parent, title, obj, fields) -> None:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="x", padx=10, pady=8)

        for attr, label, typ in fields:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=8, pady=3)
            ttk.Label(row, text=label, width=38).pack(side="left")

            current = getattr(obj, attr)
            if typ is bool:
                var = tk.BooleanVar(value=bool(current))
                ttk.Checkbutton(row, variable=var).pack(side="left")
            else:
                var = tk.StringVar(value=str(current))
                ttk.Entry(row, textvariable=var, width=15).pack(side="left")

            self._settings_vars.append((obj, attr, typ, var))

    def save_settings(self) -> None:
        try:
            for obj, attr, typ, var in self._settings_vars:
                raw_value = var.get()
                if typ is bool:
                    setattr(obj, attr, bool(raw_value))
                else:
                    setattr(obj, attr, typ(raw_value))
            self.cfg.save(DEFAULT_CONFIG_PATH)
            messagebox.showinfo("Salvato", "Impostazioni salvate. Riavvia il riconoscimento per applicarle.")
        except Exception as e:
            messagebox.showerror("Errore", f"Valore non valido: {e}")

    # ------------------------------------------------------------------ #
    # Motore: avvio/stop su thread separato per non bloccare la UI
    # ------------------------------------------------------------------ #
    def start_engine(self) -> None:
        if self._running:
            return
        self.cfg = EngineConfig.load(DEFAULT_CONFIG_PATH)  # ricarica eventuali modifiche
        self.engine = GestureEngine(config=self.cfg, project_root=PROJECT_ROOT)

        # Sottoscrizione eventi generici -> log nella UI (thread-safe via .after)
        for event_name in self._all_watchable_events():
            self.engine.on(event_name, self._make_event_logger(event_name))

        self.engine.register_sequence("boom_explosion", ["fist", "open_palm"])

        self._running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._engine_thread.start()

    def _all_watchable_events(self):
        events = [
            "gesture.fist", "gesture.open_palm", "gesture.pinch",
            "gesture.1_fingers", "gesture.2_fingers", "gesture.3_fingers", "gesture.4_fingers",
            "sequence.boom_explosion",
            "swipe.left", "swipe.right", "swipe.up", "swipe.down",
        ]
        # aggiunge dinamicamente un evento custom_gesture.<label> per ogni
        # gesture enrollata via k-NN, cosi' compaiono anche nel log Live
        try:
            from gesture_engine.recognition.knn_classifier import EmbeddingKNNClassifier
            knn = EmbeddingKNNClassifier(self.cfg.knn)
            knn.load()
            events.extend(f"custom_gesture.{label}" for label in knn.known_classes())
        except Exception:
            pass
        return events

    def _make_event_logger(self, event_name: str):
        def _cb(event):
            self.after(0, lambda: self._log_event(f"{event_name}"))
        return _cb

    def _engine_loop(self) -> None:
        try:
            self.engine.open()
            while self._running:
                results = self.engine.step()
                frame = getattr(self.engine, "_last_frame_bgr", None)
                if frame is None:
                    continue

                for det in getattr(self.engine, "_last_detections", []):
                    pts_px = landmarks_to_pixel_coords(det.landmarks[:, :2], frame.shape)
                    draw_hand(frame, pts_px, self.cfg.visualization)

                for i, r in enumerate(results):
                    draw_debug_panel(
                        frame, hand_index=i, rule_label=r.rule_label,
                        knn_label=r.knn_label, knn_confidence=r.knn_confidence,
                        viz_config=self.cfg.visualization,
                    )

                # Enrollment: se una sessione e' attiva, cattura un campione per frame
                if self._enrollment_capturing and results:
                    embedding_input = self.engine._last_results[0].embedding_input
                    count = self.engine.enrollment.capture_sample(embedding_input.astype("float32"))
                    self.after(0, lambda c=count: self._update_enrollment_progress(c))
                    if count >= self._enrollment_target_samples:
                        label = self.engine.enrollment.finish_session()
                        self._enrollment_capturing = False
                        self.after(0, lambda l=label: self._on_enrollment_finished(l))

                self._update_video_frame(frame)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Errore motore", str(e)))
        finally:
            self.engine.close()

    def _update_enrollment_progress(self, count: int) -> None:
        pct = min(100, int(100 * count / self._enrollment_target_samples))
        self.enroll_progress["value"] = pct
        self.enroll_status.config(text=f"Campioni raccolti: {count}/{self._enrollment_target_samples}")

    def _on_enrollment_finished(self, label: str) -> None:
        self.enroll_status.config(text=f"Gesture '{label}' registrata con successo!")
        self.refresh_gesture_list()

    def _update_video_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((760, 560))
        photo = ImageTk.PhotoImage(image=img)

        def _set():
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # mantieni riferimento (evita garbage collection)

        self.after(0, _set)

    def stop_engine(self) -> None:
        self._running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def on_close(self) -> None:
        self.stop_engine()
        time.sleep(0.2)
        self.destroy()


def main() -> None:
    app = GestureEngineApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
