"""
gesture_engine/app_context/active_window.py
==============================================
Rilevamento della finestra/processo attualmente in primo piano, usato dal
ProfileManager per selezionare automaticamente il set di gesture giusto per
l'app che si sta usando (es. Unity vs un visualizzatore immagini).

Richiede 'pywin32' su Windows (non incluso in requirements.txt di base,
perche' il resto della libreria e' cross-platform: installalo con
`pip install pywin32` se vuoi il cambio automatico di profilo). Se non
disponibile, le funzioni ritornano None e ProfileManager usera' sempre il
profilo di default, senza errori.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import win32gui
    import win32process
    import psutil

    _BACKEND_AVAILABLE = True
except ImportError:
    _BACKEND_AVAILABLE = False


def backend_available() -> bool:
    return _BACKEND_AVAILABLE


def get_active_window_info() -> Optional[Tuple[str, str]]:
    """Ritorna (process_name, window_title) della finestra in primo piano,
    o None se il backend non e' disponibile o la finestra non e' risolvibile
    (es. desktop senza focus)."""
    if not _BACKEND_AVAILABLE:
        return None
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        title = win32gui.GetWindowText(hwnd) or ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = psutil.Process(pid).name() if pid else ""
        return process_name, title
    except Exception:
        return None
