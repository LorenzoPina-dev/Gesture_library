"""
gesture_engine
==============
Libreria Python industry-level per tracciamento, riconoscimento e gestione
di eventi basati su gesture della mano in tempo reale via webcam.

API pubblica principale:

    from gesture_engine import GestureEngine, EngineConfig

    engine = GestureEngine()
    engine.on("gesture.fist", lambda e: print("Pugno!"))
    with engine:
        engine.run()
"""

from .config import EngineConfig
from .pipeline import GestureEngine, HandFrameResult
from .events import EventBus, Event

__version__ = "1.0.0"

__all__ = [
    "GestureEngine",
    "EngineConfig",
    "HandFrameResult",
    "EventBus",
    "Event",
]
