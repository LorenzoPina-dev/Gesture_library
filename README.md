# 🖐️ HandControl Engine

**HandControl Engine** is a Python, event-driven Computer Vision framework powered by MediaPipe for real-time hand tracking. It allows developers to easily bind custom functions and callbacks to both static events (e.g., finger counts, pinch gestures) and complex dynamic gestures (e.g., swipes, waving).

---

## ✨ Key Features

* ⚡ **Real-Time Tracking:** Leverages MediaPipe Hands for ultra-fast, accurate 3D detection of 21 hand landmarks.
* 🎯 **Event-Driven Architecture:** Effortlessly attach Python callbacks to hand gestures using simple decorators (`@app.on(...)`).
* 📏 **Geometry-Based Static Gestures:** Instant detection of raised fingers, distances, and contact points (pinches) without heavy CPU overhead.
* 🌊 **Dynamic Gesture Processing:** Temporal tracking buffer to recognize motion-based gestures like swipes and waving.
* ⏱️ **Cooldown & Debouncing System:** Built-in mechanism to prevent unwanted rapid re-triggering of events.
* 📐 **Scale Invariance:** Normalized coordinates ensure gesture recognition works consistently regardless of distance from the camera.

---

## 🛠️ Prerequisites & Installation

Ensure you have **Python 3.8+** installed, then install the required dependencies:

```bash
pip install opencv-python mediapipe numpy