"""
SIGHTAI – Inference utilities
Loads the saved model once and exposes predict_image().
Uses the ROC-tuned threshold saved during training for better clinical accuracy.
"""

import sys
import os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))) 

import io
import json
import numpy as np
from functools import lru_cache
from PIL import Image
import tensorflow as tf
from tensorflow import keras

import config
from data.pipeline import _clahe_numpy


# ── Model loader (cached) ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model() -> keras.Model:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. "
            "Run `python model/train.py` first."
        )
    model = keras.models.load_model(
        str(config.MODEL_PATH),
        compile=False,   # re-compile not needed for inference
    )
    print(f"Model loaded from {config.MODEL_PATH}")
    return model


# ── Threshold loader ──────────────────────────────────────────────────────────

def load_threshold() -> float:
    """Load the ROC-tuned threshold saved after training. Falls back to default."""
    if config.OPTIMAL_THRESHOLD_PATH.exists():
        with open(config.OPTIMAL_THRESHOLD_PATH) as f:
            return float(json.load(f)["threshold"])
    return config.DEFAULT_THRESHOLD


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Raw image bytes → (1, H, W, 3) float32 in [0, 1] with CLAHE applied.
    Mirrors the training pipeline so test-time distribution matches training.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(config.IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = _clahe_numpy(arr)                        # same preprocessing as training
    return np.expand_dims(arr, axis=0)


# ── Risk stratification ───────────────────────────────────────────────────────

def _risk_from_prob(prob: float, threshold: float) -> dict:
    """Map TB probability to risk level and clinical message."""
    if prob < config.RISK_LOW:
        return dict(
            label="Normal",
            risk_level="Low",
            confidence=round(1.0 - prob, 4),
            message=(
                "No significant radiological signs of tuberculosis detected. "
                "Continue routine screening as recommended by your clinician."
            ),
        )
    if prob < config.RISK_MEDIUM:
        return dict(
            label="Inconclusive",
            risk_level="Medium",
            confidence=round(1.0 - abs(prob - 0.5) * 2, 4),
            message=(
                "Findings are inconclusive. Further clinical evaluation and "
                "sputum AFB smear testing are strongly recommended."
            ),
        )
    return dict(
        label="TB Detected",
        risk_level="High",
        confidence=round(prob, 4),
        message=(
            "Radiological features suggestive of tuberculosis detected. "
            "Please refer patient immediately for sputum AFB smear and culture. "
            "This result requires urgent clinical correlation."
        ),
    )


# ── Main prediction ───────────────────────────────────────────────────────────

def predict_image(image_bytes: bytes,
                  explain: bool = False) -> dict:
    """
    Run inference on a chest X-ray image.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        explain:     If True, also compute and return a Grad-CAM heatmap.

    Returns dict with keys:
        tb_probability, label, risk_level, confidence, message,
        and optionally `gradcam_image` (base64 PNG data URI).
    """
    model     = load_model()
    threshold = load_threshold()
    tensor    = preprocess_image(image_bytes)

    prob = float(model.predict(tensor, verbose=0)[0][0])
    result = {
        "tb_probability": round(prob, 4),
        **_risk_from_prob(prob, threshold),
        "threshold_used": round(threshold, 4),
    }

    if explain:
        try:
            from api.gradcam import gradcam_to_base64
            result["gradcam_image"] = gradcam_to_base64(model, image_bytes)
        except Exception as e:
            result["gradcam_error"] = str(e)

    return result
