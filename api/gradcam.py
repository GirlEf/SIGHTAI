"""
SIGHTAI – Grad-CAM (Gradient-weighted Class Activation Mapping)
Generates a heatmap overlay showing which regions of the X-ray
most influenced the model's TB prediction.

Reference: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391
"""

import sys
import os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))) 

import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
from model.architecture import get_last_conv_layer_name


def _build_grad_model(model: keras.Model) -> tuple[keras.Model, str]:
    """
    Build a sub-model that outputs (last_conv_features, prediction).
    Caches the conv layer name so it's only searched once.
    """
    conv_layer_name = get_last_conv_layer_name(model)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.output,
        ],
    )
    return grad_model, conv_layer_name


def compute_gradcam(model: keras.Model,
                    img_array: np.ndarray) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single image.

    Args:
        model:     Trained SIGHTAI Keras model.
        img_array: Preprocessed image (1, H, W, 3) float32 in [0, 1].

    Returns:
        heatmap: (H, W) float32 array in [0, 1].
    """
    grad_model, _ = _build_grad_model(model)

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        # Gradient of TB probability w.r.t. last conv feature map
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)          # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)

    # Weight each feature map channel by its gradient importance
    conv_outputs = conv_outputs[0]                           # (h, w, C)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap      = tf.squeeze(heatmap)                       # (h, w)

    # ReLU: keep only features that positively influence TB prediction
    heatmap = tf.nn.relu(heatmap)

    # Normalise to [0, 1]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    # Resize heatmap to original image size
    heatmap_np = heatmap.numpy()
    heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8), mode="L")
    heatmap_img = heatmap_img.resize(config.IMG_SIZE[::-1], Image.LANCZOS)
    return np.array(heatmap_img, dtype=np.float32) / 255.0


def overlay_heatmap(original_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.45,
                    colormap: str = "jet") -> np.ndarray:
    """
    Superimpose the heatmap on the original X-ray.

    Args:
        original_img: (H, W, 3) float32 in [0, 1] — the chest X-ray.
        heatmap:      (H, W) float32 in [0, 1] — Grad-CAM output.
        alpha:        Heatmap opacity (0 = invisible, 1 = fully opaque).
        colormap:     Matplotlib colormap name for the heatmap.

    Returns:
        (H, W, 3) uint8 overlay image.
    """
    cmap       = cm.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[:, :, :3]          # (H, W, 3) RGBA → RGB

    # Blend
    overlay = (1 - alpha) * original_img + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def gradcam_to_base64(model: keras.Model,
                      image_bytes: bytes,
                      alpha: float = 0.45) -> str:
    """
    Full pipeline: raw image bytes → base64-encoded PNG of the Grad-CAM overlay.

    Args:
        model:       Trained SIGHTAI model.
        image_bytes: Raw bytes of the uploaded chest X-ray.
        alpha:       Heatmap blend strength.

    Returns:
        Base64-encoded PNG string (data URI ready).
    """
    # Decode and preprocess (same as inference pipeline)
    img_pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_pil  = img_pil.resize(config.IMG_SIZE, Image.LANCZOS)
    img_np   = np.array(img_pil, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_np, axis=0)

    # Compute Grad-CAM
    heatmap = compute_gradcam(model, img_batch)

    # Overlay
    overlay_img = overlay_heatmap(img_np, heatmap, alpha=alpha)

    # Encode to base64 PNG
    buf = io.BytesIO()
    Image.fromarray(overlay_img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_gradcam_figure(model: keras.Model,
                        image_bytes: bytes,
                        save_path: str,
                        prob: float = None):
    """Save a side-by-side figure: original | Grad-CAM overlay."""
    img_pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_pil  = img_pil.resize(config.IMG_SIZE, Image.LANCZOS)
    img_np   = np.array(img_pil, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_np, axis=0)

    heatmap     = compute_gradcam(model, img_batch)
    overlay_img = overlay_heatmap(img_np, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"SIGHTAI – Grad-CAM Explanation" +
        (f"   |   TB Probability: {prob*100:.1f}%" if prob is not None else ""),
        fontsize=12, fontweight="bold",
    )
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original X-Ray", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title("AI Attention Map (Grad-CAM)", fontweight="bold", color="#CE1126")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM figure saved → {save_path}")
