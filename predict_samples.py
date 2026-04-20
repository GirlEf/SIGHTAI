"""
SIGHTAI – Visual prediction script
Shows chest X-ray images with AI prediction results overlaid.
Usage:  python predict_samples.py
Output: saved_models/predictions_preview.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from data.pipeline import load_dataframe, _clahe_numpy
from api.inference import load_model, preprocess_image, load_threshold

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model     = load_model()
threshold = load_threshold()
print(f"Decision threshold: {threshold:.4f}\n")

# ── Sample images — 4 normal + 4 TB, mix of both datasets ────────────────────
df = load_dataframe()

mont_normal = df[(df["label"] == 0) & (df["dataset"] == "montgomery")].sample(2, random_state=3)
shen_normal = df[(df["label"] == 0) & (df["dataset"] == "shenzhen")].sample(2, random_state=3)
mont_tb     = df[(df["label"] == 1) & (df["dataset"] == "montgomery")].sample(2, random_state=3)
shen_tb     = df[(df["label"] == 1) & (df["dataset"] == "shenzhen")].sample(2, random_state=3)

samples = pd.concat([mont_normal, shen_normal, mont_tb, shen_tb]).reset_index(drop=True)

# ── Run predictions ───────────────────────────────────────────────────────────
results = []
for _, row in samples.iterrows():
    image_path = Path(row["image_path"])
    true_label = "TB" if row["label"] == 1 else "Normal"

    image_bytes = image_path.read_bytes()
    prob        = float(model.predict(preprocess_image(image_bytes), verbose=0)[0][0])
    predicted   = "TB" if prob >= threshold else "Normal"
    correct     = predicted == true_label

    # Load image for display (apply CLAHE same as inference)
    img = Image.open(image_path).convert("RGB").resize(config.IMG_SIZE, Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = _clahe_numpy(img_np)

    results.append({
        "img":        img_np,
        "filename":   image_path.name,
        "dataset":    row["dataset"].capitalize(),
        "true_label": true_label,
        "predicted":  predicted,
        "prob":       prob,
        "correct":    correct,
    })

# ── Plot ──────────────────────────────────────────────────────────────────────
n      = len(results)
ncols  = 4
nrows  = n // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
fig.patch.set_facecolor("#F4F7F5")
fig.suptitle(
    "SIGHTAI  –  TB Detection Predictions",
    fontsize=16, fontweight="bold", color="#006B3F", y=1.01,
)

# Row labels
for r, label in enumerate(["Normal Cases", "TB Cases"]):
    fig.text(
        0.01, 1 - (r + 0.5) / nrows,
        label, va="center", ha="left",
        fontsize=11, fontweight="bold", color="#1F2937",
        rotation=90,
    )

for idx, (ax, res) in enumerate(zip(axes.flat, results)):
    ax.imshow(res["img"], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    correct    = res["correct"]
    prob_pct   = res["prob"] * 100
    pred_label = res["predicted"]

    # Border colour: green = correct, red = wrong
    border_color = "#006B3F" if correct else "#CE1126"
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)

    # Probability bar at bottom
    bar_color = "#10B981" if res["prob"] < config.RISK_LOW else \
                "#F59E0B" if res["prob"] < config.RISK_MEDIUM else "#EF4444"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), res["prob"], 0.045,
        boxstyle="square,pad=0",
        transform=ax.transAxes,
        color=bar_color, alpha=0.85, zorder=5,
    ))

    # Prediction label box
    verdict_bg = "#D1FAE5" if pred_label == "Normal" else "#FEE2E2"
    verdict_fg = "#065F46" if pred_label == "Normal" else "#991B1B"
    ax.text(
        0.5, 0.97,
        f"Predicted: {pred_label}  ({prob_pct:.1f}%)",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9, fontweight="bold", color=verdict_fg,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=verdict_bg,
                  edgecolor=verdict_fg, alpha=0.92),
        zorder=6,
    )

    # Ground truth label
    ax.text(
        0.5, 0.86,
        f"True: {res['true_label']}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#374151", alpha=0.75),
        zorder=6,
    )

    # Correct / Wrong badge top-right
    badge_text = "CORRECT" if correct else "WRONG"
    badge_bg   = "#006B3F" if correct else "#CE1126"
    ax.text(
        0.98, 0.02, badge_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_bg, alpha=0.9),
        zorder=6,
    )

    # Filename + dataset
    ax.set_title(
        f"{res['filename']}\n{res['dataset']}",
        fontsize=7.5, color="#6B7280", pad=4,
    )

# Summary bar at bottom
correct_count = sum(r["correct"] for r in results)
acc = correct_count / len(results) * 100
fig.text(
    0.5, -0.02,
    f"Sample accuracy: {correct_count}/{len(results)} correct  ({acc:.0f}%)   |"
    f"   Decision threshold: {threshold:.4f}   |"
    f"   Green border = correct   Red border = wrong",
    ha="center", fontsize=9, color="#6B7280",
)

plt.tight_layout()
out = config.SAVED_MODELS_DIR / "predictions_preview.png"
config.SAVED_MODELS_DIR.mkdir(exist_ok=True)
plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPrediction grid saved -> {out}")
plt.close()

# ── Text summary ──────────────────────────────────────────────────────────────
print(f"\n{'File':<35} {'Dataset':<12} {'True':>6} {'Pred':>7} {'Prob':>7}  Result")
print("-" * 80)
for r in results:
    mark = "OK   " if r["correct"] else "WRONG"
    print(f"{r['filename']:<35} {r['dataset']:<12} {r['true_label']:>6} "
          f"{r['predicted']:>7} {r['prob']:>7.4f}  {mark}")
print("-" * 80)
print(f"Sample accuracy: {correct_count}/{len(results)} = {acc:.0f}%")
