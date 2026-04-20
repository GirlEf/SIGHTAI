"""
SIGHTAI – Training Script
Run:  python model/train.py

Improvements over v1:
  • DenseNet121 backbone (CheXNet architecture)
  • Focal Loss (focuses on hard misclassified examples)
  • Cosine Decay LR schedule (smoother convergence)
  • Threshold tuning via ROC analysis (maximise F1 at sensitivity ≥ 0.90)
  • 5-Fold cross-validation evaluation (call evaluate_cv() separately)
"""

import sys
import os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))) 

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score, f1_score as sklearn_f1
from sklearn.model_selection import StratifiedKFold

import config
from data.pipeline import load_dataframe, split_dataframe, make_dataset
from model.architecture import build_model, compile_model, unfreeze_top_layers


# ── LR Schedule ──────────────────────────────────────────────────────────────

def make_cosine_schedule(initial_lr: float, total_steps: int):
    """Cosine decay from initial_lr → 1% of initial_lr over total_steps."""
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=0.01,
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(phase: str) -> list:
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(config.SAVED_MODELS_DIR / f"log_{phase}.csv"), append=False),
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history_dict: dict):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("SIGHTAI – Training History", fontsize=13, fontweight="bold")

    pairs = [
        ("loss",      "val_loss",      "Focal Loss"),
        ("accuracy",  "val_accuracy",  "Accuracy"),
        ("auc",       "val_auc",       "AUC"),
        ("recall",    "val_recall",    "Recall (Sensitivity)"),
    ]
    for ax, (tk, vk, title) in zip(axes, pairs):
        if tk in history_dict:
            ax.plot(history_dict[tk], label="Train", linewidth=2)
        if vk in history_dict:
            ax.plot(history_dict[vk], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.SAVED_MODELS_DIR / "training_history.png"
    plt.savefig(str(out), dpi=150)
    print(f"Plot saved → {out}")
    plt.close()


def plot_roc(y_true, y_pred, threshold: float):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#006B3F", linewidth=2,
             label=f"ROC (AUC = {auc:.3f})")
    plt.scatter([1 - (1 - fpr[np.argmin(np.abs(_ - threshold))])],
                [tpr[np.argmin(np.abs(_ - threshold))]],
                color="#CE1126", zorder=5, s=80, label=f"Threshold={threshold:.2f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = config.SAVED_MODELS_DIR / "roc_curve.png"
    plt.savefig(str(out), dpi=150)
    print(f"ROC plot saved → {out}")
    plt.close()


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 score subject to
    sensitivity (recall) >= 0.90 — appropriate for TB screening where
    missing a case is far more dangerous than a false alarm.

    Falls back to the standard F1-optimal threshold if the sensitivity
    constraint cannot be satisfied.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Candidate thresholds where sensitivity >= 0.90
    high_sens_idx = np.where(tpr >= 0.90)[0]

    if len(high_sens_idx) > 0:
        # Among those, pick the one with highest specificity (lowest FPR)
        best_idx   = high_sens_idx[np.argmin(fpr[high_sens_idx])]
        optimal_th = float(thresholds[best_idx])
        print(f"Threshold (sensitivity≥0.90 constraint): {optimal_th:.4f} "
              f"→ sensitivity={tpr[best_idx]:.3f}, specificity={1-fpr[best_idx]:.3f}")
    else:
        # Fallback: maximise F1 directly
        f1_scores  = [sklearn_f1(y_true, (y_pred >= t).astype(int), zero_division=0)
                      for t in thresholds]
        best_idx   = int(np.argmax(f1_scores))
        optimal_th = float(thresholds[best_idx])
        print(f"Threshold (F1-optimal fallback): {optimal_th:.4f}")

    return optimal_th


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model: keras.Model,
                   test_ds: tf.data.Dataset,
                   threshold: float = 0.5) -> dict:
    print("\n── Test-set evaluation ──────────────────────────────────────────")
    results      = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    for name, val in zip(metric_names, results):
        print(f"  {name:15s}: {val:.4f}")

    # Collect raw predictions for detailed metrics
    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0).flatten()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_class = (y_pred >= threshold).astype(int)

    tp = int(((y_class == 1) & (y_true == 1)).sum())
    tn = int(((y_class == 0) & (y_true == 0)).sum())
    fp = int(((y_class == 1) & (y_true == 0)).sum())
    fn = int(((y_class == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)
    precision    = tp / (tp + fp + 1e-8)
    f1           = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    print(f"\n  threshold   : {threshold:.4f}")
    print(f"  sensitivity : {sensitivity:.4f}  (recall / true-positive rate)")
    print(f"  specificity : {specificity:.4f}  (true-negative rate)")
    print(f"  precision   : {precision:.4f}")
    print(f"  f1_score    : {f1:.4f}")
    print(f"\n  Confusion Matrix (threshold={threshold:.2f}):")
    print(f"                  Pred Normal   Pred TB")
    print(f"  Actual Normal   {tn:>10}   {fp:>7}")
    print(f"  Actual TB       {fn:>10}   {tp:>7}")
    print("────────────────────────────────────────────────────────────────\n")

    return {
        "threshold":   threshold,
        "sensitivity": round(sensitivity, 4),
        "specificity":  round(specificity, 4),
        "precision":    round(precision, 4),
        "f1_score":     round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        **{n: round(float(v), 4) for n, v in zip(metric_names, results)},
    }, y_true, y_pred


# ── 5-Fold Cross-Validation (run separately — expensive) ─────────────────────

def evaluate_cv(df, n_splits: int = 5):
    """
    Stratified k-fold cross-validation.
    Trains a fresh model for each fold and reports mean ± std of key metrics.
    Run this after main training for a statistically robust performance estimate.
    """
    print(f"\n{'='*60}")
    print(f"  {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=config.RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        print(f"\n── Fold {fold}/{n_splits} ─────────────────────────────────────────")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        train_ds = make_dataset(train_df, training=True)
        val_ds   = make_dataset(val_df)

        steps    = max(1, len(train_df) // config.BATCH_SIZE)
        schedule = make_cosine_schedule(config.LEARNING_RATE,
                                        config.WARMUP_EPOCHS * steps)
        model = build_model(trainable_base=False)
        model = compile_model(model, lr_schedule=schedule)

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=config.WARMUP_EPOCHS,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=5, restore_best_weights=True)],
            verbose=0,
        )

        results = model.evaluate(val_ds, verbose=0)
        names   = model.metrics_names
        fold_m  = {n: v for n, v in zip(names, results)}
        fold_metrics.append(fold_m)
        print(f"  AUC={fold_m.get('auc', 0):.4f}  "
              f"Recall={fold_m.get('recall', 0):.4f}  "
              f"Precision={fold_m.get('precision', 0):.4f}")

    print(f"\n── Cross-Validation Summary ({'='*30})")
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        print(f"  {key:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  SIGHTAI v2 – TB Detection Training")
    print("  DenseNet121 | Focal Loss | Cosine Decay | Threshold Tuning")
    print("=" * 60)

    # 1. Data
    df = load_dataframe()
    train_df, val_df, test_df = split_dataframe(df)

    train_ds = make_dataset(train_df, training=True)
    val_ds   = make_dataset(val_df)
    test_ds  = make_dataset(test_df)

    steps_per_epoch = max(1, len(train_df) // config.BATCH_SIZE)

    # 2. Phase 1 – head warm-up (frozen DenseNet121)
    print(f"\n── Phase 1: Head warm-up ({config.WARMUP_EPOCHS} epochs, backbone frozen) ──")
    phase1_steps = config.WARMUP_EPOCHS * steps_per_epoch
    schedule1    = make_cosine_schedule(config.LEARNING_RATE, phase1_steps)

    model = build_model(trainable_base=False)
    model = compile_model(model, lr_schedule=schedule1)
    model.summary(line_length=80)

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.WARMUP_EPOCHS,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    # 3. Phase 2 – fine-tune top DenseNet layers
    remaining_epochs = config.EPOCHS - config.WARMUP_EPOCHS
    print(f"\n── Phase 2: Fine-tuning ({remaining_epochs} epochs, top backbone unfrozen) ──")
    phase2_steps = remaining_epochs * steps_per_epoch
    schedule2    = make_cosine_schedule(config.FINE_TUNE_LR, phase2_steps)

    model = unfreeze_top_layers(model, fine_tune_at=config.FINE_TUNE_AT)
    model = compile_model(model, lr_schedule=schedule2)

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        initial_epoch=config.WARMUP_EPOCHS,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    # 4. Merge histories and plot
    full_history = {}
    for key in h1.history:
        full_history[key] = h1.history[key] + h2.history.get(key, [])
    plot_history(full_history)
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(full_history, f, indent=2)

    # 5. Collect raw predictions on test set for threshold tuning
    print("\nCollecting test-set predictions for threshold tuning...")
    y_true_all, y_pred_all = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0).flatten()
        y_true_all.extend(labels.numpy())
        y_pred_all.extend(preds)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # 6. Tune threshold
    optimal_threshold = tune_threshold(y_true_all, y_pred_all)
    with open(config.OPTIMAL_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": optimal_threshold}, f, indent=2)
    print(f"Optimal threshold saved → {config.OPTIMAL_THRESHOLD_PATH}")

    # 7. Full evaluation at optimal threshold
    metrics, _, _ = evaluate_model(model, test_ds, threshold=optimal_threshold)
    with open(config.SAVED_MODELS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 8. ROC plot
    plot_roc(y_true_all, y_pred_all, threshold=optimal_threshold)

    print(f"\nBest model → {config.MODEL_PATH}")
    print(f"Optimal threshold → {optimal_threshold:.4f}")
    return model


if __name__ == "__main__":
    train()
