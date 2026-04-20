"""
SIGHTAI – Model Architecture
DenseNet121 backbone (proven on chest X-rays via CheXNet) with a custom
binary classification head. Focal Loss handles hard examples better than BCE.
Two-phase training: head warm-up → partial fine-tuning.
"""

import sys
import os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))) 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


# ── Focal Loss ────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = config.FOCAL_GAMMA,
               alpha: float = config.FOCAL_ALPHA):
    """
    Focal Loss for binary classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma > 0 down-weights easy examples so the model focuses on hard,
    misclassified cases — exactly what we need for subtle TB radiological signs.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce    = -(y_true * tf.math.log(y_pred) +
                   (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        weight = alpha * tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(weight * bce)

    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(trainable_base: bool = False) -> keras.Model:
    """
    Build the SIGHTAI v2 model.

    Architecture:
        DenseNet121 (ImageNet weights, frozen by default)
            — same backbone used in CheXNet (Stanford chest X-ray model)
        → GlobalAveragePooling2D
        → Dense(512, relu) + BatchNorm + Dropout(0.5)
        → Dense(128, relu) + Dropout(0.3)
        → Dense(1, sigmoid)  ←  TB probability

    Args:
        trainable_base: Unfreeze the backbone for fine-tuning.
    """
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3), name="xray_input")

    # DenseNet121 — uses input_tensor to inline layers into the outer model,
    # making all intermediate tensors accessible for Grad-CAM.
    base = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base.trainable = trainable_base

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)

    x = layers.Dense(512, name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(0.5, name="drop1")(x)

    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    output = layers.Dense(1, activation="sigmoid", name="tb_probability")(x)

    return keras.Model(inputs=inputs, outputs=output, name="SIGHTAI_v2")


# ── Compile ───────────────────────────────────────────────────────────────────

def compile_model(model: keras.Model,
                  learning_rate: float = config.LEARNING_RATE,
                  lr_schedule=None) -> keras.Model:
    """Attach focal loss, Adam, and medical metrics."""
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule if lr_schedule is not None else learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def unfreeze_top_layers(model: keras.Model,
                        fine_tune_at: int = config.FINE_TUNE_AT) -> keras.Model:
    """
    Unfreeze DenseNet121 layers from index `fine_tune_at` onward.
    DenseNet layers are inlined into the outer model via input_tensor,
    so we operate on model.layers directly.

    Earlier layers retain generic ImageNet edge/texture features.
    Later layers are allowed to specialise for TB radiological patterns.
    """
    head_names = {
        "xray_input", "gap",
        "fc1", "bn1", "relu1", "drop1",
        "fc2", "drop2", "tb_probability",
    }
    backbone_layers = [l for l in model.layers if l.name not in head_names]

    # Freeze all, then selectively unfreeze the top portion
    for layer in backbone_layers:
        layer.trainable = False
    for layer in backbone_layers[fine_tune_at:]:
        layer.trainable = True

    n_trainable = sum(1 for l in backbone_layers if l.trainable)
    print(f"Fine-tuning: {n_trainable}/{len(backbone_layers)} backbone layers "
          f"unfrozen (from index {fine_tune_at})")
    return model


# ── Utility ───────────────────────────────────────────────────────────────────

def get_last_conv_layer_name(model: keras.Model) -> str:
    """Return the name of the last Conv2D layer — used for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


if __name__ == "__main__":
    m = build_model()
    m = compile_model(m)
    m.summary(line_length=90)
    print(f"\nLast conv layer for Grad-CAM: {get_last_conv_layer_name(m)}")
    total = sum(p.numpy().size for p in m.trainable_weights)
    print(f"Trainable params: {total:,}")
