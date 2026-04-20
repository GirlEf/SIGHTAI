"""
SIGHTAI – Data Pipeline
Loads, labels, applies CLAHE, splits, and creates TF datasets from the
Montgomery + Shenzhen TB chest-X-ray collections.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

import config


# ── Label classifier ──────────────────────────────────────────────────────────

def _classify_finding(finding: str) -> int:
    """Return 0 (normal) or 1 (TB). Both datasets are TB-specific screening
    programmes, so every non-normal finding is a TB case."""
    return 0 if str(finding).strip().lower() == "normal" else 1


# ── CLAHE preprocessing ───────────────────────────────────────────────────────

def _clahe_numpy(img_np: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a [H, W, 3] float32 image in [0, 1].
    Converts to grayscale, enhances, then stacks back to 3 channels.
    Falls back to PIL histogram equalisation if scikit-image is absent.
    """
    gray = (0.299 * img_np[:, :, 0] +
            0.587 * img_np[:, :, 1] +
            0.114 * img_np[:, :, 2])
    try:
        from skimage.exposure import equalize_adapthist
        enhanced = equalize_adapthist(gray, clip_limit=config.CLAHE_CLIP_LIMIT)
    except ImportError:
        from PIL import Image, ImageOps
        pil = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
        pil = ImageOps.equalize(pil)
        enhanced = np.array(pil, dtype=np.float32) / 255.0

    out = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return out.astype(np.float32)


def _apply_clahe_tf(image: tf.Tensor, label: tf.Tensor):
    """TF dataset-compatible CLAHE wrapper."""
    def _fn(img):
        return _clahe_numpy(img.numpy())

    image = tf.py_function(_fn, [image], tf.float32)
    image.set_shape((*config.IMG_SIZE, 3))
    return image, label


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataframe() -> pd.DataFrame:
    """
    Combine Montgomery + Shenzhen metadata into one DataFrame.
    Columns: [image_path, label, dataset, age, sex, findings]
    """
    mont = pd.read_csv(config.MONTGOMERY_META)
    mont = mont.rename(columns={"study_id": "filename", "gender": "sex"})
    mont["image_path"] = mont["filename"].apply(
        lambda f: str(config.MONTGOMERY_IMAGES / f))
    mont["dataset"] = "montgomery"

    shen = pd.read_csv(config.SHENZHEN_META)
    shen = shen.rename(columns={"study_id": "filename"})
    shen["image_path"] = shen["filename"].apply(
        lambda f: str(config.SHENZHEN_IMAGES / f))
    shen["dataset"] = "shenzhen"

    df = pd.concat(
        [mont[["image_path", "findings", "age", "sex", "dataset"]],
         shen[["image_path", "findings", "age", "sex", "dataset"]]],
        ignore_index=True,
    )

    df["label"] = df["findings"].apply(_classify_finding)

    # Drop rows where the image file is missing on disk
    exists = df["image_path"].apply(lambda p: Path(p).exists())
    excluded = (~exists).sum()
    df = df[exists].reset_index(drop=True)

    tb_count     = (df["label"] == 1).sum()
    normal_count = (df["label"] == 0).sum()
    print(f"Dataset loaded: {len(df)} images | "
          f"TB={tb_count} | Normal={normal_count} | Excluded={excluded}")
    return df


# ── Train / Val / Test split ──────────────────────────────────────────────────

def split_dataframe(df: pd.DataFrame):
    """Stratified split → (train_df, val_df, test_df)."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE,
        stratify=df["label"], random_state=config.RANDOM_STATE)
    relative_val = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=relative_val,
        stratify=train_val["label"], random_state=config.RANDOM_STATE)
    print(f"Split -> train={len(train)} | val={len(val)} | test={len(test)}")
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ── TF dataset factory ────────────────────────────────────────────────────────

def _load_image(path: str) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_png(raw, channels=3)
    img = tf.image.resize(img, config.IMG_SIZE)
    return tf.cast(img, tf.float32) / 255.0


def _augment(img: tf.Tensor, label: tf.Tensor):
    """Augmentations for chest X-rays — applied only during training."""
    img = tf.image.random_flip_left_right(img)
    img = _random_zoom(img, zoom_range=0.10)
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img, label


def _random_zoom(img: tf.Tensor, zoom_range: float = 0.10) -> tf.Tensor:
    h, w = config.IMG_SIZE
    scale = tf.random.uniform((), 1.0, 1.0 + zoom_range)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img = tf.image.resize(img, [new_h, new_w])
    return tf.image.random_crop(img, [h, w, 3])


def make_dataset(df: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a split DataFrame.
    Pipeline: load -> CLAHE -> [augment if training] -> batch -> prefetch
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (df["image_path"].tolist(), df["label"].tolist()))
    ds = ds.map(
        lambda p, l: (_load_image(p), l),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_apply_clahe_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=config.RANDOM_STATE)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_dataframe()
    train_df, val_df, test_df = split_dataframe(df)
    ds = make_dataset(train_df, training=True)
    for imgs, labels in ds.take(1):
        print(f"Batch: {imgs.shape} | range [{imgs.numpy().min():.3f}, {imgs.numpy().max():.3f}]")
