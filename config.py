from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Dataset paths ─────────────────────────────────────────────────────────────
MONTGOMERY_DIR = BASE_DIR / "tuberculosis-chest-xrays-montgomery"
SHENZHEN_DIR   = BASE_DIR / "tuberculosis-chest-xrays-shenzhen"

MONTGOMERY_META   = MONTGOMERY_DIR / "montgomery_metadata.csv"
SHENZHEN_META     = SHENZHEN_DIR   / "shenzhen_metadata.csv"
MONTGOMERY_IMAGES = MONTGOMERY_DIR / "images" / "images"
SHENZHEN_IMAGES   = SHENZHEN_DIR   / "images" / "images"

# ── Model persistence ─────────────────────────────────────────────────────────
SAVED_MODELS_DIR       = BASE_DIR / "saved_models"
MODEL_PATH             = SAVED_MODELS_DIR / "sightai_model.keras"
HISTORY_PATH           = SAVED_MODELS_DIR / "training_history.json"
OPTIMAL_THRESHOLD_PATH = SAVED_MODELS_DIR / "optimal_threshold.json"

# ── Training hyperparameters ──────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS        = 40          # more epochs for DenseNet fine-tuning
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5
FINE_TUNE_AT  = 249         # DenseNet121 has ~427 layers; unfreeze top ~178
WARMUP_EPOCHS = 15
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15
RANDOM_STATE  = 42

# ── Focal Loss ────────────────────────────────────────────────────────────────
FOCAL_GAMMA = 2.0    # focusing parameter — higher = more focus on hard examples
FOCAL_ALPHA = 0.25   # class weighting for positive class

# ── CLAHE (contrast enhancement for X-rays) ───────────────────────────────────
CLAHE_CLIP_LIMIT = 0.03   # higher = more contrast enhancement

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Risk thresholds — overridden at runtime by optimal_threshold.json ─────────
RISK_LOW       = 0.30
RISK_MEDIUM    = 0.60
DEFAULT_THRESHOLD = 0.50   # fallback if no tuned threshold saved yet
