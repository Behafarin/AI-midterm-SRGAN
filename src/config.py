"""Central configuration for the SRGAN mid-term pipeline.

Every value can be overridden with an environment variable of the same name, e.g.
    SRGAN_EPOCHS=200 SRGAN_TRAIN_SUBSET=8000 bash run_all.sh
"""
import os
import json


def _env(name, default, cast):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if cast is None:  # "None"-able int
        return None if v.lower() == "none" else int(v)
    return cast(v)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"   # tiny run (1-2 epochs) to check the pipeline
SEED = _env("SEED", 42, int)

# ---- Paths ------------------------------------------------------------------------------------
OUTPUT_DIR = _env("OUTPUT_DIR", os.path.join(ROOT, "outputs"), str)
RAW_DIR = _env("RAW_DIR", os.path.join(ROOT, "data_raw"), str)
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")          # resized image arrays (.npy)
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")         # final .keras models
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")     # SRGAN + classifier checkpoints
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")          # all figures
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")       # metric tables, per-epoch CSV logs
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")             # one .log file per stage
for _d in [CACHE_DIR, MODEL_DIR, CKPT_DIR, FIG_DIR, RESULT_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

KAGGLE_DATASET_SLUG = "blourdhuraju/dogs-vs-cats-classification"
CLASSES = ["cat", "dog"]

# ---- Image sizes ------------------------------------------------------------------------------
HR_SIZE = 128                  # classifier input + SRGAN target
LR_SIZE = 32                   # SRGAN input
SCALE = HR_SIZE // LR_SIZE     # 4x

# ---- Split ------------------------------------------------------------------------------------
TEST_SIZE = 0.30               # 70 / 30 (exam requirement)
VAL_SPLIT = 0.10               # 10% of TRAIN held out for early stopping / SRGAN monitoring

# ---- Classifiers A and B (Assignment 1 recipe) -------------------------------------------------
CLS_BATCH = _env("CLS_BATCH", 32, int)
CLS_HEAD_EPOCHS = _env("CLS_HEAD_EPOCHS", 8, int)
CLS_FT_EPOCHS = _env("CLS_FT_EPOCHS", 6, int)
CLS_HEAD_LR = 1e-3
CLS_FT_LR = 1e-5
FINE_TUNE_AT = -30

# ---- SRGAN (Ledig et al., 2017) ---------------------------------------------------------------
SRGAN_BATCH = _env("SRGAN_BATCH", 16, int)
SRGAN_PRETRAIN_EPOCHS = _env("SRGAN_PRETRAIN_EPOCHS", 10, int)   # MSE warm-up (SRResNet init)
SRGAN_EPOCHS = _env("SRGAN_EPOCHS", 150, int)                     # adversarial epochs (>= 150)
SRGAN_LR = 1e-4
SRGAN_LR_DECAY_AT = SRGAN_EPOCHS // 2                             # 1e-4 -> 1e-5 halfway
ADV_WEIGHT = 1e-3
N_RES_BLOCKS = _env("N_RES_BLOCKS", 16, int)
SAVE_EVERY = _env("SAVE_EVERY", 5, int)                           # checkpoint every N epochs
SRGAN_TRAIN_SUBSET = _env("SRGAN_TRAIN_SUBSET", None, None)       # None = whole training split

PRETRAINED = "imagenet"

if SMOKE_TEST:
    CLS_HEAD_EPOCHS, CLS_FT_EPOCHS = 1, 1
    SRGAN_PRETRAIN_EPOCHS, SRGAN_EPOCHS, SRGAN_LR_DECAY_AT, SAVE_EVERY = 1, 2, 1, 1
    SRGAN_TRAIN_SUBSET, N_RES_BLOCKS = None, 2
    PRETRAINED = os.environ.get("PRETRAINED", "imagenet") or None


def as_dict():
    return {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, list, type(None)))}


def dump(stage):
    """Save the exact configuration a stage ran with (for reproducibility / the write-up)."""
    with open(os.path.join(RESULT_DIR, f"config_{stage}.json"), "w") as f:
        json.dump(as_dict(), f, indent=2)
