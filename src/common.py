"""Shared helpers: logging, data loading, LR/HR transforms, classifier pipeline."""
import os
import sys
import time
import random
import logging
import platform

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless server: write figures to files
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, callbacks
from keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

import config as C

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------------------
def setup(stage):
    """Seed everything, configure GPU memory growth, and log to outputs/logs/<stage>.log + stdout."""
    random.seed(C.SEED); np.random.seed(C.SEED); tf.random.set_seed(C.SEED)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    log = logging.getLogger(stage)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in [logging.FileHandler(os.path.join(C.LOG_DIR, f"{stage}.log")), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt); log.addHandler(h)

    log.info("=" * 90)
    log.info(f"STAGE {stage} started | smoke test: {C.SMOKE_TEST}")
    log.info(f"python {platform.python_version()} | tensorflow {tf.__version__} | keras {keras.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    log.info(f"GPUs visible: {len(gpus)} {[g.name for g in gpus]} | CUDA_VISIBLE_DEVICES="
             f"{os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    if not gpus:
        log.warning("No GPU visible to TensorFlow -- training will be very slow.")
    log.info(f"output dir: {C.OUTPUT_DIR}")
    C.dump(stage)
    return log


class EpochLogger(callbacks.Callback):
    """Write one line per epoch into the stage log (Keras' own progress bar goes to stdout)."""
    def __init__(self, log, tag):
        super().__init__(); self.log, self.tag, self.t0 = log, tag, None
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        m = "  ".join(f"{k}={v:.4g}" for k, v in (logs or {}).items())
        self.log.info(f"[{self.tag}] epoch {epoch + 1}: {m}  ({time.time() - self.t0:.0f}s)")


def savefig(name):
    plt.savefig(os.path.join(C.FIG_DIR, name), dpi=120, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
CACHE_FILES = {k: os.path.join(C.CACHE_DIR, f"{k}.npy") for k in ["X_train", "y_train", "X_test", "y_test"]}
SR_FILES = {k: os.path.join(C.CACHE_DIR, f"SR_{k}.npy") for k in ["tr", "val", "test"]}


def load_data():
    """Return X_tr, y_tr, X_val, y_val, X_test, y_test (uint8 128x128 images, int labels)."""
    missing = [p for p in CACHE_FILES.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {missing} -- run src/01_prepare_data.py first.")
    X_train, y_train, X_test, y_test = (np.load(CACHE_FILES[k]) for k in ["X_train", "y_train", "X_test", "y_test"])
    tr_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=C.VAL_SPLIT,
                                       stratify=y_train, random_state=C.SEED)
    return X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx], X_test, y_test


def load_sr_data():
    missing = [p for p in SR_FILES.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {missing} -- run src/05_generate_sr_dataset.py first.")
    return tuple(np.load(SR_FILES[k]) for k in ["tr", "val", "test"])


def to_lr(hr_uint8):
    """HR uint8 [0,255] (N,128,128,3) -> LR float32 [0,1] (N,32,32,3), bicubic + anti-alias."""
    hr = tf.cast(hr_uint8, tf.float32) / 255.0
    return tf.clip_by_value(tf.image.resize(hr, (C.LR_SIZE, C.LR_SIZE), method="bicubic", antialias=True), 0.0, 1.0)


def bicubic_up(lr):
    """LR [0,1] -> 128x128 [0,1] with plain bicubic interpolation (non-learned baseline)."""
    return tf.clip_by_value(tf.image.resize(lr, (C.HR_SIZE, C.HR_SIZE), method="bicubic"), 0.0, 1.0)


# ---------------------------------------------------------------------------------------------
# Classifier pipeline (Model A and Model B)
# ---------------------------------------------------------------------------------------------
def make_augment():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(20 / 360),             # +-20 degrees
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomBrightness(0.20, value_range=(0, 255)),
    ], name="augmentation")


_AUGMENT = None


def make_cls_ds(X, y, training):
    """tf.data pipeline. Images stay in [0,255]; the model's Rescaling layer normalizes them."""
    global _AUGMENT
    ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.float32)))
    if training:
        ds = ds.shuffle(len(X), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(C.CLS_BATCH).map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=AUTOTUNE)
    if training:
        _AUGMENT = _AUGMENT or make_augment()
        ds = ds.map(lambda a, b: (_AUGMENT(a, training=True), b), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def build_classifier(name):
    inputs = keras.Input(shape=(C.HR_SIZE, C.HR_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(inputs)   # [0,255] -> [-1,1]
    base = MobileNetV2(include_top=False, weights=C.PRETRAINED, input_shape=(C.HR_SIZE, C.HR_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)            # BatchNorm stays in inference mode, also when fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name)


def _compile(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


def _stage_callbacks(run, log):
    return [
        callbacks.BackupAndRestore(os.path.join(C.CKPT_DIR, f"{run}_backup")),  # resume after a crash
        callbacks.ModelCheckpoint(os.path.join(C.CKPT_DIR, f"{run}_epoch{{epoch:02d}}.keras")),  # every epoch
        callbacks.ModelCheckpoint(os.path.join(C.CKPT_DIR, f"{run}_best.keras"), monitor="val_loss", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        callbacks.CSVLogger(os.path.join(C.RESULT_DIR, f"{run}_log.csv"), append=True),
        EpochLogger(log, run),
    ]


def train_classifier(name, train_ds, val_ds, log):
    """Head training + fine-tuning. Finished stages are saved to models/ and skipped on re-runs."""
    head_path = os.path.join(C.MODEL_DIR, f"{name}_head.keras")
    final_path = os.path.join(C.MODEL_DIR, f"{name}_final.keras")
    if os.path.exists(final_path):
        log.info(f"[{name}] {final_path} exists -- loading, skipping training.")
        return keras.models.load_model(final_path)

    if os.path.exists(head_path):
        log.info(f"[{name}] head stage already done -- loading {head_path}")
        model = keras.models.load_model(head_path)
    else:
        model = build_classifier(name)
        _compile(model, C.CLS_HEAD_LR)
        log.info(f"[{name}] stage 1: head training (frozen backbone), lr={C.CLS_HEAD_LR}, "
                 f"max {C.CLS_HEAD_EPOCHS} epochs")
        model.fit(train_ds, validation_data=val_ds, epochs=C.CLS_HEAD_EPOCHS, verbose=2,
                  callbacks=_stage_callbacks(f"{name}_head", log))
        model.save(head_path); log.info(f"[{name}] saved {head_path}")

    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:C.FINE_TUNE_AT]:
        layer.trainable = False
    _compile(model, C.CLS_FT_LR)
    log.info(f"[{name}] stage 2: fine-tuning last {-C.FINE_TUNE_AT} backbone layers, lr={C.CLS_FT_LR}, "
             f"max {C.CLS_FT_EPOCHS} epochs")
    model.fit(train_ds, validation_data=val_ds, epochs=C.CLS_FT_EPOCHS, verbose=2,
              callbacks=_stage_callbacks(f"{name}_finetune", log))
    model.save(final_path); log.info(f"[{name}] saved {final_path}")
    return model


def plot_cls_logs(name):
    import pandas as pd
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    offset = 0
    for stage in ["head", "finetune"]:
        p = os.path.join(C.RESULT_DIR, f"{name}_{stage}_log.csv")
        if not os.path.exists(p):
            continue
        lg = pd.read_csv(p).drop_duplicates("epoch", keep="last")
        ep = lg["epoch"] + 1 + offset
        for ax, m in zip(axes, ["loss", "accuracy", "auc"]):
            ax.plot(ep, lg[m], "-o", label=f"{stage} train")
            ax.plot(ep, lg[f"val_{m}"], "--o", label=f"{stage} val")
            ax.set_title(f"{name} -- {m}"); ax.set_xlabel("epoch"); ax.legend()
        offset = ep.max()
    plt.tight_layout(); savefig(f"{name}_training_curves.png")
