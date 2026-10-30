"""Stage 1 -- download the dataset, resize to 128x128, stratified 70/30 split, cache as .npy.
Also saves: scaled-image examples, augmentation samples, normalization histogram."""
import os
import zipfile
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

import config as C
import common as U

log = U.setup("01_prepare_data")


def find_images(root_dir, exts=(".jpg", ".jpeg", ".png")):
    """Label each image from its folder or file name (same helper as Assignment 1)."""
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        parent = os.path.basename(dirpath).lower()
        for fname in sorted(filenames):
            if not fname.lower().endswith(exts):
                continue
            lname = fname.lower()
            if "cat" in parent or lname.startswith("cat"):
                label = 0
            elif "dog" in parent or lname.startswith("dog"):
                label = 1
            else:
                continue
            found.append((os.path.join(dirpath, fname), label))
    return sorted(found)


def load_resized(paths):
    imgs, keep = [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                imgs.append(np.asarray(im.convert("RGB").resize((C.HR_SIZE, C.HR_SIZE), Image.BICUBIC), dtype=np.uint8))
            keep.append(i)
        except Exception as e:
            log.warning(f"skipping unreadable image {p}: {e}")
        if (i + 1) % 5000 == 0:
            log.info(f"  {i + 1}/{len(paths)} images resized")
    return np.stack(imgs), np.array(keep)


# ---- 1. Download --------------------------------------------------------------------------------
if all(os.path.exists(p) for p in U.CACHE_FILES.values()):
    log.info("Cached arrays already exist -- skipping download and resizing.")
else:
    if not os.path.isdir(RAW := C.RAW_DIR) or not find_images(RAW):
        log.info(f"Downloading {C.KAGGLE_DATASET_SLUG} into {RAW} (needs ~/.kaggle/kaggle.json)")
        os.makedirs(RAW, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", C.KAGGLE_DATASET_SLUG, "-p", RAW, "--unzip"], check=True)
        for fname in os.listdir(RAW):          # some re-uploads nest a second zip
            if fname.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(RAW, fname)) as z:
                    z.extractall(RAW)
    else:
        log.info(f"Raw dataset found in {RAW}")

    # ---- 2. Split + resize ----------------------------------------------------------------------
    df = pd.DataFrame(find_images(C.RAW_DIR), columns=["path", "label"])
    log.info(f"Labeled images found: {len(df)} (cats={int((df.label == 0).sum())}, dogs={int((df.label == 1).sum())})")
    train_df, test_df = train_test_split(df, test_size=C.TEST_SIZE, stratify=df["label"], random_state=C.SEED)
    train_df.to_csv(os.path.join(C.RESULT_DIR, "split_train_files.csv"), index=False)
    test_df.to_csv(os.path.join(C.RESULT_DIR, "split_test_files.csv"), index=False)

    log.info("Resizing train split to 128x128 ...")
    X_train, keep = load_resized(train_df["path"].tolist()); y_train = train_df["label"].values[keep].astype(np.int64)
    log.info("Resizing test split to 128x128 ...")
    X_test, keep = load_resized(test_df["path"].tolist()); y_test = test_df["label"].values[keep].astype(np.int64)
    for k, arr in zip(["X_train", "y_train", "X_test", "y_test"], [X_train, y_train, X_test, y_test]):
        np.save(U.CACHE_FILES[k], arr)
    log.info(f"Saved cache to {C.CACHE_DIR}")

X_tr, y_tr, X_val, y_val, X_test, y_test = U.load_data()
y_train_all = np.concatenate([y_tr, y_val])
split = pd.DataFrame({
    "split": ["train (70%)", "train-fit", "validation (10% of train)", "test (30%)"],
    "cat": [(y_train_all == 0).sum(), (y_tr == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()],
    "dog": [(y_train_all == 1).sum(), (y_tr == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()],
})
split["total"] = split["cat"] + split["dog"]
split.to_csv(os.path.join(C.RESULT_DIR, "split_table.csv"), index=False)
log.info("Split table:\n" + split.to_string(index=False))

# ---- 3. Examples of scaled images ----------------------------------------------------------------
import matplotlib.pyplot as plt
show_idx = np.random.default_rng(C.SEED).choice(len(X_test), 6, replace=False)
np.save(os.path.join(C.CACHE_DIR, "show_idx.npy"), show_idx)   # same images reused in later figures
lr_show = U.to_lr(X_test[show_idx]).numpy()
fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
for j, i in enumerate(show_idx):
    axes[0, j].imshow(X_test[i]); axes[0, j].set_title(f"{C.CLASSES[y_test[i]]} -- 128x128")
    axes[1, j].imshow(lr_show[j], interpolation="nearest"); axes[1, j].set_title("downscaled 32x32")
    axes[0, j].axis("off"); axes[1, j].axis("off")
plt.suptitle("High-resolution (top) vs. 4x downscaled SRGAN input (bottom)")
plt.tight_layout(); U.savefig("01_scaled_examples.png")

# ---- 4. Transformations + normalization ------------------------------------------------------------
augment = U.make_augment()
fig, axes = plt.subplots(4, 5, figsize=(13, 10.5))
for r in range(4):
    axes[r, 0].imshow(X_tr[r]); axes[r, 0].set_title("original")
    for c in range(1, 5):
        out = augment(tf.cast(X_tr[r:r + 1], tf.float32), training=True)[0].numpy()
        axes[r, c].imshow(np.clip(out, 0, 255).astype(np.uint8)); axes[r, c].set_title(f"augmented #{c}")
    for c in range(5):
        axes[r, c].axis("off")
plt.suptitle("Random transformations applied to training images")
plt.tight_layout(); U.savefig("02_augmentations.png")

sample = X_tr[:32].astype(np.float32)
normed = sample / 127.5 - 1.0          # identical to the model's Rescaling(1/127.5, offset=-1) layer
log.info(f"Normalization: before min={sample.min():.0f} max={sample.max():.0f} mean={sample.mean():.1f} | "
         f"after min={normed.min():.2f} max={normed.max():.2f} mean={normed.mean():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].hist(sample.ravel(), bins=50, color="tab:blue"); axes[0].set_title("pixel values -- raw [0, 255]")
axes[1].hist(normed.ravel(), bins=50, color="tab:orange"); axes[1].set_title("pixel values -- normalized [-1, 1]")
plt.tight_layout(); U.savefig("03_normalization.png")

log.info("STAGE 01_prepare_data finished.")
