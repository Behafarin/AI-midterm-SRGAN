"""Stage 5 -- pass every image (downscaled to 32x32) through the trained generator and cache the
SRGAN 128x128 dataset (cache/SR_tr.npy, SR_val.npy, SR_test.npy) for Model B."""
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import config as C
import common as U
import srgan as S

log = U.setup("05_generate_sr_dataset")

if all(os.path.exists(p) for p in U.SR_FILES.values()):
    log.info("SR dataset already cached -- skipping generation.")
else:
    generator = S.load_generator(os.path.join(C.MODEL_DIR, "srgan_generator_final.keras"))
    X_tr, _, X_val, _, X_test, _ = U.load_data()

    def generate(X, batch=256):
        return np.concatenate([(S.sr_images(generator, U.to_lr(X[i:i + batch])).numpy() * 255.0).round().astype(np.uint8)
                               for i in range(0, len(X), batch)])

    t0 = time.time()
    for k, X in [("tr", X_tr), ("val", X_val), ("test", X_test)]:
        arr = generate(X)
        np.save(U.SR_FILES[k], arr)
        log.info(f"SR_{k}: {arr.shape} saved")
    log.info(f"Generation took {time.time() - t0:.0f}s")

X_tr, y_tr, *_ = U.load_data()
SR_tr, _, _ = U.load_sr_data()
fig, axes = plt.subplots(2, 8, figsize=(17, 4.8))
for j in range(8):
    axes[0, j].imshow(X_tr[j]); axes[0, j].set_title(f"real -- {C.CLASSES[y_tr[j]]}", fontsize=9)
    axes[1, j].imshow(SR_tr[j]); axes[1, j].set_title("SRGAN (from 32x32)", fontsize=9)
    axes[0, j].axis("off"); axes[1, j].axis("off")
plt.suptitle("Training images for Model A (top) vs. Model B (bottom)")
plt.tight_layout(); U.savefig("06_model_B_training_images.png")
log.info("STAGE 05_generate_sr_dataset finished.")
