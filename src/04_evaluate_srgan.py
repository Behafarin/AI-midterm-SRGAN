"""Stage 4 -- SRGAN results: loss curves, visual samples on test images, PSNR/SSIM vs. bicubic."""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import config as C
import common as U
import srgan as S

log = U.setup("04_evaluate_srgan")
generator = S.load_generator(os.path.join(C.MODEL_DIR, "srgan_generator_final.keras"))
_, _, _, _, X_test, _ = U.load_data()

# ---- Loss curves --------------------------------------------------------------------------------
gl = pd.read_csv(os.path.join(C.RESULT_DIR, "srgan_loss_log.csv")).drop_duplicates("epoch", keep="last")
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
axes[0].plot(gl.epoch, gl.content_loss, label="content (VGG54)")
axes[0].plot(gl.epoch, gl.g_loss, label="total generator", alpha=0.7); axes[0].set_title("Generator loss"); axes[0].legend()
axes[1].plot(gl.epoch, gl.d_loss, color="tab:red"); axes[1].set_title("Discriminator loss")
axes[2].plot(gl.epoch, gl.d_acc_real, label="D acc. on real")
axes[2].plot(gl.epoch, gl.d_acc_fake, label="D acc. on generated"); axes[2].set_title("Discriminator accuracy"); axes[2].legend()
for ax in axes:
    ax.set_xlabel("adversarial epoch")
plt.tight_layout(); U.savefig("04_srgan_losses.png")

# ---- Visual comparison on unseen test images ----------------------------------------------------------
show_idx = np.load(os.path.join(C.CACHE_DIR, "show_idx.npy"))
lr = U.to_lr(X_test[show_idx])
rows = [(lr.numpy(), "input 32x32"), (U.bicubic_up(lr).numpy(), "bicubic 128x128"),
        (S.sr_images(generator, lr).numpy(), "SRGAN 128x128"), (X_test[show_idx] / 255.0, "original 128x128")]
fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
for r, (imgs, title) in enumerate(rows):
    for j in range(6):
        axes[r, j].imshow(imgs[j], interpolation="nearest" if r == 0 else None)
        axes[r, j].set_title(title, fontsize=9); axes[r, j].axis("off")
plt.suptitle("SRGAN on unseen test images"); plt.tight_layout(); U.savefig("05_srgan_test_examples.png")

# ---- PSNR / SSIM on the whole test set -----------------------------------------------------------------
pb, ps, sb, ss = [], [], [], []
for i in range(0, len(X_test), 128):
    hr = tf.cast(X_test[i:i + 128], tf.float32) / 255.0
    lr_b = U.to_lr(X_test[i:i + 128])
    bic, sr = U.bicubic_up(lr_b), S.sr_images(generator, lr_b)
    pb.append(tf.image.psnr(hr, bic, 1.0)); ps.append(tf.image.psnr(hr, sr, 1.0))
    sb.append(tf.image.ssim(hr, bic, 1.0)); ss.append(tf.image.ssim(hr, sr, 1.0))
m = lambda v: float(tf.reduce_mean(tf.concat(v, 0)))
q = pd.DataFrame({"method": ["bicubic", "SRGAN"], "PSNR_dB": [m(pb), m(ps)], "SSIM": [m(sb), m(ss)]})
q.to_csv(os.path.join(C.RESULT_DIR, "sr_quality_test.csv"), index=False)
log.info(f"Super-resolution quality on {len(X_test)} test images:\n" + q.round(4).to_string(index=False))
log.info("STAGE 04_evaluate_srgan finished.")
