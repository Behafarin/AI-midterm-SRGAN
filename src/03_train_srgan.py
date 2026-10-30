"""Stage 3 -- train the SRGAN (32x32 -> 128x128).

MSE warm-up of the generator for SRGAN_PRETRAIN_EPOCHS, then SRGAN_EPOCHS adversarial epochs.
A full checkpoint (G, D, both optimizers, epoch counters) is written every SAVE_EVERY epochs;
re-running the script resumes from the latest one.
"""
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

import config as C
import common as U
import srgan as S

log = U.setup("03_train_srgan")

X_tr, _, X_val, _, _, _ = U.load_data()
X_srgan = X_tr
if C.SRGAN_TRAIN_SUBSET and C.SRGAN_TRAIN_SUBSET < len(X_tr):
    X_srgan = X_tr[np.random.default_rng(C.SEED).choice(len(X_tr), C.SRGAN_TRAIN_SUBSET, replace=False)]
log.info(f"SRGAN training images per epoch: {len(X_srgan)} | batch {C.SRGAN_BATCH} | "
         f"pre-train {C.SRGAN_PRETRAIN_EPOCHS} + adversarial {C.SRGAN_EPOCHS} epochs | save every {C.SAVE_EVERY}")


def srgan_pair(hr_uint8):
    hr = tf.image.random_flip_left_right(tf.cast(hr_uint8, tf.float32) / 255.0)
    lr = tf.image.resize(hr[None], (C.LR_SIZE, C.LR_SIZE), method="bicubic", antialias=True)[0]
    return tf.clip_by_value(lr, 0.0, 1.0), hr * 2.0 - 1.0      # LR [0,1], HR [-1,1]


ds = (tf.data.Dataset.from_tensor_slices(X_srgan)
      .shuffle(len(X_srgan), seed=C.SEED, reshuffle_each_iteration=True)
      .map(srgan_pair, num_parallel_calls=U.AUTOTUNE)
      .batch(C.SRGAN_BATCH, drop_remainder=True)
      .prefetch(U.AUTOTUNE))

mon_hr = tf.cast(X_val[:4], tf.float32) / 255.0      # fixed validation images to watch progress
mon_lr = U.to_lr(X_val[:4])

# ---- Models, optimizers, checkpoint ---------------------------------------------------------------
generator, discriminator, vgg = S.build_generator(), S.build_discriminator(), S.build_vgg_extractor()
log.info(f"Generator params: {generator.count_params():,} | Discriminator params: {discriminator.count_params():,}")
generator.summary(print_fn=lambda s, **kw: log.info(s), line_length=110)

mse = keras.losses.MeanSquaredError()
bce = keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = keras.optimizers.Adam(C.SRGAN_LR, beta_1=0.9)
d_opt = keras.optimizers.Adam(C.SRGAN_LR, beta_1=0.9)
g_opt.build(generator.trainable_variables)
d_opt.build(discriminator.trainable_variables)

pretrain_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
gan_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator, g_opt=g_opt, d_opt=d_opt,
                           pretrain_epoch=pretrain_epoch, gan_epoch=gan_epoch)
manager = tf.train.CheckpointManager(ckpt, os.path.join(C.CKPT_DIR, "srgan"), max_to_keep=3)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    log.info(f"RESUMED from {manager.latest_checkpoint}: pre-train epoch {int(pretrain_epoch.numpy())}, "
             f"adversarial epoch {int(gan_epoch.numpy())}")
else:
    log.info("No SRGAN checkpoint found -- starting from scratch.")

PRETRAIN_LOG = os.path.join(C.RESULT_DIR, "srgan_pretrain_log.csv")
GAN_LOG = os.path.join(C.RESULT_DIR, "srgan_loss_log.csv")
PROGRESS_DIR = os.path.join(C.FIG_DIR, "srgan_progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)


def log_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_progress(tag):
    sr = S.sr_images(generator, mon_lr).numpy()
    bic = U.bicubic_up(mon_lr).numpy()
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for j in range(4):
        for r, (img, t) in enumerate([(bic[j], "bicubic"), (sr[j], "SRGAN"), (mon_hr[j].numpy(), "original HR")]):
            axes[r, j].imshow(img); axes[r, j].set_title(t); axes[r, j].axis("off")
    plt.suptitle(tag); plt.tight_layout()
    plt.savefig(os.path.join(PROGRESS_DIR, f"{tag.replace(' ', '_')}.png"), dpi=90); plt.close(fig)
    psnr = float(tf.reduce_mean(tf.image.psnr(mon_hr, S.sr_images(generator, mon_lr), 1.0)))
    return psnr


def checkpoint(tag):
    path = manager.save()
    generator.save(os.path.join(C.MODEL_DIR, "srgan_generator_latest.keras"))
    psnr = save_progress(tag)
    log.info(f"   checkpoint saved: {path} | monitor PSNR={psnr:.2f} dB")


@tf.function
def pretrain_step(lr, hr):
    with tf.GradientTape() as tape:
        loss = mse(hr, generator(lr, training=True))
    g_opt.apply_gradients(zip(tape.gradient(loss, generator.trainable_variables), generator.trainable_variables))
    return loss


@tf.function
def gan_step(lr, hr):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        sr = generator(lr, training=True)
        real_logits = discriminator(hr, training=True)
        fake_logits = discriminator(sr, training=True)
        content = mse(S.vgg_features(vgg, hr), S.vgg_features(vgg, sr))
        adv = bce(tf.ones_like(fake_logits), fake_logits)
        g_loss = content + C.ADV_WEIGHT * adv
        d_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)
    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, generator.trainable_variables), generator.trainable_variables))
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    return (g_loss, content, adv, d_loss,
            tf.reduce_mean(tf.cast(real_logits > 0, tf.float32)), tf.reduce_mean(tf.cast(fake_logits < 0, tf.float32)))


# ---- Stage 1: MSE warm-up -----------------------------------------------------------------------
for epoch in range(int(pretrain_epoch.numpy()), C.SRGAN_PRETRAIN_EPOCHS):
    t0, losses = time.time(), []
    for lr_b, hr_b in ds:
        losses.append(float(pretrain_step(lr_b, hr_b)))
    pretrain_epoch.assign(epoch + 1)
    log_row(PRETRAIN_LOG, {"epoch": epoch + 1, "mse": np.mean(losses), "seconds": time.time() - t0})
    log.info(f"[pre-train {epoch + 1}/{C.SRGAN_PRETRAIN_EPOCHS}] MSE={np.mean(losses):.5f} ({time.time() - t0:.0f}s)")
    if (epoch + 1) % C.SAVE_EVERY == 0 or epoch + 1 == C.SRGAN_PRETRAIN_EPOCHS:
        checkpoint(f"pretrain epoch {epoch + 1:03d}")

# ---- Stage 2: adversarial training -----------------------------------------------------------------
for epoch in range(int(gan_epoch.numpy()), C.SRGAN_EPOCHS):
    lr_now = C.SRGAN_LR if epoch < C.SRGAN_LR_DECAY_AT else C.SRGAN_LR / 10
    g_opt.learning_rate.assign(lr_now); d_opt.learning_rate.assign(lr_now)
    t0, sums, n = time.time(), np.zeros(6), 0
    for lr_b, hr_b in ds:
        sums += np.array([float(v) for v in gan_step(lr_b, hr_b)]); n += 1
    g, c, a, d, dr, dfk = sums / n
    gan_epoch.assign(epoch + 1)
    secs = time.time() - t0
    log_row(GAN_LOG, {"epoch": epoch + 1, "g_loss": g, "content_loss": c, "adv_loss": a, "d_loss": d,
                      "d_acc_real": dr, "d_acc_fake": dfk, "lr": lr_now, "seconds": secs})
    log.info(f"[SRGAN {epoch + 1:3d}/{C.SRGAN_EPOCHS}] G={g:.5f} (content {c:.5f}, adv {a:.3f}) "
             f"D={d:.3f} (acc real {dr:.2f} / fake {dfk:.2f}) lr={lr_now:.0e} ({secs:.0f}s)")
    if (epoch + 1) % C.SAVE_EVERY == 0 or epoch + 1 == C.SRGAN_EPOCHS:
        checkpoint(f"gan epoch {epoch + 1:03d}")

generator.save(os.path.join(C.MODEL_DIR, "srgan_generator_final.keras"))
log.info(f"SRGAN complete: {int(gan_epoch.numpy())} adversarial epochs. Saved models/srgan_generator_final.keras")
log.info("STAGE 03_train_srgan finished.")
