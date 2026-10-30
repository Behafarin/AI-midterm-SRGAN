"""Pre-flight check: run this before starting the long training run.
    python check_setup.py
"""
import os
import sys
import shutil
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ok = True


def check(name, cond, hint=""):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}" + (f"   -> {hint}" if not cond and hint else ""))
    ok &= bool(cond)


print("Environment check")
check(f"python {sys.version.split()[0]} (3.9-3.13)", (3, 9) <= sys.version_info[:2] <= (3, 13))

import tensorflow as tf, keras  # noqa: E402
print(f"  info  tensorflow {tf.__version__} | keras {keras.__version__}")
check("keras 3", keras.__version__.startswith("3"))
gpus = tf.config.list_physical_devices("GPU")
check(f"GPU visible to TensorFlow ({len(gpus)} found)", gpus,
      "check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and that tensorflow[and-cuda] installed")
if gpus:
    with tf.device("/GPU:0"):
        x = tf.random.normal((2048, 2048)); y = float(tf.reduce_sum(x @ x))
    check("GPU matmul runs", y == y)
try:
    print("  nvidia-smi:\n" + subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                                             "--format=csv"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    pass

import config as C  # noqa: E402
free_gb = shutil.disk_usage(C.OUTPUT_DIR).free / 1e9
check(f"free disk at {C.OUTPUT_DIR}: {free_gb:.0f} GB (need ~10 GB)", free_gb > 10)

cache_done = os.path.exists(os.path.join(C.CACHE_DIR, "X_train.npy"))
raw_present = os.path.isdir(C.RAW_DIR) and any(os.scandir(C.RAW_DIR))
kaggle_ok = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) or bool(os.environ.get("KAGGLE_KEY"))
check("dataset: cache, raw folder, or Kaggle credentials", cache_done or raw_present or kaggle_ok,
      "put kaggle.json in ~/.kaggle/ (chmod 600), or copy the dataset to data_raw/")
check("kaggle CLI on PATH", shutil.which("kaggle") or cache_done or raw_present)

print("Pretrained weights (downloads once to ~/.keras/models)")
try:
    keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    check("MobileNetV2 + VGG19 ImageNet weights", True)
except Exception as e:
    check("MobileNetV2 + VGG19 ImageNet weights", False, f"no internet to storage.googleapis.com? {e}")

print(f"  info  SRGAN: {C.SRGAN_PRETRAIN_EPOCHS} warm-up + {C.SRGAN_EPOCHS} adversarial epochs, "
      f"subset={C.SRGAN_TRAIN_SUBSET}, checkpoint every {C.SAVE_EVERY}")
print("\nALL CHECKS PASSED -- ready to run." if ok else "\nSome checks FAILED -- fix them before training.")
sys.exit(0 if ok else 1)
