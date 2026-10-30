"""Stage 2 / 6 -- train a MobileNetV2 transfer-learning classifier.

    python src/02_train_classifier.py --model A    # real 128x128 images
    python src/02_train_classifier.py --model B    # SRGAN-generated 128x128 images (needs stage 5)
"""
import argparse

import config as C
import common as U

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["A", "B"], required=True)
args = p.parse_args()

name = f"model_{args.model}"
log = U.setup(f"train_{name}")

X_tr, y_tr, X_val, y_val, _, _ = U.load_data()
if args.model == "B":
    X_tr, X_val, _ = U.load_sr_data()
    log.info("Training data: SRGAN-generated images (cache/SR_*.npy)")
else:
    log.info("Training data: real 128x128 images")
log.info(f"train-fit: {X_tr.shape}  validation: {X_val.shape}")

model = U.train_classifier(name, U.make_cls_ds(X_tr, y_tr, True), U.make_cls_ds(X_val, y_val, False), log)
val = model.evaluate(U.make_cls_ds(X_val, y_val, False), verbose=0, return_dict=True)
log.info(f"[{name}] final validation: " + "  ".join(f"{k}={v:.4f}" for k, v in val.items()))
U.plot_cls_logs(name)
log.info(f"STAGE train_{name} finished.")
