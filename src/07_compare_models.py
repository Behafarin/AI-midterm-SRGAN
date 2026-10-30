"""Stage 7 -- compare Model A and Model B on the 30% test set (real / SRGAN / bicubic versions).
Metrics: Accuracy, Precision, Recall, F1, AUC; plus ROC curves, confusion matrices, reports."""
import os

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report)

import config as C
import common as U

log = U.setup("07_compare_models")
_, _, _, _, X_test, y_test = U.load_data()
_, _, SR_test = U.load_sr_data()
BIC_test = np.concatenate([(U.bicubic_up(U.to_lr(X_test[i:i + 512])).numpy() * 255.0).round().astype(np.uint8)
                           for i in range(0, len(X_test), 512)])
test_sets = {"real 128x128": X_test, "SRGAN 128x128": SR_test, "bicubic 128x128": BIC_test}
models = {m: keras.models.load_model(os.path.join(C.MODEL_DIR, f"model_{m[-1]}_final.keras"))
          for m in ["Model A", "Model B"]}

rows, probs = [], {}
for mname, model in models.items():
    for tname, X in test_sets.items():
        p = model.predict(U.make_cls_ds(X, np.zeros(len(X)), False), verbose=0).ravel()
        pred = (p > 0.5).astype(int)
        probs[(mname, tname)] = p
        rows.append({"model": mname, "test_images": tname,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred, zero_division=0),
                     "F1": f1_score(y_test, pred, zero_division=0),
                     "AUC": roc_auc_score(y_test, p)})
        np.save(os.path.join(C.RESULT_DIR, f"probs_{mname[-1]}_{tname.split()[0]}.npy"), p)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(C.RESULT_DIR, "model_comparison.csv"), index=False)
log.info("Model comparison on the test set (positive class = dog):\n" + res.round(4).to_string(index=False))

pairs = [("Model A", "real 128x128"), ("Model B", "SRGAN 128x128")]
with open(os.path.join(C.RESULT_DIR, "classification_reports.txt"), "w") as f:
    for mname, model in models.items():
        for tname in test_sets:
            rep = classification_report(y_test, (probs[(mname, tname)] > 0.5).astype(int),
                                        target_names=C.CLASSES, digits=4, zero_division=0)
            f.write(f"===== {mname} on {tname} =====\n{rep}\n")
            if (mname, tname) in pairs:
                log.info(f"{mname} on {tname}:\n{rep}")

# ---- Figures ------------------------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
fig, ax = plt.subplots(figsize=(14, 4.5)); w = 0.13
for k, (_, r) in enumerate(res.iterrows()):
    ax.bar(np.arange(5) + (k - 2.5) * w, r[metrics].astype(float), w, label=f"{r['model']} -- {r['test_images']}")
ax.set_xticks(np.arange(5)); ax.set_xticklabels(metrics)
ax.set_ylim(max(0.0, res[metrics].values.min() - 0.05), 1.0); ax.set_ylabel("score")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.set_title("Model A vs. Model B on the 30% test set")
plt.tight_layout(); U.savefig("07_metrics_comparison.png")

styles = {"real 128x128": "-", "SRGAN 128x128": "--", "bicubic 128x128": ":"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mname in zip(axes, models):
    for tname in test_sets:
        fpr, tpr, _ = roc_curve(y_test, probs[(mname, tname)])
        auc = res[(res.model == mname) & (res.test_images == tname)].AUC.item()
        ax.plot(fpr, tpr, styles[tname], label=f"{tname} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_title(f"ROC -- {mname}"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right")
plt.tight_layout(); U.savefig("08_roc_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (mname, tname) in zip(axes, pairs):
    cm = confusion_matrix(y_test, (probs[(mname, tname)] > 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{mname} on {tname}")
plt.tight_layout(); U.savefig("09_confusion_matrices.png")
log.info("STAGE 07_compare_models finished.")
