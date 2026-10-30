# How to run (lab server)

> Full documentation (README) will be written after the run finishes.

## Layout

```
AI-midterm/
├── setup.sh                 # one-time: uv venv + packages + check_setup.py
├── check_setup.py           # pre-flight: GPU, disk, Kaggle creds, ImageNet weights
├── run_all.sh               # runs all stages in order, logs everything, skips finished stages
├── requirements.txt
├── report.ipynb             # Jupyter view of all figures/tables (executed at the end -> report_executed.ipynb)
├── src/
│   ├── config.py            # all hyper-parameters (overridable via env vars)
│   ├── common.py            # logging, data, classifier pipeline
│   ├── srgan.py             # generator / discriminator / VGG19 loss
│   ├── 01_prepare_data.py   # download, resize 128x128, 70/30 split, sample figures
│   ├── 02_train_classifier.py --model A|B
│   ├── 03_train_srgan.py    # 10 warm-up + 150 adversarial epochs, checkpoint every 5
│   ├── 04_evaluate_srgan.py # loss curves, test samples, PSNR/SSIM vs bicubic
│   ├── 05_generate_sr_dataset.py
│   └── 07_compare_models.py # Accuracy / Precision / Recall / F1 / AUC, ROC, confusion matrices
└── outputs/                 # created by the run
    ├── logs/        <stage>.log (clean per-stage log), run_all.log (full terminal output), <stage>.done
    ├── results/     per-epoch CSV logs, metric tables, config_<stage>.json (exact settings used)
    ├── figures/     all plots + srgan_progress/ sample grid every 5 epochs
    ├── models/      final .keras models
    ├── checkpoints/ SRGAN checkpoints (every 5 epochs) + classifier checkpoints (every epoch)
    └── cache/       resized image arrays (.npy)
```

## Stages

| Tag | Script | Output |
|---|---|---|
| `01_prepare` | `01_prepare_data.py` | `cache/*.npy`, split table, figures 01–03 |
| `02_model_A` | `02_train_classifier.py --model A` | `models/model_A_final.keras` |
| `03_srgan` | `03_train_srgan.py` | `models/srgan_generator_final.keras` |
| `04_eval_sr` | `04_evaluate_srgan.py` | figures 04–05, `sr_quality_test.csv` |
| `05_gen_sr` | `05_generate_sr_dataset.py` | `cache/SR_*.npy`, figure 06 |
| `06_model_B` | `02_train_classifier.py --model B` | `models/model_B_final.keras` |
| `07_compare` | `07_compare_models.py` | `model_comparison.csv`, figures 07–09 |

Run one stage only: `bash run_all.sh 03_srgan`. Re-run a finished stage: delete `outputs/logs/<tag>.done`.
If a run crashes or is killed, run the same command again. Classifiers resume from the last epoch, the SRGAN from the last checkpoint.

## Settings you may want to change (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `SRGAN_EPOCHS` | 150 | adversarial epochs |
| `SRGAN_TRAIN_SUBSET` | None (all ~15.7k) | images per SRGAN epoch, e.g. `8000` to halve the time |
| `SAVE_EVERY` | 5 | SRGAN checkpoint interval |
| `SRGAN_BATCH` / `CLS_BATCH` | 16 / 32 | batch sizes |
| `CUDA_VISIBLE_DEVICES` | – | which GPU to use |
| `SMOKE_TEST` | – | `1` = 1–2 epochs of everything (quick bug check) |
