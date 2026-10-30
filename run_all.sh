#!/usr/bin/env bash
# Run the whole SRGAN mid-term pipeline, stage by stage.
#   bash run_all.sh                 # everything (skips stages that already finished)
#   FORCE=1 bash run_all.sh         # re-run every stage (training still resumes from checkpoints)
#   bash run_all.sh 03_srgan        # run a single stage by its tag (see STAGES below)
#
# Logs:  outputs/logs/<stage>.log      -- one clean log per stage (written by Python logging)
#        outputs/logs/run_all.log      -- everything printed to the terminal, incl. errors / tracebacks
#        outputs/logs/<stage>.done     -- marker written when a stage finished successfully
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] && source .venv/bin/activate
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
OUT="${OUTPUT_DIR:-outputs}"
LOGS="$OUT/logs"; mkdir -p "$LOGS"
MASTER="$LOGS/run_all.log"

STAGES=(
  "01_prepare   src/01_prepare_data.py"
  "02_model_A   src/02_train_classifier.py --model A"
  "03_srgan     src/03_train_srgan.py"
  "04_eval_sr   src/04_evaluate_srgan.py"
  "05_gen_sr    src/05_generate_sr_dataset.py"
  "06_model_B   src/02_train_classifier.py --model B"
  "07_compare   src/07_compare_models.py"
)

run_stage() {
  local tag="$1"; shift
  if [ -f "$LOGS/$tag.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date '+%F %T')] skip $tag (already done -- delete $LOGS/$tag.done to re-run)" | tee -a "$MASTER"
    return
  fi
  echo "[$(date '+%F %T')] >>> START $tag: python $*" | tee -a "$MASTER"
  local t0=$SECONDS
  python "$@" 2>&1 | tee -a "$MASTER"
  echo "[$(date '+%F %T')] <<< DONE  $tag in $(( (SECONDS - t0) / 60 )) min" | tee -a "$MASTER"
  date '+%F %T' > "$LOGS/$tag.done"
}

echo "[$(date '+%F %T')] ===== run_all.sh on $(hostname) | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} =====" | tee -a "$MASTER"
for s in "${STAGES[@]}"; do
  read -r tag cmd <<< "$s"
  if [ $# -gt 0 ] && [ "$1" != "$tag" ]; then continue; fi
  # shellcheck disable=SC2086
  run_stage "$tag" $cmd
done

# Execute the Jupyter report (figures + tables) so the outputs are visible inside a notebook.
if [ $# -eq 0 ]; then
  echo "[$(date '+%F %T')] >>> building report.ipynb" | tee -a "$MASTER"
  jupyter nbconvert --to notebook --execute report.ipynb --output report_executed.ipynb 2>&1 | tee -a "$MASTER"
  echo "[$(date '+%F %T')] ===== pipeline finished =====" | tee -a "$MASTER"
fi
