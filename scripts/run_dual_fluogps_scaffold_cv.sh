#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK="${TASK:-both}"
SPLIT_SCHEME="${SPLIT_SCHEME:-both}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
FOLDS="${FOLDS:-0 1 2 3 4}"
DUAL_WEIGHT_MODE="${DUAL_WEIGHT_MODE:-separate}"

if [[ "${USE_CONDA:-1}" == "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-graphgps}"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [[ -x "/home/panshangyang/ENTER/envs/graphgps/bin/python" ]]; then
    PYTHON_BIN="/home/panshangyang/ENTER/envs/graphgps/bin/python"
  else
    echo "No python found. Activate graphgps or set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

read -r -a GPU_ARGS <<< "$GPUS"
read -r -a FOLD_ARGS <<< "$FOLDS"

echo "Dual-FluoGPS scaffold CV pipeline"
echo "  task: $TASK"
echo "  split_scheme: $SPLIT_SCHEME"
echo "  folds: $FOLDS"
echo "  gpus: $GPUS"
echo "  dual_weight_mode: $DUAL_WEIGHT_MODE"
echo "  python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

"$PYTHON_BIN" scripts/preprocess_scaffold_cv_data.py \
  --split_scheme "$SPLIT_SCHEME" \
  --task "$TASK" \
  --dual_graph \
  --folds "${FOLD_ARGS[@]}"

"$PYTHON_BIN" scripts/5_cross_validation_dual_fluogps.py \
  --split_scheme "$SPLIT_SCHEME" \
  --task "$TASK" \
  --dual_graph \
  --dual_weight_mode "$DUAL_WEIGHT_MODE" \
  --folds "${FOLD_ARGS[@]}" \
  --gpus "${GPU_ARGS[@]}" \
  --require_preprocessed
