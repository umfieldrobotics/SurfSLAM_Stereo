#!/usr/bin/env bash
# Single entry point for every training run. An experiment is a model crossed
# with an ablation.
#
#   ./train_scripts/launch.sh <model>/<ablation> [gpu_list] [options] [key=value ...]
#   ./train_scripts/launch.sh --model <model> --ablation <ablation> [...]
#
#   gpu_list    comma-separated device ids, e.g. 0,1,2. Defaults to every GPU
#               nvidia-smi reports. Accepted as a bare first argument or as
#               --gpus 0,1.
#
# Options:
#   --model <m>       model from config/train/models/
#   --ablation <a>    ablation from config/train/ablations/
#   --gpus <ids>      same as the positional gpu list
#   --name <str>      run-name prefix; defaults to <model>_<ablation>
#   --timm <ver>      pip install timm==<ver> before launching
#   --no-pip          skip the timm install (also: SURF_SKIP_PIP=1)
#   --list            print the available models and ablations
#   --dry-run         print the command instead of running it
#
# Anything else is forwarded to train.py, so OmegaConf overrides work as before:
#   ./train_scripts/launch.sh defom_stereo/warp_finetune_full 0,1 optimization.num_steps=200
#
# config/train/models/<model>/<ablation>.yaml is used when it exists (those files carry
# per-pair tuning); otherwise the two axes are composed directly.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
CONFIG_DIR="$REPO_DIR/config/train"

usage () { sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

list_ablations () {
  ( cd "$CONFIG_DIR/ablations" && ls *.yaml 2>/dev/null | sed 's/\.yaml$//' | sort | tr '\n' ' ' )
}
list_models () {
  for d in "$CONFIG_DIR"/models/*/; do
    m=$(basename "$d"); [[ -f "$d/$m.yaml" ]] && echo "$m"
  done | sort | tr '\n' ' '
}
list_all () {
  echo "models:    $(list_models)"
  echo "ablations: $(list_ablations)"
}
model_file () {  # $1 = model name; prints its yaml path if there is one
  [[ -f "$CONFIG_DIR/models/$1/$1.yaml" ]] && echo "$CONFIG_DIR/models/$1/$1.yaml"
}

MODEL=""
ABLATION=""
GPUS=""
RUN_NAME_PREFIX=""
DRY_RUN=0
SKIP_PIP="${SURF_SKIP_PIP:-0}"
TIMM_VERSION="${TIMM_VERSION:-unset}"
FORWARD=()

if [[ $# -eq 0 ]]; then usage; echo; list_all; exit 1; fi

# A leading <model>/<ablation>, then optionally a bare gpu list -- the calling
# convention of the per-experiment wrappers.
case "$1" in
  -h|--help) usage; echo; list_all; exit 0 ;;
  --list)    list_all; exit 0 ;;
  -*)        ;;
  *)         MODEL="${1%%/*}"; ABLATION="${1#*/}"; ABLATION="${ABLATION%.yaml}"; shift ;;
esac
if [[ $# -gt 0 && $1 =~ ^[0-9]+(,[0-9]+)*$ ]]; then GPUS=$1; shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2"; shift 2 ;;
    --model=*)   MODEL="${1#*=}"; shift ;;
    --ablation)  ABLATION="$2"; shift 2 ;;
    --ablation=*) ABLATION="${1#*=}"; shift ;;
    --gpus)      GPUS="$2"; shift 2 ;;
    --gpus=*)    GPUS="${1#*=}"; shift ;;
    --name)      RUN_NAME_PREFIX="$2"; shift 2 ;;
    --name=*)    RUN_NAME_PREFIX="${1#*=}"; shift ;;
    --timm)      TIMM_VERSION="$2"; shift 2 ;;
    --timm=*)    TIMM_VERSION="${1#*=}"; shift ;;
    --no-pip)    SKIP_PIP=1; shift ;;
    --list)      list_all; exit 0 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   usage; echo; list_all; exit 0 ;;
    *)           FORWARD+=("$1"); shift ;;
  esac
done

# ---------- validate the pair ------------------------------------------------
if [[ -z "$MODEL" || -z "$ABLATION" || "$MODEL" == "$ABLATION" ]]; then
  echo "error: name an experiment, e.g. 'defom_stereo/warp_finetune_full' or --model defom_stereo --ablation warp_finetune_full" >&2
  list_all >&2
  exit 1
fi

fail=0
[[ -n "$(model_file "$MODEL")" ]] || { echo "error: no model '$MODEL'" >&2; fail=1; }
[[ -f "$CONFIG_DIR/ablations/$ABLATION.yaml" ]] || { echo "error: no ablation '$ABLATION'" >&2; fail=1; }
if [[ "$fail" == "1" ]]; then list_all >&2; exit 1; fi

CONFIG="$MODEL/$ABLATION"
if [[ -f "$CONFIG_DIR/models/$MODEL/$ABLATION.yaml" ]]; then
  CONFIG_SOURCE="models/$MODEL/$ABLATION.yaml (model + ablation + per-pair tuning)"
else
  CONFIG_SOURCE="models/$MODEL/$MODEL.yaml + ablations/$ABLATION.yaml (composed)"
fi
: "${RUN_NAME_PREFIX:=${MODEL}_${ABLATION}}"

# ---------- timm version per model -------------------------------------------
# IGEV++ pins an old timm; DEFOM-Stereo wants a current one. The
# FoundationStereo scripts never installed anything, so neither do we.
if [[ "$TIMM_VERSION" == "unset" ]]; then
  case "$MODEL" in
    igev_pp*)       TIMM_VERSION="0.5.4"  ;;
    defom_stereo*)  TIMM_VERSION="1.0.22" ;;
    *)              TIMM_VERSION=""       ;;
  esac
fi

if [[ -z "$GPUS" ]]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd ',' -)
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -ra ids <<<"$GPUS"
NGPUS=${#ids[@]}

# ---------- runtime environment ---------------------------------------------
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1             # comment out if you really have IB
PORT=$((10000 + RANDOM % 20000))     # avoid collisions

if [[ -n "$TIMM_VERSION" && "$SKIP_PIP" != "1" ]]; then
  pip3 install "timm==$TIMM_VERSION"
fi

CURRENT_DATE_STR=$(date +%y%m%d_%H%M%S)
RUN_NAME="${RUN_NAME_PREFIX}_${CURRENT_DATE_STR}"

RUN_ARGS=(
  "config_path=$CONFIG"
  "name=$RUN_NAME"
  "logging.wandb_name=$RUN_NAME"
)

if [[ "$NGPUS" -gt 1 ]]; then
  CMD=( python3 -m torch.distributed.run --standalone --master_port="$PORT"
        --nproc_per_node="$NGPUS" train.py "${RUN_ARGS[@]}" )
else
  CMD=( python3 train.py "${RUN_ARGS[@]}" )
fi
CMD+=( "${FORWARD[@]+"${FORWARD[@]}"}" )

echo "experiment: $CONFIG"
echo "config:     $CONFIG_SOURCE"
echo "run name:   $RUN_NAME"
echo "GPUs:       $GPUS ($NGPUS)"
echo "command:    ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

# Refuse to train against unpatched submodules. Without this the failure surfaces as
# a confusing ImportError (or, worse, silently wrong numbers if a pin moved).
if ! "$REPO_DIR/scripts/setup_submodules.sh" --check >/dev/null 2>&1; then
  echo "error: submodules are not in the expected state." >&2
  "$REPO_DIR/scripts/setup_submodules.sh" --check >&2 || true
  echo "Run ./scripts/setup_submodules.sh (or --reset) first." >&2
  exit 1
fi

# train.py resolves a few paths relative to the CWD (the augmentation config,
# ./runs/), so always launch from the repo root.
cd "$REPO_DIR"
exec "${CMD[@]}"
