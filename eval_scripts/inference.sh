#!/usr/bin/env bash
# Run a stereo model on evaluation data and write per-frame predictions and
# metrics. An evaluation is a model crossed with a suite -- the same shape as
# train_scripts/launch.sh, whose conventions this deliberately mirrors.
# score.sh and reconstruction.sh post-process outputs; this is the only script
# here that loads a model.
#
#   ./eval_scripts/inference.sh <model>/<suite> [gpu_list] [options] [key=value ...]
#   ./eval_scripts/inference.sh --model <model> --suite <suite> [...]
#
#   gpu_list    comma-separated device ids, e.g. 0,1,2. Defaults to every GPU
#               nvidia-smi reports. Accepted as a bare first argument or as
#               --gpus 0,1.
#
# Options:
#   --model <m>       model from config/eval/models/
#   --suite <s>       suite from config/eval/suites/
#   --gpus <ids>      same as the positional gpu list
#   --name <str>      run-name prefix; defaults to <model>_<suite>
#   --timm <ver>      pip install timm==<ver> before launching
#   --no-pip          skip the timm install (also: SURF_SKIP_PIP=1)
#   --list            print the available models and suites
#   --dry-run         print the command instead of running it
#
# Anything else is forwarded to evaluate.py, so OmegaConf overrides work:
#   ./eval_scripts/inference.sh defom_stereo/suds_test 0 evaluation.max_frames=8
#   ./eval_scripts/inference.sh defom_stereo/suds_test 0 evaluation.checkpoints=[ours,full_pretrain]
#
# config/eval/models/<model>/<suite>.yaml is used when it exists (for per-pair
# tuning); otherwise the two axes are composed directly.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
CONFIG_DIR="$REPO_DIR/config/eval"

usage () { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

list_suites () {
  ( cd "$CONFIG_DIR/suites" && ls *.yaml 2>/dev/null | sed 's/\.yaml$//' | sort | tr '\n' ' ' )
}
list_models () {
  for d in "$CONFIG_DIR"/models/*/; do
    m=$(basename "$d"); [[ -f "$d/$m.yaml" ]] && echo "$m"
  done | sort | tr '\n' ' '
}
list_all () {
  echo "models: $(list_models)"
  echo "suites: $(list_suites)"
}
model_file () {  # $1 = model name; prints its yaml path if there is one
  [[ -f "$CONFIG_DIR/models/$1/$1.yaml" ]] && echo "$CONFIG_DIR/models/$1/$1.yaml"
}

MODEL=""
SUITE=""
GPUS=""
RUN_NAME_PREFIX=""
DRY_RUN=0
SKIP_PIP="${SURF_SKIP_PIP:-0}"
TIMM_VERSION="${TIMM_VERSION:-unset}"
FORWARD=()

if [[ $# -eq 0 ]]; then usage; echo; list_all; exit 1; fi

# A leading <model>/<suite>, then optionally a bare gpu list.
case "$1" in
  -h|--help) usage; echo; list_all; exit 0 ;;
  --list)    list_all; exit 0 ;;
  -*)        ;;
  *)         MODEL="${1%%/*}"; SUITE="${1#*/}"; SUITE="${SUITE%.yaml}"; shift ;;
esac
if [[ $# -gt 0 && $1 =~ ^[0-9]+(,[0-9]+)*$ ]]; then GPUS=$1; shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2"; shift 2 ;;
    --model=*)   MODEL="${1#*=}"; shift ;;
    --suite)     SUITE="$2"; shift 2 ;;
    --suite=*)   SUITE="${1#*=}"; shift ;;
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
if [[ -z "$MODEL" || -z "$SUITE" || "$MODEL" == "$SUITE" ]]; then
  echo "error: name an evaluation, e.g. 'defom_stereo/suds_test' or --model defom_stereo --suite suds_test" >&2
  list_all >&2
  exit 1
fi

fail=0
[[ -n "$(model_file "$MODEL")" ]] || { echo "error: no model '$MODEL'" >&2; fail=1; }
[[ -f "$CONFIG_DIR/suites/$SUITE.yaml" ]] || { echo "error: no suite '$SUITE'" >&2; fail=1; }
if [[ "$fail" == "1" ]]; then list_all >&2; exit 1; fi

CONFIG="$MODEL/$SUITE"
if [[ -f "$CONFIG_DIR/models/$MODEL/$SUITE.yaml" ]]; then
  CONFIG_SOURCE="models/$MODEL/$SUITE.yaml (model + suite + per-pair tuning)"
else
  CONFIG_SOURCE="models/$MODEL/$MODEL.yaml + suites/$SUITE.yaml (composed)"
fi
: "${RUN_NAME_PREFIX:=${MODEL}_${SUITE}}"

# ---------- timm version per model -------------------------------------------
# Same pins the training launcher uses: IGEV++ needs an old timm, DEFOM-Stereo a
# current one. FoundationStereo never installed anything, so neither do we.
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

# Unlike training, the run name has no timestamp: an evaluation of a fixed
# checkpoint on a fixed split is reproducible, and a stable directory is what
# makes evaluation.skip_existing able to resume. Pass --name to keep two runs
# apart, or evaluation.output_dir to put one somewhere specific.
RUN_NAME="${RUN_NAME_PREFIX}"

RUN_ARGS=(
  "config_path=$CONFIG"
  "name=$RUN_NAME"
)

if [[ "$NGPUS" -gt 1 ]]; then
  CMD=( python3 -m torch.distributed.run --standalone --master_port="$PORT"
        --nproc_per_node="$NGPUS" evaluate.py "${RUN_ARGS[@]}" )
else
  CMD=( python3 evaluate.py "${RUN_ARGS[@]}" )
fi
CMD+=( "${FORWARD[@]+"${FORWARD[@]}"}" )

echo "evaluation: $CONFIG"
echo "config:     $CONFIG_SOURCE"
echo "run name:   $RUN_NAME"
echo "GPUs:       $GPUS ($NGPUS)"
echo "command:    ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

# Refuse to evaluate against unpatched submodules. A moved pin would otherwise
# produce numbers that look fine and are not comparable to the reported ones.
if ! "$REPO_DIR/scripts/setup_submodules.sh" --check >/dev/null 2>&1; then
  echo "error: submodules are not in the expected state." >&2
  "$REPO_DIR/scripts/setup_submodules.sh" --check >&2 || true
  echo "Run ./scripts/setup_submodules.sh (or --reset) first." >&2
  exit 1
fi

# evaluate.py resolves output paths relative to the CWD (./eval/), so always
# launch from the repo root.
cd "$REPO_DIR"
exec "${CMD[@]}"
