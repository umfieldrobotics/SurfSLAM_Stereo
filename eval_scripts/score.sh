#!/usr/bin/env bash
# Score precomputed predictions against ground truth -- no model, no GPU.
#
#   ./eval_scripts/score.sh <config> [options] [key=value ...]
#
#   <config>    a file from config/eval/scoring/, e.g. 'tbnms' or 'legacy_cvpr'
#
# Options:
#   --list      print the available scoring configs
#   --dry-run   print the command instead of running it
#
# Anything else is forwarded to score_predictions.py, so OmegaConf overrides work:
#   ./eval_scripts/score.sh tbnms scoring.predictions_dir=eval/defom_stereo_suds_test/ours
#   ./eval_scripts/score.sh tbnms scoring.run=eval/defom_stereo_suds_test scoring.variants=[masked]
#   ./eval_scripts/score.sh legacy_cvpr scoring.legacy_gt_dir=... scoring.legacy_mask_dir=... \
#       scoring.predictions_dir=...
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
CONFIG_DIR="$REPO_DIR/config/eval/scoring"

usage () { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

list_configs () {
  echo "scoring configs: $( cd "$CONFIG_DIR" && ls *.yaml 2>/dev/null | sed 's/\.yaml$//' | sort | tr '\n' ' ' )"
}

CONFIG=""
DRY_RUN=0
FORWARD=()

if [[ $# -eq 0 ]]; then usage; echo; list_configs; exit 1; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  usage; echo; list_configs; exit 0 ;;
    --list)     list_configs; exit 0 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -*)         echo "error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
    *=*)        FORWARD+=("$1"); shift ;;
    *)          CONFIG="${1%.yaml}"; shift ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "error: name a scoring config, e.g. 'tbnms'" >&2
  list_configs >&2
  exit 1
fi
if [[ ! -f "$CONFIG_DIR/$CONFIG.yaml" ]]; then
  echo "error: no scoring config '$CONFIG'" >&2
  list_configs >&2
  exit 1
fi

CMD=( python3 score_predictions.py "config_path=scoring/$CONFIG" )
CMD+=( "${FORWARD[@]+"${FORWARD[@]}"}" )

echo "scoring:  $CONFIG"
echo "command:  ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

# score_predictions.py resolves relative prediction paths against the CWD, so
# launch from the repo root like the other entry points.
cd "$REPO_DIR"
exec "${CMD[@]}"
