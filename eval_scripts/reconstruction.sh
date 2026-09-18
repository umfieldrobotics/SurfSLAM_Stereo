#!/usr/bin/env bash
# Evaluate 3D reconstructions against ground-truth point clouds.
#
#   ./eval_scripts/reconstruction.sh <config> [options] [key=value ...]
#
#   <config>    a file from config/eval/reconstruction/, e.g. 'mesh' or 'frames'
#
# Options:
#   --list      print the available reconstruction configs
#   --dry-run   print the command instead of running it
#
# Anything else is forwarded to evaluate_reconstruction.py as OmegaConf overrides:
#   ./eval_scripts/reconstruction.sh mesh reconstruction.mesh=recon.obj \
#       reconstruction.gt_cloud=fused.ply reconstruction.output_dir=results/
#   ./eval_scripts/reconstruction.sh frames reconstruction.results_root=... \
#       reconstruction.gt_root=... reconstruction.intrinsics=intrinsics.txt
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
CONFIG_DIR="$REPO_DIR/config/eval/reconstruction"

usage () { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

list_configs () {
  echo "reconstruction configs: $( cd "$CONFIG_DIR" && ls *.yaml 2>/dev/null | sed 's/\.yaml$//' | sort | tr '\n' ' ' )"
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
  echo "error: name a reconstruction config, e.g. 'mesh'" >&2
  list_configs >&2
  exit 1
fi
if [[ ! -f "$CONFIG_DIR/$CONFIG.yaml" ]]; then
  echo "error: no reconstruction config '$CONFIG'" >&2
  list_configs >&2
  exit 1
fi

CMD=( python3 evaluate_reconstruction.py "config_path=reconstruction/$CONFIG" )
CMD+=( "${FORWARD[@]+"${FORWARD[@]}"}" )

echo "reconstruction: $CONFIG"
echo "command:        ${CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

cd "$REPO_DIR"
exec "${CMD[@]}"
