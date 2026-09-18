#!/usr/bin/env bash
# Thin wrapper: defom_stereo x warp_finetune_no_inair. See train_scripts/launch.sh for options.
exec "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )/launch.sh" \
  defom_stereo/warp_finetune_no_inair "$@"
