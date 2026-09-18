#!/usr/bin/env bash
# Thin wrapper: foundation_stereo x warp_finetune_full. See train_scripts/launch.sh for options.
exec "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )/launch.sh" \
  foundation_stereo/warp_finetune_full "$@"
