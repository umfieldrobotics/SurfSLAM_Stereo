#!/usr/bin/env bash
# Thin wrapper: defom_stereo x haze_only_pretrain. See train_scripts/launch.sh for options.
exec "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )/launch.sh" \
  defom_stereo/haze_only_pretrain "$@"
