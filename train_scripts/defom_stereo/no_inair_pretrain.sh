#!/usr/bin/env bash
# Thin wrapper: defom_stereo x no_inair_pretrain. See train_scripts/launch.sh for options.
exec "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )/launch.sh" \
  defom_stereo/no_inair_pretrain "$@"
