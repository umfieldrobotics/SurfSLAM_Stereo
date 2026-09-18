#!/usr/bin/env bash
# Thin wrapper: igev_pp x full_pretrain. See train_scripts/launch.sh for options.
exec "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )/launch.sh" \
  igev_pp/full_pretrain "$@"
