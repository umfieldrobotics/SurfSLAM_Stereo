#!/usr/bin/env bash
# Regenerate the two mechanical rename patches from their pinned upstream trees.
#
#   ./scripts/regenerate_rename_patches.sh
#
# DEFOM-Stereo and IGEV-plusplus both ship a top-level package called `core`, and this
# project puts them and FoundationStereo on one sys.path. FoundationStereo keeps the
# name (so it needs no patch); these two rename theirs. That rename is purely
# mechanical, so it is generated rather than maintained by hand.
#
# Run this after deliberately moving either pin: update BASE below to the new commit,
# run this, then ./scripts/setup_submodules.sh --reset.
#
# The hand-authored patches (underwater-stereo, dtd) are NOT regenerable this way --
# see docs/submodules.md for how to rebase those.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
PATCH_DIR="$REPO_DIR/patches"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Keep the generated tree reproducible regardless of who runs it.
export GIT_AUTHOR_NAME=patchgen GIT_AUTHOR_EMAIL=patchgen@localhost
export GIT_COMMITTER_NAME=patchgen GIT_COMMITTER_EMAIL=patchgen@localhost

# Read the pinned base out of the existing patch header, so there is exactly one
# source of truth for it.
base_of () { sed -n 's/^# base: *\([0-9a-f]\{40\}\).*/\1/p' "$1" | head -1; }

# Everything above the first 'diff --git' is the rationale header; preserve it verbatim.
header_of () { sed '/^diff --git/,$d' "$1"; }

gen () {                       # sub  patchfile  rename-spec...
  local sub=$1 patch=$2; shift 2
  local base; base=$(base_of "$PATCH_DIR/$patch")
  if [[ -z $base ]]; then echo "no '# base:' in $patch" >&2; exit 1; fi

  local d="$WORK/$(basename "$sub")"
  mkdir -p "$d"
  git -C "$REPO_DIR/$sub" archive "$base" | tar -x -C "$d"
  git -C "$d" init -q .
  git -C "$d" add -A
  git -C "$d" commit -qm base

  # Longest source name first, so `core_rt` is rewritten before `core`.
  local spec from to
  for spec in "$@"; do
    from=${spec%%:*}; to=${spec##*:}
    git -C "$d" mv "$from" "$to"
  done
  for spec in "$@"; do
    from=${spec%%:*}; to=${spec##*:}
    grep -rlE "(from $from[. ]|import $from\.|append\('$from'\))" --include='*.py' "$d" 2>/dev/null \
      | xargs -r sed -i -E \
        "s/from $from\./from $to./g; s/import $from\./import $to./g; s/sys\.path\.append\('$from'\)/sys.path.append('$to')/g"
  done
  git -C "$d" add -A

  # Nothing should still reference the old package names.
  local leftover
  leftover=$(grep -rnE "(from ($(IFS='|'; set -- "$@"; printf '%s' "${*%%:*}"))\.|import core[._ ])" --include='*.py' "$d" || true)
  if [[ -n $leftover ]]; then echo "$sub: unrewritten references remain:" >&2; echo "$leftover" >&2; exit 1; fi

  { header_of "$PATCH_DIR/$patch"; git -C "$d" diff --cached -M; } > "$PATCH_DIR/$patch.new"
  # Added lines must not carry trailing whitespace, or git apply warns on every run.
  sed -i -E 's/^\+[[:space:]]+$/+/' "$PATCH_DIR/$patch.new"
  mv "$PATCH_DIR/$patch.new" "$PATCH_DIR/$patch"
  printf '  %-22s regenerated against %s (%s lines)\n' "$patch" "${base:0:7}" "$(wc -l < "$PATCH_DIR/$patch")"
}

echo "regenerating rename patches:"
gen models/DEFOM-Stereo  defom-stereo.patch   core:defom_core
gen models/IGEV-plusplus igev-plusplus.patch  core_rt:igev_core_rt core:igev_core
echo
echo "now run: ./scripts/setup_submodules.sh --reset"
