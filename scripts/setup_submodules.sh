#!/usr/bin/env bash
# Check out the upstream submodules and apply this project's patches to them.
#
#   ./scripts/setup_submodules.sh            # idempotent: safe to re-run
#   ./scripts/setup_submodules.sh --reset    # discard submodule changes, re-apply
#   ./scripts/setup_submodules.sh --check    # report state, change nothing
#
# The submodules point at their real upstreams (see .gitmodules), pinned to exact
# commits. Everything we changed lives in patches/, one file per submodule, and each
# patch carries the base commit it was generated against in a `# base:` header. This
# script refuses to apply a patch to anything other than that commit, because a
# half-applied patch is much worse than a loud failure.
#
# To regenerate a patch after deliberately bumping a submodule, see docs/submodules.md.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
PATCH_DIR="$REPO_DIR/patches"

# submodule path : patch file
MANIFEST=(
  "models/FoundationStereo:foundation-stereo.patch"
  "models/DEFOM-Stereo:defom-stereo.patch"
  "models/IGEV-plusplus:igev-plusplus.patch"
  "models/Underwater_Stereo:underwater-stereo.patch"
  "utils/dtd:dtd.patch"
)

MODE=apply
case "${1:-}" in
  --reset) MODE=reset ;;
  --check) MODE=check ;;
  "")      ;;
  -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
esac

red ()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn ()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw ()  { printf '\033[33m%s\033[0m\n' "$*"; }

if [[ $MODE != check ]]; then
  echo "== syncing submodule URLs and checking out pinned commits"
  # Deliberately not --recursive. utils/dtd declares three nested submodules of its
  # own (unimatch, and two RobotCar dataset tools) that nothing here imports; we only
  # use dtd.losses.photometric_loss and dtd.models.image_warping. Recursing would
  # clone ~34MB of unrelated code and make setup fail if any of those hosts is down.
  git -C "$REPO_DIR" submodule sync >/dev/null
  if [[ $MODE == reset ]]; then
    git -C "$REPO_DIR" submodule update --init --force
  else
    git -C "$REPO_DIR" submodule update --init
  fi
fi

failed=0
for entry in "${MANIFEST[@]}"; do
  sub="${entry%%:*}"
  patch="$PATCH_DIR/${entry##*:}"
  dir="$REPO_DIR/$sub"
  label=$(printf '%-24s' "$sub")

  if [[ ! -d $dir/.git && ! -f $dir/.git ]]; then
    red "$label not initialized -- run: git submodule update --init --recursive"
    failed=1; continue
  fi
  if [[ ! -f $patch ]]; then
    red "$label missing patch file: $patch"
    failed=1; continue
  fi

  want=$(sed -n 's/^# base: *\([0-9a-f]\{40\}\).*/\1/p' "$patch" | head -1)
  if [[ -z $want ]]; then
    red "$label $patch has no '# base: <40-hex-sha>' header"
    failed=1; continue
  fi
  have=$(git -C "$dir" rev-parse HEAD)

  if [[ $have != "$want" ]]; then
    red "$label checked out $have"
    red "                         but $(basename "$patch") expects $want"
    red "                         -> the pin moved; regenerate the patch (docs/submodules.md)"
    failed=1; continue
  fi

  # Keep bytecode out of the way. Local to the clone, so it stays out of the patch.
  excl="$(git -C "$dir" rev-parse --git-path info/exclude)"
  mkdir -p "$(dirname "$excl")"
  grep -qxF '__pycache__/' "$excl" 2>/dev/null || printf '__pycache__/\n*.pyc\n' >> "$excl"

  # A patch with no hunks is just a pin declaration, nothing to apply.
  if ! grep -q '^diff --git' "$patch"; then
    grn "$label at $want (upstream unmodified, by design)"
    continue
  fi

  if [[ $MODE == reset ]]; then
    # `submodule update --force` only restores *tracked* files, so a rename patch
    # leaves its targets behind as untracked ones -- e.g. core/ comes back while
    # defom_core/ is still there. In that state the patch neither applies nor
    # reverse-applies. Delete just the paths this patch creates; never the ones it
    # only modifies, which --force has already restored.
    while IFS= read -r created; do
      [[ -n $created && -e $dir/$created ]] && rm -rf -- "$dir/$created"
    done < <(awk '
      /^diff --git/      { isnew = 0 }
      /^new file mode/   { isnew = 1 }
      /^rename to /      { sub(/^rename to /, ""); print; next }
      /^\+\+\+ b\//      { if (isnew) { sub(/^\+\+\+ b\//, ""); print } }
    ' "$patch" | sort -u)
  fi

  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    grn "$label at $want, patch already applied"
    continue
  fi

  if [[ $MODE == check ]]; then
    if git -C "$dir" apply --check "$patch" 2>/dev/null; then
      ylw "$label at $want, patch NOT applied"
    else
      red "$label at $want, patch neither applied nor applicable (tree modified?)"
      failed=1
    fi
    continue
  fi

  if ! git -C "$dir" apply --check "$patch" 2>/dev/null; then
    red "$label patch will not apply and is not already applied."
    red "                         The working tree has diverged. Re-run with --reset"
    red "                         to discard submodule changes and start clean."
    failed=1; continue
  fi

  git -C "$dir" apply "$patch"
  grn "$label at $want, patch applied"
done

if (( failed )); then
  echo
  red "submodule setup incomplete -- see above"
  exit 1
fi

if [[ $MODE == check ]]; then
  echo; grn "all submodules report the expected state"
else
  echo; grn "submodules ready"
fi
