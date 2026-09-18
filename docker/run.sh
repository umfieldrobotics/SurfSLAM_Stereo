#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( readlink -f "$SCRIPT_DIR/.." )
IMAGE_TAG=surfslam:stereo_$(whoami)
CONTAINER_NAME=surfslam_stereo_$(whoami)

# Where config/dataset_paths.yaml entries get mounted inside the container.
# utils/dataset_paths.py re-points registry entries here when it detects it is running
# in Docker.
CONTAINER_DATA_ROOT=/data
DATASET_PATHS_FILE="${SURF_DATASET_PATHS:-$REPO_DIR/config/dataset_paths.yaml}"

DOCKER_OPTIONS=(
  --gpus all
  --env NVIDIA_DISABLE_REQUIRE=1
  -it
  -e "DISPLAY=$DISPLAY"
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -v "$REPO_DIR:/home/$(whoami)/SurfSLAM_stereo"
  -v "$HOME/.Xauthority:/home/$(whoami)/.Xauthority"
  -v /etc/group:/etc/group:ro
  --name "$CONTAINER_NAME"
  --privileged
  -e NVIDIA_DRIVER_CAPABILITIES=all
  --cap-add=SYS_PTRACE
  --ipc=host
  --network=host
  --pid=host
  --security-opt seccomp=unconfined
  --runtime=nvidia
  -e SDL_VIDEODRIVER=x11
  -u "$(id -u):$(id -g)"
  --shm-size 32G
  # Tell the code inside the container to resolve dataset roots under $CONTAINER_DATA_ROOT.
  -e SURF_IN_DOCKER=1
  -e "SURF_DOCKER_DATA_ROOT=$CONTAINER_DATA_ROOT"
  -w "/home/$(whoami)/SurfSLAM_stereo"
)

RUNTIME_OPTIONS=(
  -w "/home/$(whoami)/SurfSLAM_stereo"
  -it "$CONTAINER_NAME"
)

for gid in $(id -G); do
  DOCKER_OPTIONS+=(--group-add "${gid}")
done

if [ -n "${DATA_DIR:-}" ]; then
  DOCKER_OPTIONS+=(-v "$DATA_DIR:/home/$(whoami)/data")
fi

for cam in /dev/video*; do
  [ -e "$cam" ] && DOCKER_OPTIONS+=(--device="${cam}")
done

# --- dataset mounts, from config/dataset_paths.yaml -------------------------
# Each `key: /host/path` entry is bind-mounted at $CONTAINER_DATA_ROOT/<key>, so
# `lizard_island: /path/to/LizardIslandColmap` becomes /data/lizard_island.
# Keys left null (or pointing at something that doesn't exist) are skipped.
if [ -f "$DATASET_PATHS_FILE" ]; then
  echo "Dataset mounts (from $DATASET_PATHS_FILE):"
  while read -r key value; do
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"

    # YAML nulls, before ~ means $HOME
    case "$value" in
      null|Null|NULL|"~"|"") continue ;;
    esac

    value="${value/#\~/$HOME}"

    if [ ! -d "$value" ]; then
      echo "  $key: skipped, $value is not a directory"
      continue
    fi

    DOCKER_OPTIONS+=(-v "$value:$CONTAINER_DATA_ROOT/$key")
    echo "  $key: $value -> $CONTAINER_DATA_ROOT/$key"
  done < <(
    sed -e 's/#.*$//' -e 's/[[:space:]]*$//' "$DATASET_PATHS_FILE" \
      | grep -E '^[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*[^[:space:]]' \
      | sed -E 's/^([A-Za-z_][A-Za-z0-9_]*):[[:space:]]*/\1 /'
  )
else
  echo "No $DATASET_PATHS_FILE - no dataset mounts. See docs/data.md."
fi

echo "$CONTAINER_NAME"

if [ "${1:-""}" == "restart" ]; then
  echo "Restarting Container"
  docker rm -f "$CONTAINER_NAME"
  docker run "${DOCKER_OPTIONS[@]}" "$IMAGE_TAG" /bin/bash
# https://stackoverflow.com/questions/38576337/how-to-execute-a-bash-command-only-if-a-docker-container-with-a-given-name-does
elif [ ! "$(docker ps -q -f name="$CONTAINER_NAME")" ]; then # If container isn't running

    # If it exists, but needs to be started
    if [ "$(docker ps -aq -f name="$CONTAINER_NAME")" ]; then

          echo "Resuming Container"
          docker start "$CONTAINER_NAME"
          docker exec "${RUNTIME_OPTIONS[@]}" /entrypoint.sh
    else
      echo "Running Container"
      docker run "${DOCKER_OPTIONS[@]}" "$IMAGE_TAG"
    fi
else
  echo "Attaching to existing container"
  docker exec "${RUNTIME_OPTIONS[@]}" /entrypoint.sh
fi
