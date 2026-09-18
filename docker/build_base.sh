#!/usr/bin/env bash
#
# Build the shared base image: CUDA, torch, and every python dependency.
#
# This is the slow one (it compiles flash-attn from source). It has nothing
# user-specific in it, so it is built once per machine and then reused by
# build_user.sh for every user on that machine.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Must match BASE_IMAGE in container_user.Dockerfile.
IMAGE_TAG="sethgi/surfslam:stereo"

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-t $IMAGE_TAG "
DOCKER_OPTIONS+="-f $SCRIPT_DIR/container_base.Dockerfile "

DOCKER_CMD="docker build $DOCKER_OPTIONS $SCRIPT_DIR"
echo $DOCKER_CMD
exec $DOCKER_CMD
