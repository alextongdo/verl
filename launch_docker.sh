#!/bin/bash

set -e  # exit on error

# Get the absolute path of the project directory
PROJECT_DIR=$(realpath "$(pwd)")
DOCKER_IMAGE="verlai/verl:vllm012.latest"

# -e HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "") \
docker run \
    --gpus '"device=0,1,2,3"' \
    --name alex_verl \
    --rm \
    --shm-size=10g \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    -v "${PROJECT_DIR}/:/workspace" \
    -e LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat \
    -w /workspace \
    -it "${DOCKER_IMAGE}" /bin/bash /workspace/extra_setup_in_docker.sh