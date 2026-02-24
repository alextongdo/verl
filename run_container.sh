#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${VERL_IMAGE:-verlai/verl:vllm015.dev}"
container_name="${VERL_CONTAINER_NAME:-verl}"
container_workspace="/workspace/verl"

if [[ -n "$(docker ps -aq -f "name=^/${container_name}$")" ]]; then
  docker rm -f "${container_name}" >/dev/null
fi

docker create \
  --runtime=nvidia \
  --gpus all \
  --net=host \
  --shm-size="10g" \
  --cap-add=SYS_ADMIN \
  --mount "type=bind,src=${repo_dir},dst=${container_workspace}" \
  -w "${container_workspace}" \
  --name "${container_name}" \
  "${image}" \
  sleep infinity >/dev/null

docker start "${container_name}" >/dev/null
docker exec -it -w "${container_workspace}" "${container_name}" bash
