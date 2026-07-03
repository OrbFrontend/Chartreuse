#!/usr/bin/env bash
# Build the GPU image and run the purple-prose web app. Requires Docker.
#   scripts/host_docker.sh                          # build + run on :8000 (purple)
#   DEPURPLE_AXIS=euphemism scripts/host_docker.sh  # serve the euphemism classifier
#   PORT=9000 scripts/host_docker.sh                # different port
#   IMAGE=foo scripts/host_docker.sh                # different tag
set -euo pipefail
cd "$(dirname "$0")/.."                       # run from repo root (scripts/ lives one level down)

IMAGE="${IMAGE:-purple-prose}"
PORT="${PORT:-8000}"
AXIS="${DEPURPLE_AXIS:-purple}"
# Dir slug tracks the trained encoder; override MODEL_DIR if ETTIN_MODEL wasn't the default.
MODEL_DIR="${MODEL_DIR:-models/ettin400m-${AXIS}}"

[ -d "$MODEL_DIR" ] || { echo "no $MODEL_DIR -- train the $AXIS axis first"; exit 1; }

docker build -t "$IMAGE" .

# Use the GPU if Docker can actually expose it; otherwise run on CPU (app.py
# falls back automatically). For the GPU path install nvidia-container-toolkit:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
GPU=""
if docker run --rm --gpus all "$IMAGE" true 2>/dev/null; then
  GPU="--gpus all"
  echo "GPU detected -> running with --gpus all"
else
  echo "no Docker GPU runtime (nvidia-container-toolkit) -> running on CPU"
fi

# Mount the host models dir so DEPURPLE_AXIS picks the served checkpoint (+ its
# threshold.json) without an image rebuild; the baked-in purple model is the
# fallback when run without this script.
echo "axis=$AXIS  model=$MODEL_DIR  -> http://localhost:${PORT}"
exec docker run --rm $GPU -p "${PORT}:8000" \
  -v "$PWD/models:/app/models:ro" -e MODEL_DIR="$MODEL_DIR" "$IMAGE"
