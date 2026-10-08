#!/usr/bin/env sh
# Start ComfyUI for localgen. Run from your ComfyUI folder's parent, or set COMFY_DIR.
set -e
COMFY_DIR="${COMFY_DIR:-$PWD/ComfyUI}"
cd "$COMFY_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
[ -x ../venv/bin/python ] && PY=../venv/bin/python
exec "$PY" main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --lowvram "$@"
