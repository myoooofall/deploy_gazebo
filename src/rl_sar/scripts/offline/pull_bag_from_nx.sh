#!/usr/bin/env bash
set -euo pipefail

# Run this script on NX to push bag directories to your Mac.
# -------- User-configurable defaults (can be overridden by env) --------
MAC_USER="${MAC_USER:-YOUR_MAC_USER}"
MAC_HOST="${MAC_HOST:-192.168.1.100}"
SSH_PORT="${SSH_PORT:-22}"
NX_BASE_DIR="${NX_BASE_DIR:-$HOME/bags/lite3/raw}"
MAC_BASE_DIR="${MAC_BASE_DIR:-~/bags/lite3/raw}"
# ----------------------------------------------------------------------

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") <run_name_or_abs_nx_bag_path>

Examples (run on NX):
  $(basename "$0") run_maze_01
  $(basename "$0") /home/ysc/bags/lite3/raw/run_maze_01

Current config:
  MAC_USER=${MAC_USER}
  MAC_HOST=${MAC_HOST}
  SSH_PORT=${SSH_PORT}
  NX_BASE_DIR=${NX_BASE_DIR}
  MAC_BASE_DIR=${MAC_BASE_DIR}

Tip (one-off override):
  MAC_USER=alice MAC_HOST=192.168.1.20 MAC_BASE_DIR=~/bags/lite3/raw ./$(basename "$0") run_maze_01
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUN_ARG="$1"
if [[ "$RUN_ARG" = /* ]]; then
  NX_BAG_PATH="$RUN_ARG"
  RUN_NAME="$(basename "$RUN_ARG")"
else
  NX_BAG_PATH="${NX_BASE_DIR%/}/${RUN_ARG}"
  RUN_NAME="$RUN_ARG"
fi

if [[ ! -d "$NX_BAG_PATH" ]]; then
  echo "[push-bag] nx bag path not found: $NX_BAG_PATH" >&2
  exit 2
fi

REMOTE_DEST="${MAC_USER}@${MAC_HOST}:${MAC_BASE_DIR%/}/"

echo "[push-bag] source (NX): ${NX_BAG_PATH}"
echo "[push-bag] dest   (Mac): ${REMOTE_DEST}${RUN_NAME}"

# Push the directory with progress and resume-friendly behavior.
rsync -avP -e "ssh -p ${SSH_PORT}" "${NX_BAG_PATH}" "${REMOTE_DEST}"

echo "[push-bag] done"
echo "[push-bag] verify on Mac: ros2 bag info \"${MAC_BASE_DIR%/}/${RUN_NAME}\""
