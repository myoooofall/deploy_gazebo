#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--lite|--full] [run_name]

Examples:
  $(basename "$0")
  $(basename "$0") run_maze_01
  $(basename "$0") --lite run_maze_01
  $(basename "$0") --full run_maze_01

Notes:
- Default mode is --full.
- Path behavior is unchanged:
  OUT_DIR = \
    \${BAG_BASE_DIR:-\$HOME/bags/lite3/raw}/<run_name>
USAGE
}

MODE="full"
if [[ "${1:-}" == "--lite" ]]; then
  MODE="lite"
  shift
elif [[ "${1:-}" == "--full" ]]; then
  MODE="full"
  shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BAG_BASE_DIR:-$HOME/bags/lite3/raw}"
OUT_DIR="${BASE_DIR}/${RUN_NAME}"

mkdir -p "${BASE_DIR}"

echo "[offline-record] mode: ${MODE}"
echo "[offline-record] output: ${OUT_DIR}"
echo "[offline-record] press Ctrl+C to stop"

TOPICS_COMMON=(
  /rslidar_points
  /imu/data
  /rosout
  /tf
  /tf_static
  /Odometry
  /nav/goal_actual_map
  /nav/goal_pred_map
  /nav/goal_error_body
  /nav_goal_body
)

TOPICS_DEPTH=(
  /camera/depth/processed
  /camera/depth/processed_norm
)

if [[ "${MODE}" == "lite" ]]; then
  TOPICS=("${TOPICS_COMMON[@]}")
else
  TOPICS=("${TOPICS_COMMON[@]}" "${TOPICS_DEPTH[@]}")
fi

echo "[offline-record] topics(${#TOPICS[@]}): ${TOPICS[*]}"
exec ros2 bag record -o "${OUT_DIR}" "${TOPICS[@]}"
