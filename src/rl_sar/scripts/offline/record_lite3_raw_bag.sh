#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BAG_BASE_DIR:-$HOME/bags/lite3/raw}"
OUT_DIR="${BASE_DIR}/${RUN_NAME}"

mkdir -p "${BASE_DIR}"

echo "[offline-record] output: ${OUT_DIR}"
echo "[offline-record] press Ctrl+C to stop"

# Minimum topics for offline localization + trajectory replay.
# Goal topics are optional: if available they are recorded too.
TOPICS=(
  /rslidar_points
  /imu/data
  /tf
  /tf_static
  /Odometry
  /camera/depth/processed
  /camera/depth/processed_norm
  /nav/goal_actual_map
  /nav/goal_pred_map
  /nav/goal_error_body
  /nav_goal_body
)

exec ros2 bag record -o "${OUT_DIR}" "${TOPICS[@]}"
