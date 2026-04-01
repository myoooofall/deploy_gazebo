#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

set +u
if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -f "${WORKSPACE_DIR}/lite_cog_ros2/nav/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE_DIR}/lite_cog_ros2/nav/install/setup.bash"
fi
if [[ -f "${WORKSPACE_DIR}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE_DIR}/install/setup.bash"
fi
set -u

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <bag_dir> [map_yaml] [map_pcd]"
  echo "Example:"
  echo "  $0 /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/3_25/run_20260331_122023 \\"
  echo "     /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/3_25/map/lite3.yaml \\"
  echo "     /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/3_25/map/lite3.pcd"
  exit 1
fi

BAG_DIR="$1"
MAP_YAML="${2:-/home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/3_25/map/lite3.yaml}"
MAP_PCD="${3:-/home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/3_25/map/lite3.pcd}"
DEFAULT_RVIZ_CONFIG="/home/teleai/Desktop/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/rviz/goal_only.rviz"
RVIZ_CONFIG="${4:-$DEFAULT_RVIZ_CONFIG}"
if [[ ! -f "$RVIZ_CONFIG" ]]; then
  RVIZ_CONFIG="/home/teleai/Desktop/liangwang_ws/rl_sar_new/lite_cog_ros2/nav/src/dr_nav2/rviz/dr_nav2.rviz"
fi

ENABLE_MAP_SERVER="false"
if ros2 pkg list 2>/dev/null | grep -qx "nav2_map_server"; then
  ENABLE_MAP_SERVER="true"
else
  echo "[offline-replay] nav2_map_server not found; disabling map_server (RViz will still launch)."
fi

# Isolate playback from other running ROS graph to avoid /clock and /tf conflicts.
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  export ROS_DOMAIN_ID=88
  echo "[offline-replay] ROS_DOMAIN_ID not set; using isolated domain: $ROS_DOMAIN_ID"
fi

LOCALIZATION_CMD="ros2 launch hdl_localization lite_localization.launch.py \
  enable_nav2:=false \
  enable_rviz:=true \
  enable_map_server:=${ENABLE_MAP_SERVER} \
  rviz_config:=${RVIZ_CONFIG} \
  map_server_config_file:=${MAP_YAML} \
  use_sim_time:=true \
  map_path:=${MAP_PCD} \
  publish_base2lidar_tf:=false"

export LOCALIZATION_CMD
# Force non-paused playback in this wrapper to avoid stale env vars
# (e.g. PLAY_START_PAUSED=1) keeping rosbag in paused state.
export PLAY_START_PAUSED=0
export PLAY_BEFORE_NODES=0
echo "[offline-replay] playback mode: PLAY_START_PAUSED=${PLAY_START_PAUSED}, PLAY_BEFORE_NODES=${PLAY_BEFORE_NODES}, PLAY_RATE=${PLAY_RATE:-1.0}, PLAY_LOOP=${PLAY_LOOP:-0}"

"${SCRIPT_DIR}/replay_localize_and_record.sh" "${BAG_DIR}"
