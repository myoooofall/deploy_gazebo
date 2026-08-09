#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${RL_SAR_DIR}/.." && pwd)"
DEFAULT_BAG_BASE_DIR="${WS_DIR}/bag"

usage() {
  cat <<EOF
Usage: $(basename "$0") <raw_bag_path_or_name> [localized_output_bag] [--rate RATE]

Run hdl_localization on a raw/split bag and record a new bag containing odom.

Defaults:
  bag base       ${DEFAULT_BAG_BASE_DIR}
  map pcd        ${WS_DIR}/map/map/lite3.pcd
  map yaml       ${WS_DIR}/map/map/lite3.yaml
  replay rate    0.5
  rviz           true

Environment overrides:
  PLAY_RATE=0.5
  ENABLE_RVIZ=true
  ENABLE_OCCUPANCY_MAP=true
  ENABLE_ODOM_PATH=true
  INIT_POSE_WAIT_SEC=8
  MAP_PCD=/abs/path/lite3.pcd
  MAP_YAML=/abs/path/lite3.yaml
  LOCALIZED_BAG_BASE_DIR=/abs/path/output_dir

Notes:
- If RViz opens, set 2D Pose Estimate during INIT_POSE_WAIT_SEC if the initial
  pose differs from the map.
- Stop live lidar/IMU publishers before replaying offline bags.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

INPUT="$1"
shift

OUT_BAG_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rate)
      PLAY_RATE="${2:-}"
      if [[ -z "${PLAY_RATE}" ]]; then
        echo "[localize-record] ERROR: --rate requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[localize-record] ERROR: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${OUT_BAG_ARG}" ]]; then
        echo "[localize-record] ERROR: unexpected extra argument '$1'" >&2
        usage >&2
        exit 2
      fi
      OUT_BAG_ARG="$1"
      shift
      ;;
  esac
done

if [[ -d "${INPUT}" ]]; then
  RAW_BAG="$(cd "${INPUT}" && pwd)"
else
  RAW_BAG="${DEFAULT_BAG_BASE_DIR}/${INPUT}"
fi

if [[ ! -d "${RAW_BAG}" ]]; then
  echo "[localize-record] ERROR: raw bag not found: ${RAW_BAG}" >&2
  exit 2
fi

RUN_NAME="$(basename "${RAW_BAG}")"
LOCALIZED_BASE="${LOCALIZED_BAG_BASE_DIR:-${DEFAULT_BAG_BASE_DIR}}"
OUT_BAG="${OUT_BAG_ARG:-${LOCALIZED_BASE}/${RUN_NAME}_localized}"

if [[ -e "${OUT_BAG}" ]]; then
  echo "[localize-record] ERROR: output already exists: ${OUT_BAG}" >&2
  exit 3
fi

PLAY_RATE="${PLAY_RATE:-0.5}"
ENABLE_RVIZ="${ENABLE_RVIZ:-true}"
ENABLE_OCCUPANCY_MAP="${ENABLE_OCCUPANCY_MAP:-true}"
ENABLE_ODOM_PATH="${ENABLE_ODOM_PATH:-true}"
INIT_POSE_WAIT_SEC="${INIT_POSE_WAIT_SEC:-8}"
MAP_PCD="${MAP_PCD:-${WS_DIR}/map/map/lite3.pcd}"
MAP_YAML="${MAP_YAML:-${WS_DIR}/map/map/lite3.yaml}"
MAP_SERVER="${MAP_SERVER:-false}"
USE_SIM_TIME="${USE_SIM_TIME:-false}"
PUBLISH_BASE2LIDAR_TF="${PUBLISH_BASE2LIDAR_TF:-true}"
START_LOCALIZATION_ONLY="${START_LOCALIZATION_ONLY:-${WS_DIR}/lite_cog_ros2/system/scripts/slam/start_localization_only.sh}"

set +u
if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f "/opt/ros/foxy/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
fi
if [[ -f "${WS_DIR}/lite_cog_ros2/nav/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/lite_cog_ros2/nav/install/setup.bash"
fi
if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${WS_DIR}/install/setup.bash"
fi
set -u

PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  echo
  echo "[localize-record] stopping background processes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -INT "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait "${PIDS[@]:-}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[localize-record] raw_bag=${RAW_BAG}"
echo "[localize-record] output=${OUT_BAG}"
echo "[localize-record] map_pcd=${MAP_PCD}"
echo "[localize-record] map_yaml=${MAP_YAML}"
echo "[localize-record] rviz=${ENABLE_RVIZ} occupancy_map=${ENABLE_OCCUPANCY_MAP} odom_path=${ENABLE_ODOM_PATH} init_pose_wait=${INIT_POSE_WAIT_SEC}s replay_rate=${PLAY_RATE}"

if [[ ! -x "${START_LOCALIZATION_ONLY}" ]]; then
  echo "[localize-record] ERROR: start_localization_only.sh not found/executable: ${START_LOCALIZATION_ONLY}" >&2
  exit 4
fi

ENABLE_RVIZ="${ENABLE_RVIZ}" \
ENABLE_MAP_SERVER="${MAP_SERVER}" \
ENABLE_OCCUPANCY_MAP="${ENABLE_OCCUPANCY_MAP}" \
ENABLE_ODOM_PATH="${ENABLE_ODOM_PATH}" \
USE_SIM_TIME="${USE_SIM_TIME}" \
PUBLISH_BASE2LIDAR_TF="${PUBLISH_BASE2LIDAR_TF}" \
MAP_PCD="${MAP_PCD}" \
MAP_YAML="${MAP_YAML}" \
"${START_LOCALIZATION_ONLY}" false &
PIDS+=("$!")

sleep 3
if ! kill -0 "${PIDS[0]}" >/dev/null 2>&1; then
  echo "[localize-record] ERROR: localization exited early." >&2
  exit 4
fi

echo "[localize-record] starting recorder..."
ros2 bag record -o "${OUT_BAG}" \
  /odom /Odometry /odom_viz /odom_path \
  /map \
  /aligned_points \
  /rslidar_points /imu/data \
  /nav/goal_pred_map /nav/goal_actual_map /nav/goal_error_body \
  /nav/cmd_high /nav/cmd_applied \
  /tf /tf_static &
REC_PID="$!"
PIDS+=("${REC_PID}")

if [[ "${INIT_POSE_WAIT_SEC}" != "0" ]]; then
  echo "[localize-record] waiting ${INIT_POSE_WAIT_SEC}s before replay."
  echo "[localize-record] If needed, set 2D Pose Estimate in RViz now."
  sleep "${INIT_POSE_WAIT_SEC}"
fi

echo "[localize-record] replaying bag..."
if [[ -d "${RAW_BAG}/lidar" && -d "${RAW_BAG}/small" ]]; then
  "${SCRIPT_DIR}/replay_nav_split_bag.sh" "${RAW_BAG}" --rate "${PLAY_RATE}" &
else
  ros2 bag play "${RAW_BAG}" --rate "${PLAY_RATE}" &
fi
PLAY_PID="$!"
PIDS+=("${PLAY_PID}")

wait "${PLAY_PID}"

echo "[localize-record] replay finished, stopping recorder/localization..."
if kill -0 "${REC_PID}" >/dev/null 2>&1; then
  kill -INT "${REC_PID}" >/dev/null 2>&1 || true
  wait "${REC_PID}" 2>/dev/null || true
fi

echo "[localize-record] done: ${OUT_BAG}"
