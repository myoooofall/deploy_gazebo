#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BAG_BASE_DIR="${RL_SAR_DIR}/bag"

usage() {
  cat <<EOF
Usage: $(basename "$0") <run_name> [--full]

Record raw topics needed for offline localization/SLAM while protecting
rl_real_lite3 on Jetson NX. This script can live inside rl_sar; it only needs
the ROS environment and visible topics.

Default output:
  ${DEFAULT_BAG_BASE_DIR}/<run_name>/lidar
  ${DEFAULT_BAG_BASE_DIR}/<run_name>/small
  ${DEFAULT_BAG_BASE_DIR}/<run_name>/depth_norm

Environment overrides:
  BAG_BASE_DIR            default: ${DEFAULT_BAG_BASE_DIR}
  SMALL_BAG_TASKSET_CPUS  default: 4
  LIDAR_BAG_TASKSET_CPUS  default: 5
  DEPTH_BAG_TASKSET_CPUS  default: 4
  SMALL_BAG_NICE          default: 0
  LIDAR_BAG_NICE          default: 10
  DEPTH_BAG_NICE          default: 5
  SMALL_BAG_IONICE_CLASS  default: 2
  SMALL_BAG_IONICE_LEVEL  default: 4
  LIDAR_BAG_IONICE_CLASS  default: 2
  LIDAR_BAG_IONICE_LEVEL  default: 7
  DEPTH_BAG_IONICE_CLASS  default: 2
  DEPTH_BAG_IONICE_LEVEL  default: 6
  DEPTH_BAG_ENABLE        default: true

Examples:
  ./record_nav_raw_bag_nx.sh walk_01
  SMALL_BAG_TASKSET_CPUS=4 LIDAR_BAG_TASKSET_CPUS=5 ./record_nav_raw_bag_nx.sh walk_02
  DEPTH_BAG_ENABLE=false ./record_nav_raw_bag_nx.sh walk_03
  BAG_BASE_DIR=/mnt/ssd/bags ./record_nav_raw_bag_nx.sh walk_03

Use --full only when debugging localization outputs; it records derived topics
that are not needed for offline replay.

Depth recording only records /camera/depth/processed_norm. Enable
depth_debug_publish_enable in rl_real_lite3 config before using it.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUN_NAME="$1"
MODE="raw"
if [[ "${2:-}" == "--full" ]]; then
  MODE="full"
elif [[ -n "${2:-}" ]]; then
  echo "[record-nav-raw] ERROR: unknown option '${2}'" >&2
  usage >&2
  exit 2
fi

BASE_DIR="${BAG_BASE_DIR:-${DEFAULT_BAG_BASE_DIR}}"
OUT_DIR="${BASE_DIR%/}/${RUN_NAME}"
LIDAR_OUT_DIR="${OUT_DIR}/lidar"
SMALL_OUT_DIR="${OUT_DIR}/small"
DEPTH_OUT_DIR="${OUT_DIR}/depth_norm"
mkdir -p "${BASE_DIR}"

# Recommended pairing:
#   rl_real_lite3: taskset -c 0-3 ros2 run rl_sar rl_real_lite3
#   rosbag small: this script defaults to taskset -c 4, normal CPU priority
#   rosbag lidar: this script defaults to taskset -c 5, lower CPU/I/O priority
SMALL_BAG_TASKSET_CPUS="${SMALL_BAG_TASKSET_CPUS:-4}"
LIDAR_BAG_TASKSET_CPUS="${LIDAR_BAG_TASKSET_CPUS:-5}"
DEPTH_BAG_TASKSET_CPUS="${DEPTH_BAG_TASKSET_CPUS:-4}"
SMALL_BAG_NICE="${SMALL_BAG_NICE:-0}"
LIDAR_BAG_NICE="${LIDAR_BAG_NICE:-10}"
DEPTH_BAG_NICE="${DEPTH_BAG_NICE:-5}"
SMALL_BAG_IONICE_CLASS="${SMALL_BAG_IONICE_CLASS:-2}"
SMALL_BAG_IONICE_LEVEL="${SMALL_BAG_IONICE_LEVEL:-4}"
LIDAR_BAG_IONICE_CLASS="${LIDAR_BAG_IONICE_CLASS:-2}"
LIDAR_BAG_IONICE_LEVEL="${LIDAR_BAG_IONICE_LEVEL:-7}"
DEPTH_BAG_IONICE_CLASS="${DEPTH_BAG_IONICE_CLASS:-2}"
DEPTH_BAG_IONICE_LEVEL="${DEPTH_BAG_IONICE_LEVEL:-6}"
DEPTH_BAG_ENABLE="${DEPTH_BAG_ENABLE:-true}"

LIDAR_TOPICS=(
  /rslidar_points
)

DEPTH_TOPICS=(
  /camera/depth/processed_norm
)

SMALL_TOPICS=(
  /imu/data
  /tf
  /tf_static
  /odom
  /Odometry
  /status
  /nav_goal_body
  /nav/goal_actual_map
  /nav/goal_pred_map
  /nav/goal_error_body
  /nav/cmd_high
  /nav/cmd_applied
)

if [[ "${MODE}" == "full" ]]; then
  SMALL_TOPICS+=(
    /nav/goal_compare_markers
    /aligned_points
    /map
    /map_updates
  )
fi

start_recorder() {
  local out_dir="$1"
  local taskset_cpus="$2"
  local nice_level="$3"
  local ionice_class="$4"
  local ionice_level="$5"
  shift 5

  local cmd=(ros2 bag record -o "${out_dir}" "$@")

  if command -v ionice >/dev/null 2>&1; then
    cmd=(ionice -c "${ionice_class}" -n "${ionice_level}" "${cmd[@]}")
  fi

  if [[ -n "${taskset_cpus}" ]] && command -v taskset >/dev/null 2>&1; then
    cmd=(taskset -c "${taskset_cpus}" "${cmd[@]}")
  fi

  if command -v nice >/dev/null 2>&1; then
    cmd=(nice -n "${nice_level}" "${cmd[@]}")
  fi

  "${cmd[@]}" &
  local pid="$!"
  PIDS+=("${pid}")
}

PIDS=()
cleanup() {
  local status=$?
  trap - INT TERM EXIT
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo
    echo "[record-nav-raw] stopping recorders..."
    for pid in "${PIDS[@]}" $(jobs -pr); do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -INT "${pid}" >/dev/null 2>&1 || true
      fi
    done
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup INT TERM EXIT

echo "[record-nav-raw] mode=${MODE}"
echo "[record-nav-raw] output=${OUT_DIR}"
echo "[record-nav-raw] lidar_output=${LIDAR_OUT_DIR}"
echo "[record-nav-raw] small_output=${SMALL_OUT_DIR}"
if [[ "${DEPTH_BAG_ENABLE}" == "true" ]]; then
  echo "[record-nav-raw] depth_output=${DEPTH_OUT_DIR}"
fi
echo "[record-nav-raw] lidar_topics=${#LIDAR_TOPICS[@]} small_topics=${#SMALL_TOPICS[@]} depth_topics=${#DEPTH_TOPICS[@]} depth_enable=${DEPTH_BAG_ENABLE}"
echo "[record-nav-raw] small taskset=${SMALL_BAG_TASKSET_CPUS:-none} nice=${SMALL_BAG_NICE} ionice=${SMALL_BAG_IONICE_CLASS}:${SMALL_BAG_IONICE_LEVEL}"
echo "[record-nav-raw] lidar taskset=${LIDAR_BAG_TASKSET_CPUS:-none} nice=${LIDAR_BAG_NICE} ionice=${LIDAR_BAG_IONICE_CLASS}:${LIDAR_BAG_IONICE_LEVEL}"
if [[ "${DEPTH_BAG_ENABLE}" == "true" ]]; then
  echo "[record-nav-raw] depth taskset=${DEPTH_BAG_TASKSET_CPUS:-none} nice=${DEPTH_BAG_NICE} ionice=${DEPTH_BAG_IONICE_CLASS}:${DEPTH_BAG_IONICE_LEVEL}"
fi
echo "[record-nav-raw] NOTE: keep NoMachine/RViz closed during recording when possible."

if [[ -e "${OUT_DIR}" ]]; then
  echo "[record-nav-raw] ERROR: output already exists: ${OUT_DIR}" >&2
  echo "[record-nav-raw] Existing content:" >&2
  find "${OUT_DIR}" -maxdepth 2 -mindepth 1 -print >&2 || true
  echo "[record-nav-raw] Choose a new run_name, for example: ${RUN_NAME}_2" >&2
  echo "[record-nav-raw] Or remove the old incomplete directory before recording again." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[record-nav-raw] starting small-topic recorder..."
start_recorder "${SMALL_OUT_DIR}" "${SMALL_BAG_TASKSET_CPUS}" "${SMALL_BAG_NICE}" "${SMALL_BAG_IONICE_CLASS}" "${SMALL_BAG_IONICE_LEVEL}" "${SMALL_TOPICS[@]}"

sleep 0.2

echo "[record-nav-raw] starting lidar recorder..."
start_recorder "${LIDAR_OUT_DIR}" "${LIDAR_BAG_TASKSET_CPUS}" "${LIDAR_BAG_NICE}" "${LIDAR_BAG_IONICE_CLASS}" "${LIDAR_BAG_IONICE_LEVEL}" "${LIDAR_TOPICS[@]}"

sleep 0.2

if [[ "${DEPTH_BAG_ENABLE}" == "true" ]]; then
  echo "[record-nav-raw] starting depth_norm recorder..."
  start_recorder "${DEPTH_OUT_DIR}" "${DEPTH_BAG_TASKSET_CPUS}" "${DEPTH_BAG_NICE}" "${DEPTH_BAG_IONICE_CLASS}" "${DEPTH_BAG_IONICE_LEVEL}" "${DEPTH_TOPICS[@]}"
fi

echo "[record-nav-raw] all recorders started. Press Ctrl+C to stop recording."

wait "${PIDS[@]}"
