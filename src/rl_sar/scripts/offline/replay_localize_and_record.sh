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
  echo "Usage: $0 <raw_bag_path> [output_run_name]"
  echo "Example: LOCALIZATION_CMD='ros2 launch hdl_localization lite_localization.launch.py enable_nav2:=false use_sim_time:=true map_path:=/path/map.pcd' $0 ~/bags/lite3/raw/run_xxx"
  exit 1
fi

RAW_BAG="$1"
RUN_NAME="${2:-$(basename "$RAW_BAG")_enriched_$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${ENRICHED_BAG_BASE_DIR:-$HOME/bags/lite3/enriched}"
OUT_DIR="${OUT_BASE}/${RUN_NAME}"

if [[ ! -d "$RAW_BAG" ]]; then
  echo "[offline-replay] raw bag path not found: $RAW_BAG"
  exit 2
fi

if [[ -z "${LOCALIZATION_CMD:-}" ]]; then
  echo "[offline-replay] LOCALIZATION_CMD is required."
  echo "Set it to your offline localization launch command (must publish /Odometry, use_sim_time:=true)."
  exit 3
fi

mkdir -p "$OUT_BASE"

echo "[offline-replay] raw bag: $RAW_BAG"
echo "[offline-replay] output bag: $OUT_DIR"
echo "[offline-replay] localization cmd: $LOCALIZATION_CMD"

PLAY_RATE="${PLAY_RATE:-1.0}"
PLAY_LOOP="${PLAY_LOOP:-0}"
PLAY_START_OFFSET="${PLAY_START_OFFSET:-0}"
PLAY_START_PAUSED="${PLAY_START_PAUSED:-0}"
echo "[offline-replay] play args: rate=${PLAY_RATE}, loop=${PLAY_LOOP}, start_offset=${PLAY_START_OFFSET}, start_paused=${PLAY_START_PAUSED}"

cleanup() {
  set +e
  if [[ -n "${PLAY_PID:-}" ]] && kill -0 "$PLAY_PID" 2>/dev/null; then
    kill -INT "$PLAY_PID" 2>/dev/null
  fi
  if [[ -n "${GOAL_REPUB_PID:-}" ]] && kill -0 "$GOAL_REPUB_PID" 2>/dev/null; then
    kill -INT "$GOAL_REPUB_PID" 2>/dev/null
  fi
  if [[ -n "${REC_PID:-}" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null
  fi
  if [[ -n "${LOC_PID:-}" ]] && kill -0 "$LOC_PID" 2>/dev/null; then
    kill -INT "$LOC_PID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

# 1) Start localization first (must use simulated clock from bag).
bash -lc "source /opt/ros/humble/setup.bash; \
  [[ -f '${WORKSPACE_DIR}/lite_cog_ros2/nav/install/setup.bash' ]] && source '${WORKSPACE_DIR}/lite_cog_ros2/nav/install/setup.bash'; \
  [[ -f '${WORKSPACE_DIR}/install/setup.bash' ]] && source '${WORKSPACE_DIR}/install/setup.bash'; \
  ${LOCALIZATION_CMD}" &
LOC_PID=$!
sleep 2
if ! kill -0 "$LOC_PID" 2>/dev/null; then
  echo "[offline-replay] localization process exited early."
  echo "[offline-replay] verify package visibility: ros2 pkg list | grep -x hdl_localization"
  exit 4
fi

# 2) Start goal republisher for RViz map-frame goal visualization.
python3 "${SCRIPT_DIR}/goal_republisher.py" &
GOAL_REPUB_PID=$!
sleep 1

# 3) Replay raw bag.
PLAY_ARGS=(
  "$RAW_BAG"
  --clock
  --qos-profile-overrides-path "${SCRIPT_DIR}/qos_overrides.yaml"
  --rate "${PLAY_RATE}"
)
if [[ "${PLAY_LOOP}" == "1" ]]; then
  PLAY_ARGS+=(--loop)
fi
if [[ "${PLAY_START_PAUSED}" == "1" ]]; then
  PLAY_ARGS+=(--start-paused)
fi
if [[ "${PLAY_START_OFFSET}" != "0" ]]; then
  PLAY_ARGS+=(--start-offset "${PLAY_START_OFFSET}")
fi

ros2 bag play "${PLAY_ARGS[@]}" &
PLAY_PID=$!
sleep 1
clock_pub_count="$(ros2 topic info /clock 2>/dev/null | awk '/Publisher count:/ {print $3}' | tail -n1)"
if [[ -n "${clock_pub_count}" ]] && [[ "${clock_pub_count}" -gt 1 ]]; then
  echo "[offline-replay] ERROR: /clock has ${clock_pub_count} publishers (expected 1)."
  echo "[offline-replay] This will cause 'Detected jump back in time'."
  echo "[offline-replay] Stop other ros2 bag play / simulators in the same ROS_DOMAIN_ID and retry."
  exit 5
fi

# 4) Wait until navigation outputs first pred+actual, then start recording.
WAIT_GOAL_TIMEOUT_SEC="${WAIT_GOAL_TIMEOUT_SEC:-180}"
echo "[offline-replay] waiting for first /nav/goal_actual_map and /nav/goal_pred_map (timeout=${WAIT_GOAL_TIMEOUT_SEC}s)"
timeout "${WAIT_GOAL_TIMEOUT_SEC}s" ros2 topic echo /nav/goal_actual_map --once --no-daemon >/dev/null &
WAIT_ACTUAL_PID=$!
timeout "${WAIT_GOAL_TIMEOUT_SEC}s" ros2 topic echo /nav/goal_pred_map --once --no-daemon >/dev/null &
WAIT_PRED_PID=$!
wait "$WAIT_ACTUAL_PID"
wait "$WAIT_PRED_PID"
echo "[offline-replay] first goal_actual + goal_pred received, starting recorder"

# 5) Record enriched topics from goal-start onward.
ros2 bag record -o "$OUT_DIR" \
  /Odometry \
  /odom \
  /odom_viz \
  /tf /tf_static \
  /nav/goal_actual_map /nav/goal_pred_map /nav/goal_error_body \
  /nav/goal_actual_map_viz /nav/goal_pred_map_viz \
  /nav/cmd_high /nav/cmd_applied \
  /nav/goal_compare_markers \
  /nav_goal_body \
  /rslidar_points /imu/data &
REC_PID=$!
sleep 1

wait "$PLAY_PID"

# Give recorder a little time to flush.
sleep 2
if kill -0 "$REC_PID" 2>/dev/null; then
  kill -INT "$REC_PID"
  wait "$REC_PID" || true
fi
if kill -0 "$LOC_PID" 2>/dev/null; then
  kill -INT "$LOC_PID"
  wait "$LOC_PID" || true
fi

echo "[offline-replay] done. enriched bag saved to: $OUT_DIR"
