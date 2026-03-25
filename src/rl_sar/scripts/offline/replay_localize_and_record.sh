#!/usr/bin/env bash
set -euo pipefail

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

cleanup() {
  set +e
  if [[ -n "${PLAY_PID:-}" ]] && kill -0 "$PLAY_PID" 2>/dev/null; then
    kill -INT "$PLAY_PID" 2>/dev/null
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
bash -lc "$LOCALIZATION_CMD" &
LOC_PID=$!
sleep 2

# 2) Record enriched topics.
ros2 bag record -o "$OUT_DIR" \
  /Odometry \
  /tf /tf_static \
  /nav/goal_actual_map /nav/goal_pred_map /nav/goal_error_body \
  /nav/goal_compare_markers \
  /nav_goal_body \
  /rslidar_points /imu/data &
REC_PID=$!
sleep 1

# 3) Replay raw bag.
ros2 bag play "$RAW_BAG" --clock &
PLAY_PID=$!
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
