#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SAR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BAG_BASE_DIR="${RL_SAR_DIR}/bag"

usage() {
  cat <<EOF
Usage: $(basename "$0") <run_name_or_split_bag_dir> [--rate RATE] [--clock]

Replay split bags recorded by record_nav_raw_bag_nx.sh.

Expected split layout:
  <bag_dir>/lidar
  <bag_dir>/small

If the first argument is not an existing path, it is resolved as:
  ${DEFAULT_BAG_BASE_DIR}/<run_name>

Options:
  --rate RATE   Playback rate, default: 0.5
  --clock       Pass --clock to both ros2 bag play commands if supported

Examples:
  ./replay_nav_split_bag.sh walk_01
  ./replay_nav_split_bag.sh walk_01 --rate 0.25
  ./replay_nav_split_bag.sh /mnt/ssd/bags/walk_01 --rate 0.5

Stop live lidar/IMU publishers before replaying offline bags.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

INPUT="$1"
shift

RATE="0.5"
USE_CLOCK="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rate)
      RATE="${2:-}"
      if [[ -z "${RATE}" ]]; then
        echo "[replay-nav-split] ERROR: --rate requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --clock)
      USE_CLOCK="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[replay-nav-split] ERROR: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -d "${INPUT}" ]]; then
  BAG_DIR="$(cd "${INPUT}" && pwd)"
else
  BAG_DIR="${DEFAULT_BAG_BASE_DIR}/${INPUT}"
fi

LIDAR_BAG="${BAG_DIR}/lidar"
SMALL_BAG="${BAG_DIR}/small"

if [[ ! -d "${LIDAR_BAG}" ]]; then
  echo "[replay-nav-split] ERROR: missing lidar bag directory: ${LIDAR_BAG}" >&2
  exit 1
fi

if [[ ! -d "${SMALL_BAG}" ]]; then
  echo "[replay-nav-split] ERROR: missing small-topic bag directory: ${SMALL_BAG}" >&2
  exit 1
fi

PLAY_ARGS=(--rate "${RATE}")
if [[ "${USE_CLOCK}" == "1" ]]; then
  PLAY_ARGS+=(--clock)
fi

bag_start_ns() {
  local bag_dir="$1"
  python3 - "$bag_dir/metadata.yaml" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8").read()
match = re.search(r"starting_time:\s*\n\s*nanoseconds_since_epoch:\s*([0-9]+)", text)
if not match:
    raise SystemExit("0")
print(match.group(1))
PY
}

SMALL_START_NS="$(bag_start_ns "${SMALL_BAG}")"
LIDAR_START_NS="$(bag_start_ns "${LIDAR_BAG}")"

start_player() {
  local label="$1"
  local bag_dir="$2"
  echo "[replay-nav-split] starting ${label}: ${bag_dir}"
  ros2 bag play "${bag_dir}" "${PLAY_ARGS[@]}" &
  PIDS+=("$!")
}

sleep_for_record_offset() {
  local first_ns="$1"
  local second_ns="$2"
  python3 - "$first_ns" "$second_ns" "$RATE" <<'PY'
import sys
first = int(sys.argv[1])
second = int(sys.argv[2])
rate = float(sys.argv[3])
delay = max(0.0, (second - first) / 1e9 / rate)
print(f"{delay:.6f}")
PY
}

PIDS=()
cleanup() {
  local status=$?
  trap - INT TERM EXIT
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo
    echo "[replay-nav-split] stopping players..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -INT "${pid}" >/dev/null 2>&1 || true
      fi
    done
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup INT TERM EXIT

echo "[replay-nav-split] bag_dir=${BAG_DIR}"
echo "[replay-nav-split] lidar=${LIDAR_BAG}"
echo "[replay-nav-split] small=${SMALL_BAG}"
echo "[replay-nav-split] rate=${RATE} clock=${USE_CLOCK}"
echo "[replay-nav-split] Stop live lidar/IMU publishers before replaying offline bags."

if (( SMALL_START_NS <= LIDAR_START_NS )); then
  DELAY="$(sleep_for_record_offset "${SMALL_START_NS}" "${LIDAR_START_NS}")"
  start_player "small-topic bag" "${SMALL_BAG}"
  if [[ "${DELAY}" != "0.000000" ]]; then
    echo "[replay-nav-split] delaying lidar start by ${DELAY}s to match recorded start offset"
    sleep "${DELAY}"
  fi
  start_player "lidar bag" "${LIDAR_BAG}"
else
  DELAY="$(sleep_for_record_offset "${LIDAR_START_NS}" "${SMALL_START_NS}")"
  start_player "lidar bag" "${LIDAR_BAG}"
  if [[ "${DELAY}" != "0.000000" ]]; then
    echo "[replay-nav-split] delaying small-topic start by ${DELAY}s to match recorded start offset"
    sleep "${DELAY}"
  fi
  start_player "small-topic bag" "${SMALL_BAG}"
fi

wait "${PIDS[@]}"
