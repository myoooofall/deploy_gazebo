#!/usr/bin/env python3
import argparse
import csv
import math
import os
from bisect import bisect_left
from typing import Dict, List, Optional

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_sec(ns: int) -> float:
    return float(ns) / 1e9


def try_open_reader(uri: str, storage_id: Optional[str]) -> rosbag2_py.SequentialReader:
    if storage_id:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)
        return reader

    last_exc = None
    for sid in ("sqlite3", "mcap"):
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=sid)
            converter_options = rosbag2_py.ConverterOptions('', '')
            reader.open(storage_options, converter_options)
            return reader
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"failed to open bag '{uri}' as sqlite3/mcap: {last_exc}")


def nearest_index(sorted_times: List[int], t: int) -> Optional[int]:
    if not sorted_times:
        return None
    i = bisect_left(sorted_times, t)
    if i == 0:
        return 0
    if i == len(sorted_times):
        return len(sorted_times) - 1
    prev_t = sorted_times[i - 1]
    next_t = sorted_times[i]
    return i - 1 if abs(prev_t - t) <= abs(next_t - t) else i


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def compose_map_from_body(base_x: float, base_y: float, base_yaw: float,
                          body_x: float, body_y: float, body_yaw: float):
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    map_x = base_x + c * body_x - s * body_y
    map_y = base_y + s * body_x + c * body_y
    map_yaw = wrap_pi(base_yaw + body_yaw)
    return map_x, map_y, map_yaw


def to_body_from_map(base_x: float, base_y: float, base_yaw: float,
                     map_x: float, map_y: float, map_yaw: float):
    dx = map_x - base_x
    dy = map_y - base_y
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    body_x = c * dx + s * dy
    body_y = -s * dx + c * dy
    body_yaw = wrap_pi(map_yaw - base_yaw)
    return body_x, body_y, body_yaw


def build_goal_segments(actual_rows: List[Dict], eps: float = 1e-6) -> List[Dict]:
    if not actual_rows:
        return []
    segs: List[Dict] = []
    cur = {
        "start_t": actual_rows[0]["t_ns"],
        "frame_id": actual_rows[0]["frame_id"],
        "goal_x": float(actual_rows[0]["x"]),
        "goal_y": float(actual_rows[0]["y"]),
        "goal_yaw": float(actual_rows[0]["yaw"]),
    }
    for r in actual_rows[1:]:
        same_frame = (r["frame_id"] == cur["frame_id"])
        dx = abs(float(r["x"]) - cur["goal_x"])
        dy = abs(float(r["y"]) - cur["goal_y"])
        dyaw = abs(wrap_pi(float(r["yaw"]) - cur["goal_yaw"]))
        if same_frame and dx <= eps and dy <= eps and dyaw <= eps:
            continue

        cur["end_t"] = r["t_ns"]
        segs.append(cur)
        cur = {
            "start_t": r["t_ns"],
            "frame_id": r["frame_id"],
            "goal_x": float(r["x"]),
            "goal_y": float(r["y"]),
            "goal_yaw": float(r["yaw"]),
        }

    cur["end_t"] = 2**63 - 1
    segs.append(cur)
    return segs


def main():
    parser = argparse.ArgumentParser(description="Export offline goal/odometry metrics from rosbag2")
    parser.add_argument("bag", help="bag directory path")
    parser.add_argument("--out-dir", default="", help="output directory (default: <bag>/analysis)")
    parser.add_argument("--storage-id", default="", help="force bag storage id: sqlite3 or mcap")
    parser.add_argument("--max-sync-dt", type=float, default=0.2, help="max seconds for nearest timestamp association")
    args = parser.parse_args()

    bag_dir = os.path.abspath(args.bag)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(bag_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    reader = try_open_reader(bag_dir, args.storage_id if args.storage_id else None)

    topic_types = reader.get_all_topics_and_types()
    type_map: Dict[str, str] = {t.name: t.type for t in topic_types}

    target_topics = {
        "/Odometry",
        "/nav/goal_actual_map",
        "/nav/goal_pred_map",
        "/nav/goal_error_body",
    }

    msg_type_cache = {}

    odom_rows: List[Dict] = []
    actual_rows: List[Dict] = []
    pred_rows: List[Dict] = []
    err_rows: List[Dict] = []

    while reader.has_next():
        topic, data, t_bag = reader.read_next()
        if topic not in target_topics or topic not in type_map:
            continue

        if topic not in msg_type_cache:
            msg_type_cache[topic] = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type_cache[topic])

        if topic == "/Odometry":
            t_ns = stamp_to_ns(msg.header.stamp)
            if t_ns <= 0:
                t_ns = int(t_bag)
            odom_rows.append({
                "t_ns": t_ns,
                "t_sec": ns_to_sec(t_ns),
                "x": float(msg.pose.pose.position.x),
                "y": float(msg.pose.pose.position.y),
                "yaw": float(quat_to_yaw(msg.pose.pose.orientation)),
            })
        elif topic in ("/nav/goal_actual_map", "/nav/goal_pred_map"):
            t_ns = stamp_to_ns(msg.header.stamp)
            if t_ns <= 0:
                t_ns = int(t_bag)
            row = {
                "t_ns": t_ns,
                "t_sec": ns_to_sec(t_ns),
                "frame_id": msg.header.frame_id,
                "x": float(msg.pose.position.x),
                "y": float(msg.pose.position.y),
                "yaw": float(quat_to_yaw(msg.pose.orientation)),
            }
            if topic == "/nav/goal_actual_map":
                actual_rows.append(row)
            else:
                pred_rows.append(row)
        else:  # /nav/goal_error_body
            t_ns = int(t_bag)
            err_rows.append({
                "t_ns": t_ns,
                "t_sec": ns_to_sec(t_ns),
                "dx": float(msg.x),
                "dy": float(msg.y),
                "dyaw": float(msg.z),
                "e_xy": float(math.hypot(msg.x, msg.y)),
            })

    odom_rows.sort(key=lambda r: r["t_ns"])
    actual_rows.sort(key=lambda r: r["t_ns"])
    pred_rows.sort(key=lambda r: r["t_ns"])
    err_rows.sort(key=lambda r: r["t_ns"])

    write_csv(
        os.path.join(out_dir, "odometry.csv"),
        ["t_ns", "t_sec", "x", "y", "yaw"],
        odom_rows,
    )
    write_csv(
        os.path.join(out_dir, "goal_actual.csv"),
        ["t_ns", "t_sec", "frame_id", "x", "y", "yaw"],
        actual_rows,
    )
    write_csv(
        os.path.join(out_dir, "goal_pred.csv"),
        ["t_ns", "t_sec", "frame_id", "x", "y", "yaw"],
        pred_rows,
    )
    write_csv(
        os.path.join(out_dir, "goal_error_body.csv"),
        ["t_ns", "t_sec", "dx", "dy", "dyaw", "e_xy"],
        err_rows,
    )

    max_sync_ns = int(args.max_sync_dt * 1e9)

    # 1) Simple same-frame nearest alignment (legacy behavior)
    actual_times = [r["t_ns"] for r in actual_rows]
    aligned_rows: List[Dict] = []
    for pr in pred_rows:
        idx = nearest_index(actual_times, pr["t_ns"])
        if idx is None:
            continue
        ar = actual_rows[idx]
        dt = abs(pr["t_ns"] - ar["t_ns"])
        if dt > max_sync_ns or ar["frame_id"] != pr["frame_id"]:
            continue

        dx = pr["x"] - ar["x"]
        dy = pr["y"] - ar["y"]
        dyaw = wrap_pi(pr["yaw"] - ar["yaw"])
        aligned_rows.append({
            "t_pred_sec": pr["t_sec"],
            "t_actual_sec": ar["t_sec"],
            "frame_id": pr["frame_id"],
            "sync_dt_sec": ns_to_sec(dt),
            "pred_x": pr["x"],
            "pred_y": pr["y"],
            "pred_yaw": pr["yaw"],
            "actual_x": ar["x"],
            "actual_y": ar["y"],
            "actual_yaw": ar["yaw"],
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "e_xy": math.hypot(dx, dy),
        })

    write_csv(
        os.path.join(out_dir, "goal_compare_aligned.csv"),
        [
            "t_pred_sec", "t_actual_sec", "frame_id", "sync_dt_sec",
            "pred_x", "pred_y", "pred_yaw",
            "actual_x", "actual_y", "actual_yaw",
            "dx", "dy", "dyaw", "e_xy",
        ],
        aligned_rows,
    )

    # 2) Offline odom-based body-error reconstruction (works in no-/Odometry online fallback)
    odom_times = [r["t_ns"] for r in odom_rows]
    goal_segments = build_goal_segments(actual_rows)
    offline_rows: List[Dict] = []

    if odom_rows and goal_segments:
        seg_idx = 0
        for pr in pred_rows:
            # Move to segment that contains pred timestamp
            while seg_idx + 1 < len(goal_segments) and pr["t_ns"] >= goal_segments[seg_idx]["end_t"]:
                seg_idx += 1
            seg = goal_segments[seg_idx]

            # Get odom at pred timestamp
            od_idx = nearest_index(odom_times, pr["t_ns"])
            if od_idx is None:
                continue
            od = odom_rows[od_idx]
            dt_pred_odom = abs(pr["t_ns"] - od["t_ns"])
            if dt_pred_odom > max_sync_ns:
                continue

            # Resolve current segment's map goal
            if seg["frame_id"] == "map":
                goal_map_x = seg["goal_x"]
                goal_map_y = seg["goal_y"]
                goal_map_yaw = seg["goal_yaw"]
            elif seg["frame_id"] == "base_link":
                od0_idx = nearest_index(odom_times, seg["start_t"])
                if od0_idx is None:
                    continue
                od0 = odom_rows[od0_idx]
                dt_goal_odom = abs(seg["start_t"] - od0["t_ns"])
                if dt_goal_odom > max_sync_ns:
                    continue
                goal_map_x, goal_map_y, goal_map_yaw = compose_map_from_body(
                    od0["x"], od0["y"], od0["yaw"],
                    seg["goal_x"], seg["goal_y"], seg["goal_yaw"],
                )
            else:
                continue

            actual_body_x, actual_body_y, actual_body_yaw = to_body_from_map(
                od["x"], od["y"], od["yaw"], goal_map_x, goal_map_y, goal_map_yaw
            )

            if pr["frame_id"] == "base_link":
                pred_body_x = pr["x"]
                pred_body_y = pr["y"]
                pred_body_yaw = pr["yaw"]
            elif pr["frame_id"] == "map":
                pred_body_x, pred_body_y, pred_body_yaw = to_body_from_map(
                    od["x"], od["y"], od["yaw"], pr["x"], pr["y"], pr["yaw"]
                )
            else:
                continue

            dx = pred_body_x - actual_body_x
            dy = pred_body_y - actual_body_y
            dyaw = wrap_pi(pred_body_yaw - actual_body_yaw)

            offline_rows.append({
                "t_pred_sec": pr["t_sec"],
                "pred_frame": pr["frame_id"],
                "goal_frame": seg["frame_id"],
                "sync_pred_odom_sec": ns_to_sec(dt_pred_odom),
                "pred_body_x": pred_body_x,
                "pred_body_y": pred_body_y,
                "pred_body_yaw": pred_body_yaw,
                "actual_body_x": actual_body_x,
                "actual_body_y": actual_body_y,
                "actual_body_yaw": actual_body_yaw,
                "dx": dx,
                "dy": dy,
                "dyaw": dyaw,
                "e_xy": math.hypot(dx, dy),
            })

    write_csv(
        os.path.join(out_dir, "goal_error_offline_odom.csv"),
        [
            "t_pred_sec", "pred_frame", "goal_frame", "sync_pred_odom_sec",
            "pred_body_x", "pred_body_y", "pred_body_yaw",
            "actual_body_x", "actual_body_y", "actual_body_yaw",
            "dx", "dy", "dyaw", "e_xy",
        ],
        offline_rows,
    )

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"bag={bag_dir}\n")
        f.write(f"out_dir={out_dir}\n")
        f.write(f"count_odom={len(odom_rows)}\n")
        f.write(f"count_goal_actual={len(actual_rows)}\n")
        f.write(f"count_goal_pred={len(pred_rows)}\n")
        f.write(f"count_goal_error_body={len(err_rows)}\n")
        f.write(f"count_goal_aligned={len(aligned_rows)}\n")
        f.write(f"count_goal_offline_odom={len(offline_rows)}\n")

        if err_rows:
            e_xy = [float(r["e_xy"]) for r in err_rows]
            e_xy_sorted = sorted(e_xy)
            p95 = e_xy_sorted[int(0.95 * (len(e_xy_sorted) - 1))]
            f.write(f"goal_error_body_e_xy_mean={sum(e_xy)/len(e_xy):.6f}\n")
            f.write(f"goal_error_body_e_xy_p95={p95:.6f}\n")
            f.write(f"goal_error_body_e_xy_max={max(e_xy):.6f}\n")

        if aligned_rows:
            e_xy2 = [float(r["e_xy"]) for r in aligned_rows]
            e_xy2_sorted = sorted(e_xy2)
            p95_2 = e_xy2_sorted[int(0.95 * (len(e_xy2_sorted) - 1))]
            f.write(f"goal_aligned_e_xy_mean={sum(e_xy2)/len(e_xy2):.6f}\n")
            f.write(f"goal_aligned_e_xy_p95={p95_2:.6f}\n")
            f.write(f"goal_aligned_e_xy_max={max(e_xy2):.6f}\n")

        if offline_rows:
            e_xy3 = [float(r["e_xy"]) for r in offline_rows]
            e_xy3_sorted = sorted(e_xy3)
            p95_3 = e_xy3_sorted[int(0.95 * (len(e_xy3_sorted) - 1))]
            f.write(f"goal_offline_odom_e_xy_mean={sum(e_xy3)/len(e_xy3):.6f}\n")
            f.write(f"goal_offline_odom_e_xy_p95={p95_3:.6f}\n")
            f.write(f"goal_offline_odom_e_xy_max={max(e_xy3):.6f}\n")

    print(f"[offline-export] done: {out_dir}")
    print(f"[offline-export] summary: {summary_path}")


if __name__ == "__main__":
    main()
