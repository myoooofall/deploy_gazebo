#!/usr/bin/env python3
"""Build a presentation rosbag from offline-localization odometry.

The output bag contains a cropped and smoothed trajectory. It is meant for RViz
playback/video overlay, not for feeding back into localization.
"""

import argparse
import math
import shutil
from pathlib import Path as FsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray


def open_reader(uri: FsPath) -> SequentialReader:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(uri), storage_id="sqlite3"), ConverterOptions("", ""))
    return reader


def find_bag_dirs(path: FsPath) -> List[FsPath]:
    if (path / "metadata.yaml").is_file():
        return [path]
    return sorted(child for child in path.iterdir() if child.is_dir() and (child / "metadata.yaml").is_file())


def collect_topic_types(bag_dirs: Sequence[FsPath]) -> Dict[str, str]:
    topic_types: Dict[str, str] = {}
    for bag_dir in bag_dirs:
        reader = open_reader(bag_dir)
        for topic in reader.get_all_topics_and_types():
            topic_types[topic.name] = topic.type
    return topic_types


def iter_bag_messages(bag_dirs: Sequence[FsPath], wanted_topics: Iterable[str]):
    wanted = set(wanted_topics)
    topic_types = collect_topic_types(bag_dirs)
    msg_type_cache = {topic: get_message(topic_types[topic]) for topic in wanted if topic in topic_types}
    messages = []
    for bag_dir in bag_dirs:
        reader = open_reader(bag_dir)
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic in wanted:
                messages.append((int(timestamp), topic, data))
    messages.sort(key=lambda item: item[0])
    for timestamp, topic, data in messages:
        yield timestamp, topic, deserialize_message(data, msg_type_cache[topic])


def stamp_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_stamp(ns: int) -> Time:
    stamp = Time()
    stamp.sec = int(ns // 1_000_000_000)
    stamp.nanosec = int(ns % 1_000_000_000)
    return stamp


def yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quat_from_yaw(yaw: float):
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > values.size:
        window = values.size if values.size % 2 == 1 else max(1, values.size - 1)
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def wrap_angle_delta(delta: float) -> float:
    return math.atan2(math.sin(delta), math.cos(delta))


def interpolate_short_outliers(
    samples: Sequence[Tuple[int, float, float, float, float]],
    max_step_dist: float,
    max_step_yaw: float,
    max_outlier_run: int,
    max_interp_speed: float,
) -> Tuple[List[Tuple[int, float, float, float, float]], int, bool]:
    if len(samples) < 2:
        return list(samples), 0, False
    if max_step_dist <= 0.0 and max_step_yaw <= 0.0:
        return list(samples), 0, False

    cleaned: List[Tuple[int, float, float, float, float]] = [samples[0]]
    replaced = 0
    truncated = False
    max_repair_run = max_outlier_run if max_outlier_run > 0 else 5

    def transition_bad(
        a: Tuple[int, float, float, float, float],
        b: Tuple[int, float, float, float, float],
    ) -> bool:
        _, ax, ay, _, ayaw = a
        _, bx, by, _, byaw = b
        dist = math.hypot(bx - ax, by - ay)
        dyaw = abs(wrap_angle_delta(byaw - ayaw))
        bad_pos = max_step_dist > 0.0 and dist > max_step_dist
        bad_yaw = max_step_yaw > 0.0 and dyaw > max_step_yaw
        return bad_pos or bad_yaw

    def bridge_ok(
        a: Tuple[int, float, float, float, float],
        b: Tuple[int, float, float, float, float],
    ) -> bool:
        at, ax, ay, _, ayaw = a
        bt, bx, by, _, byaw = b
        dt = max(1e-6, (bt - at) / 1e9)
        speed = math.hypot(bx - ax, by - ay) / dt
        dyaw_rate = abs(wrap_angle_delta(byaw - ayaw)) / dt
        speed_ok = max_interp_speed <= 0.0 or speed <= max_interp_speed
        yaw_ok = max_step_yaw <= 0.0 or dyaw_rate <= max(1.0, max_step_yaw * 3.0)
        return speed_ok and yaw_ok

    def interp_sample(
        prev: Tuple[int, float, float, float, float],
        nxt: Tuple[int, float, float, float, float],
        t: int,
    ) -> Tuple[int, float, float, float, float]:
        pt, px, py, pz, pyaw = prev
        nt, nx, ny, nz, nyaw = nxt
        alpha = 0.0 if nt == pt else (t - pt) / float(nt - pt)
        alpha = min(1.0, max(0.0, alpha))
        yaw_delta = wrap_angle_delta(nyaw - pyaw)
        return (
            t,
            px + alpha * (nx - px),
            py + alpha * (ny - py),
            pz + alpha * (nz - pz),
            pyaw + alpha * yaw_delta,
        )

    i = 1
    while i < len(samples):
        prev = cleaned[-1]
        cur = samples[i]
        if not transition_bad(prev, cur):
            cleaned.append(cur)
            i += 1
            continue

        repaired = False
        max_after = min(len(samples) - 1, i + max_repair_run)
        for after_idx in range(i + 1, max_after + 1):
            after = samples[after_idx]
            if not bridge_ok(prev, after):
                continue
            for bad_idx in range(i, after_idx):
                cleaned.append(interp_sample(prev, after, samples[bad_idx][0]))
            replaced += after_idx - i
            i = after_idx
            repaired = True
            break

        if repaired:
            continue

        cleaned.append(cur)
        i += 1
        if max_outlier_run > 0:
            recent_bad = 0
            for back_idx in range(len(cleaned) - 1, 0, -1):
                if transition_bad(cleaned[back_idx - 1], cleaned[back_idx]):
                    recent_bad += 1
                else:
                    break
            if recent_bad >= max_outlier_run:
                truncated = True
                break

    return cleaned, replaced, truncated


def parse_index_ranges(specs: Sequence[str]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ".." in part:
                left, right = part.split("..", 1)
            elif ":" in part:
                left, right = part.split(":", 1)
            elif "-" in part:
                left, right = part.split("-", 1)
            else:
                left = right = part
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            ranges.append((start, end))
    return ranges


def interpolate_index_ranges(
    samples: Sequence[Tuple[int, float, float, float, float]],
    ranges: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, float, float, float, float]], int, List[Tuple[int, int]]]:
    if not ranges:
        return list(samples), 0, []
    if len(samples) < 3:
        return list(samples), 0, []

    repaired = list(samples)
    replaced = 0
    applied: List[Tuple[int, int]] = []

    def interp_sample(
        prev: Tuple[int, float, float, float, float],
        nxt: Tuple[int, float, float, float, float],
        t: int,
    ) -> Tuple[int, float, float, float, float]:
        pt, px, py, pz, pyaw = prev
        nt, nx, ny, nz, nyaw = nxt
        alpha = 0.0 if nt == pt else (t - pt) / float(nt - pt)
        alpha = min(1.0, max(0.0, alpha))
        yaw_delta = wrap_angle_delta(nyaw - pyaw)
        return (
            t,
            px + alpha * (nx - px),
            py + alpha * (ny - py),
            pz + alpha * (nz - pz),
            pyaw + alpha * yaw_delta,
        )

    for start, end in sorted(ranges):
        if start <= 0 or end >= len(samples) - 1:
            print(f"[smooth-traj] skip repair index range {start}:{end}; need one sample before and after")
            continue
        prev = repaired[start - 1]
        nxt = repaired[end + 1]
        for idx in range(start, end + 1):
            repaired[idx] = interp_sample(prev, nxt, repaired[idx][0])
            replaced += 1
        applied.append((start, end))

    return repaired, replaced, applied


def resample_and_smooth(
    samples: Sequence[Tuple[int, float, float, float, float]],
    output_hz: float,
    smooth_window_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(samples) < 2:
        raise RuntimeError("Need at least two odometry samples in the selected navigation window.")

    ts = np.array([s[0] for s in samples], dtype=np.float64)
    x = np.array([s[1] for s in samples], dtype=np.float64)
    y = np.array([s[2] for s in samples], dtype=np.float64)
    z = np.array([s[3] for s in samples], dtype=np.float64)
    yaw = np.unwrap(np.array([s[4] for s in samples], dtype=np.float64))

    keep = np.concatenate(([True], np.diff(ts) > 0))
    ts, x, y, z, yaw = ts[keep], x[keep], y[keep], z[keep], yaw[keep]
    if ts.size < 2:
        raise RuntimeError("Odometry timestamps are not usable after removing duplicates.")

    dt_ns = int(round(1_000_000_000.0 / output_hz))
    out_ts = np.arange(int(ts[0]), int(ts[-1]) + 1, dt_ns, dtype=np.int64)
    if out_ts.size < 2:
        out_ts = np.array([int(ts[0]), int(ts[-1])], dtype=np.int64)

    out_x = np.interp(out_ts.astype(np.float64), ts, x)
    out_y = np.interp(out_ts.astype(np.float64), ts, y)
    out_z = np.interp(out_ts.astype(np.float64), ts, z)
    out_yaw = np.interp(out_ts.astype(np.float64), ts, yaw)

    window = max(1, int(round(smooth_window_s * output_hz)))
    out_x = moving_average(out_x, window)
    out_y = moving_average(out_y, window)
    out_z = moving_average(out_z, window)
    out_yaw = moving_average(out_yaw, window)
    return out_ts, out_x, out_y, out_z, out_yaw


def make_odom(timestamp_ns: int, frame_id: str, child_frame_id: str, x: float, y: float, z: float, yaw: float) -> Odometry:
    msg = Odometry()
    msg.header.stamp = ns_to_stamp(timestamp_ns)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)
    qw, qx, qy, qz = quat_from_yaw(float(yaw))
    msg.pose.pose.orientation.w = qw
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    return msg


def make_pose(timestamp_ns: int, frame_id: str, x: float, y: float, z: float, yaw: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = ns_to_stamp(timestamp_ns)
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    qw, qx, qy, qz = quat_from_yaw(float(yaw))
    msg.pose.orientation.w = qw
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    return msg


def make_tf(timestamp_ns: int, frame_id: str, child_frame_id: str, x: float, y: float, z: float, yaw: float) -> TFMessage:
    t = TransformStamped()
    t.header.stamp = ns_to_stamp(timestamp_ns)
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = float(x)
    t.transform.translation.y = float(y)
    t.transform.translation.z = float(z)
    qw, qx, qy, qz = quat_from_yaw(float(yaw))
    t.transform.rotation.w = qw
    t.transform.rotation.x = qx
    t.transform.rotation.y = qy
    t.transform.rotation.z = qz
    msg = TFMessage()
    msg.transforms.append(t)
    return msg


def compose_body_goal_to_map(
    base_x: float,
    base_y: float,
    base_z: float,
    base_yaw: float,
    body_x: float,
    body_y: float,
    body_z: float,
    body_yaw: float,
    visual_yaw_offset: float,
) -> Tuple[float, float, float, float]:
    visual_yaw = base_yaw + visual_yaw_offset
    cos_yaw = math.cos(visual_yaw)
    sin_yaw = math.sin(visual_yaw)
    return (
        base_x + cos_yaw * body_x - sin_yaw * body_y,
        base_y + sin_yaw * body_x + cos_yaw * body_y,
        base_z + body_z,
        base_yaw + body_yaw,
    )


def fill_marker_common(marker: Marker, timestamp_ns: int, frame_id: str, ns: str, marker_id: int, color: Tuple[float, float, float, float]):
    marker.header.stamp = ns_to_stamp(timestamp_ns)
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])


def make_pose_markers(
    timestamp_ns: int,
    frame_id: str,
    start_pose: Tuple[float, float, float, float],
    goal_pose: Optional[Tuple[float, float, float, float]],
    arrow_yaw_offset: float,
) -> MarkerArray:
    markers = MarkerArray()

    def add_sphere(ns: str, marker_id: int, pose: Tuple[float, float, float, float], color: Tuple[float, float, float, float]):
        x, y, z, _ = pose
        marker = Marker()
        fill_marker_common(marker, timestamp_ns, frame_id, ns, marker_id, color)
        marker.type = Marker.SPHERE
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z + 0.18)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.45
        markers.markers.append(marker)

    def add_arrow(ns: str, marker_id: int, pose: Tuple[float, float, float, float], color: Tuple[float, float, float, float]):
        x, y, z, yaw = pose
        qw, qx, qy, qz = quat_from_yaw(yaw + arrow_yaw_offset)
        marker = Marker()
        fill_marker_common(marker, timestamp_ns, frame_id, ns, marker_id, color)
        marker.type = Marker.ARROW
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z + 0.42)
        marker.pose.orientation.w = qw
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.scale.x = 0.9
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        markers.markers.append(marker)

    start_color = (0.0, 1.0, 0.1, 0.95)
    goal_color = (1.0, 0.05, 0.02, 0.95)
    add_sphere("demo_start", 0, start_pose, start_color)
    add_arrow("demo_start", 1, start_pose, start_color)
    if goal_pose is not None:
        add_sphere("demo_goal", 2, goal_pose, goal_color)
        add_arrow("demo_goal", 3, goal_pose, goal_color)
    return markers


def make_goal_pred_markers(
    timestamp_ns: int,
    frame_id: str,
    pred_pose: Optional[Tuple[float, float, float, float]],
    arrow_yaw_offset: float,
) -> MarkerArray:
    markers = MarkerArray()
    pred_color = (1.0, 0.55, 0.0, 0.95)

    def add_sphere(marker_id: int, pose: Tuple[float, float, float, float]):
        x, y, z, _ = pose
        marker = Marker()
        fill_marker_common(marker, timestamp_ns, frame_id, "goal_pred", marker_id, pred_color)
        marker.type = Marker.SPHERE
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z + 0.18)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.28
        marker.scale.y = 0.28
        marker.scale.z = 0.28
        markers.markers.append(marker)

    def add_arrow(marker_id: int, pose: Tuple[float, float, float, float]):
        x, y, z, yaw = pose
        qw, qx, qy, qz = quat_from_yaw(yaw + arrow_yaw_offset)
        marker = Marker()
        fill_marker_common(marker, timestamp_ns, frame_id, "goal_pred", marker_id, pred_color)
        marker.type = Marker.ARROW
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z + 0.36)
        marker.pose.orientation.w = qw
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.scale.x = 0.55
        marker.scale.y = 0.11
        marker.scale.z = 0.11
        markers.markers.append(marker)

    if pred_pose is not None:
        add_sphere(0, pred_pose)
        add_arrow(1, pred_pose)
    return markers


def create_writer(output_dir: FsPath, force: bool) -> SequentialWriter:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(output_dir), storage_id="sqlite3"), ConverterOptions("", ""))
    for name, msg_type in [
        ("/odom_smooth", "nav_msgs/msg/Odometry"),
        ("/odom_path_smooth", "nav_msgs/msg/Path"),
        ("/goal_pred_path_smooth", "nav_msgs/msg/Path"),
        ("/tf", "tf2_msgs/msg/TFMessage"),
        ("/demo_markers", "visualization_msgs/msg/MarkerArray"),
        ("/goal_pred_markers", "visualization_msgs/msg/MarkerArray"),
    ]:
        writer.create_topic(
            TopicMetadata(
                name=name,
                type=msg_type,
                serialization_format="cdr",
                offered_qos_profiles="",
            )
        )
    return writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bag", type=FsPath, help="Bag from offline localization; must contain /odom or /Odometry.")
    parser.add_argument("output_bag", type=FsPath, nargs="?", help="Default: <input>_traj_smooth")
    parser.add_argument("--odom-topic", default="", help="Default: prefer /odom, then /Odometry.")
    parser.add_argument("--nav-topic", default="auto", help="Topic used to infer navigation start/end. Default auto tries goal_pred/cmd_high/goal_error.")
    parser.add_argument("--start-sec", type=float, default=None, help="Manual crop start relative to bag start.")
    parser.add_argument("--end-sec", type=float, default=None, help="Manual crop end relative to bag start.")
    parser.add_argument("--output-hz", type=float, default=20.0, help="Playback trajectory frequency.")
    parser.add_argument("--path-hz", type=float, default=2.0, help="How often to publish cumulative /odom_path_smooth. 0 publishes final path only.")
    parser.add_argument("--smooth-window", type=float, default=0.8, help="Moving-average window in seconds.")
    parser.add_argument("--max-step-dist", type=float, default=0.0, help="Interpolate short odom spikes whose xy jump exceeds this many meters. 0 disables.")
    parser.add_argument("--max-step-yaw", type=float, default=0.0, help="Interpolate short odom spikes whose yaw jump exceeds this many radians. 0 disables.")
    parser.add_argument("--max-outlier-run", type=int, default=0, help="Truncate trajectory after this many consecutive unrepaired odom jumps. 0 disables truncation.")
    parser.add_argument("--max-interp-speed", type=float, default=1.5, help="Only interpolate a spike when the bridge between surrounding good samples is below this m/s. 0 disables this check.")
    parser.add_argument("--repair-index-range", action="append", default=[], help="Manually replace cropped odom sample indices by interpolation, e.g. 74:77. Can be repeated or comma-separated.")
    parser.add_argument("--goal-rel", nargs=3, type=float, metavar=("FORWARD", "LEFT", "YAW"), help="Goal pose relative to the visual start heading, in meters/radians.")
    parser.add_argument("--marker-yaw-offset", type=float, default=math.pi / 2.0, help="Yaw offset for visual arrows and goal-relative forward direction. Default pi/2 makes FORWARD align with the displayed front direction.")
    parser.add_argument("--frame-id", default="", help="Override output frame_id; default from odom header.")
    parser.add_argument("--child-frame-id", default="base_link_smooth", help="Output child frame id.")
    parser.add_argument("--force", action="store_true", help="Overwrite output bag directory.")
    args = parser.parse_args()

    input_bag = args.input_bag.expanduser().resolve()
    output_bag = (args.output_bag or input_bag.with_name(input_bag.name + "_traj_smooth")).expanduser().resolve()
    bag_dirs = find_bag_dirs(input_bag)
    if not bag_dirs:
        raise RuntimeError(f"No rosbag metadata.yaml found in {input_bag}")

    topic_types = collect_topic_types(bag_dirs)
    odom_topic = args.odom_topic
    if not odom_topic:
        if "/odom" in topic_types:
            odom_topic = "/odom"
        elif "/Odometry" in topic_types:
            odom_topic = "/Odometry"
        else:
            raise RuntimeError("No /odom or /Odometry topic found. Run offline localization first, then use its output bag.")
    if odom_topic not in topic_types:
        raise RuntimeError(f"Requested odom topic not found: {odom_topic}")

    wanted = {odom_topic}
    goal_pred_topic = "/nav/goal_pred_map"
    goal_actual_topic = "/nav/goal_actual_map"
    if args.nav_topic == "auto":
        crop_topics = [
            topic for topic in [
                "/nav/cmd_high",
                "/nav/goal_error_body",
                goal_pred_topic,
            ]
            if topic in topic_types
        ]
    else:
        crop_topics = [args.nav_topic] if args.nav_topic in topic_types else []
    wanted.update(crop_topics)
    if goal_pred_topic in topic_types:
        wanted.add(goal_pred_topic)
    if goal_actual_topic in topic_types:
        wanted.add(goal_actual_topic)

    bag_start_ns: Optional[int] = None
    nav_times: List[int] = []
    odom_samples: List[Tuple[int, float, float, float, float]] = []
    goal_pred_samples: List[Tuple[int, str, float, float, float, float]] = []
    goal_actual_samples: List[Tuple[int, str, float, float, float, float]] = []
    frame_id = args.frame_id

    for timestamp, topic, msg in iter_bag_messages(bag_dirs, wanted):
        if bag_start_ns is None:
            bag_start_ns = timestamp
        if topic in crop_topics:
            nav_times.append(timestamp)
        if topic in (goal_pred_topic, goal_actual_topic):
            # Use rosbag record time for offline demo alignment. In bags produced
            # by replay/localization, message header stamps can come from a
            # different clock domain than /odom and the bag timeline.
            stamp_ns = timestamp
            pose = msg.pose
            row = (
                stamp_ns,
                msg.header.frame_id,
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                yaw_from_quat(pose.orientation),
            )
            if topic == goal_pred_topic:
                goal_pred_samples.append(row)
            else:
                goal_actual_samples.append(row)
        elif topic == odom_topic:
            pose = msg.pose.pose
            if not frame_id:
                frame_id = msg.header.frame_id or "map"
            odom_samples.append(
                (
                    timestamp,
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    yaw_from_quat(pose.orientation),
                )
            )

    if bag_start_ns is None:
        raise RuntimeError("Input bag has no messages.")
    if len(odom_samples) < 2:
        raise RuntimeError(f"Not enough odometry samples on {odom_topic}.")
    if not frame_id:
        frame_id = "map"

    crop_start_ns = None
    crop_end_ns = None
    if args.start_sec is not None:
        crop_start_ns = bag_start_ns + int(round(args.start_sec * 1e9))
    if args.end_sec is not None:
        crop_end_ns = bag_start_ns + int(round(args.end_sec * 1e9))
    if crop_start_ns is None or crop_end_ns is None:
        if nav_times:
            crop_start_ns = nav_times[0] if crop_start_ns is None else crop_start_ns
            crop_end_ns = nav_times[-1] if crop_end_ns is None else crop_end_ns
        else:
            crop_start_ns = odom_samples[0][0] if crop_start_ns is None else crop_start_ns
            crop_end_ns = odom_samples[-1][0] if crop_end_ns is None else crop_end_ns

    selected = [s for s in odom_samples if crop_start_ns <= s[0] <= crop_end_ns]
    if len(selected) < 2:
        raise RuntimeError(
            "Navigation crop contains fewer than two odom samples. "
            "Check --nav-topic, --start-sec, or --end-sec."
        )
    selected_raw_count = len(selected)
    manual_ranges = parse_index_ranges(args.repair_index_range)
    selected, manual_replaced, applied_manual_ranges = interpolate_index_ranges(selected, manual_ranges)
    selected, replaced_outliers, truncated_on_outlier_run = interpolate_short_outliers(
        selected,
        max_step_dist=max(0.0, args.max_step_dist),
        max_step_yaw=max(0.0, args.max_step_yaw),
        max_outlier_run=max(0, args.max_outlier_run),
        max_interp_speed=max(0.0, args.max_interp_speed),
    )
    if len(selected) < 2:
        raise RuntimeError("Trajectory became too short after outlier rejection.")

    selected_goal_pred = [s for s in goal_pred_samples if crop_start_ns <= s[0] <= crop_end_ns]
    selected_goal_actual = [s for s in goal_actual_samples if crop_start_ns <= s[0] <= crop_end_ns]

    out_ts, out_x, out_y, out_z, out_yaw = resample_and_smooth(
        selected,
        output_hz=max(1e-3, args.output_hz),
        smooth_window_s=max(0.0, args.smooth_window),
    )

    writer = create_writer(output_bag, args.force)
    path_msg = PathMsg()
    path_msg.header.frame_id = frame_id
    pred_path_msg = PathMsg()
    pred_path_msg.header.frame_id = frame_id
    start_pose = (float(out_x[0]), float(out_y[0]), float(out_z[0]), float(out_yaw[0]))
    goal_pose = None
    if args.goal_rel is not None:
        rel_forward, rel_left, rel_yaw = args.goal_rel
        start_x, start_y, start_z, start_yaw = start_pose
        visual_yaw = start_yaw + args.marker_yaw_offset
        cos_yaw = math.cos(visual_yaw)
        sin_yaw = math.sin(visual_yaw)
        goal_pose = (
            start_x + cos_yaw * rel_forward - sin_yaw * rel_left,
            start_y + sin_yaw * rel_forward + cos_yaw * rel_left,
            start_z,
            start_yaw + rel_yaw,
        )

    def odom_at(timestamp_ns: int) -> Optional[Tuple[float, float, float, float, float]]:
        if timestamp_ns < int(out_ts[0]) or timestamp_ns > int(out_ts[-1]):
            return None
        t_float = float(timestamp_ns)
        return (
            float(np.interp(t_float, out_ts.astype(np.float64), out_x)),
            float(np.interp(t_float, out_ts.astype(np.float64), out_y)),
            float(np.interp(t_float, out_ts.astype(np.float64), out_z)),
            float(np.interp(t_float, out_ts.astype(np.float64), out_yaw)),
            t_float,
        )

    def goal_to_map(sample: Tuple[int, str, float, float, float, float]) -> Optional[Tuple[int, float, float, float, float]]:
        t_ns, sample_frame, gx, gy, gz, gyaw = sample
        if sample_frame == frame_id or sample_frame == "map":
            return (t_ns, gx, gy, gz, gyaw)
        if sample_frame in ("base_link", "base_link_smooth", ""):
            od = odom_at(t_ns)
            if od is None:
                return None
            bx, by, bz, byaw, _ = od
            mx, my, mz, myaw = compose_body_goal_to_map(
                bx, by, bz, byaw, gx, gy, gz, gyaw, args.marker_yaw_offset
            )
            return (t_ns, mx, my, mz, myaw)
        return None

    pred_map_samples = [mapped for mapped in (goal_to_map(s) for s in selected_goal_pred) if mapped is not None]
    actual_map_samples = [mapped for mapped in (goal_to_map(s) for s in selected_goal_actual) if mapped is not None]

    pred_out = None
    if len(pred_map_samples) >= 2:
        pred_ts = np.array([s[0] for s in pred_map_samples], dtype=np.float64)
        pred_x = np.array([s[1] for s in pred_map_samples], dtype=np.float64)
        pred_y = np.array([s[2] for s in pred_map_samples], dtype=np.float64)
        pred_z = np.array([s[3] for s in pred_map_samples], dtype=np.float64)
        pred_yaw = np.unwrap(np.array([s[4] for s in pred_map_samples], dtype=np.float64))
        pred_keep = np.concatenate(([True], np.diff(pred_ts) > 0))
        pred_ts, pred_x, pred_y, pred_z, pred_yaw = pred_ts[pred_keep], pred_x[pred_keep], pred_y[pred_keep], pred_z[pred_keep], pred_yaw[pred_keep]
        if pred_ts.size >= 2:
            pred_out = (
                np.interp(out_ts.astype(np.float64), pred_ts, pred_x),
                np.interp(out_ts.astype(np.float64), pred_ts, pred_y),
                np.interp(out_ts.astype(np.float64), pred_ts, pred_z),
                np.interp(out_ts.astype(np.float64), pred_ts, pred_yaw),
            )

    goal_error_xy: List[float] = []
    goal_error_yaw: List[float] = []
    if pred_map_samples and actual_map_samples:
        actual_times = [s[0] for s in actual_map_samples]
        max_sync_ns = int(0.2 * 1e9)
        for pred in pred_map_samples:
            pred_t, pred_x, pred_y, _, pred_yaw = pred
            idx = int(np.searchsorted(actual_times, pred_t))
            candidates = []
            if 0 <= idx < len(actual_map_samples):
                candidates.append(idx)
            if 0 <= idx - 1 < len(actual_map_samples):
                candidates.append(idx - 1)
            if not candidates:
                continue
            best_idx = min(candidates, key=lambda i: abs(actual_map_samples[i][0] - pred_t))
            actual = actual_map_samples[best_idx]
            if abs(actual[0] - pred_t) > max_sync_ns:
                continue
            goal_error_xy.append(math.hypot(pred_x - actual[1], pred_y - actual[2]))
            goal_error_yaw.append(abs(wrap_angle_delta(pred_yaw - actual[4])))

    path_period_ns = int(round(1_000_000_000.0 / args.path_hz)) if args.path_hz > 0.0 else 0
    next_path_write_ns = int(out_ts[0])
    path_writes = 0

    for idx, (ts, x, y, z, yaw) in enumerate(zip(out_ts, out_x, out_y, out_z, out_yaw)):
        ts_int = int(ts)
        odom = make_odom(ts_int, frame_id, args.child_frame_id, x, y, z, yaw)
        pose = make_pose(ts_int, frame_id, x, y, z, yaw)
        path_msg.header.stamp = ns_to_stamp(ts_int)
        path_msg.poses.append(pose)
        pred_pose_tuple = None
        if pred_out is not None:
            pred_pose_tuple = (
                float(pred_out[0][idx]),
                float(pred_out[1][idx]),
                float(pred_out[2][idx]),
                float(pred_out[3][idx]),
            )
            pred_pose_msg = make_pose(ts_int, frame_id, *pred_pose_tuple)
            pred_path_msg.header.stamp = ns_to_stamp(ts_int)
            pred_path_msg.poses.append(pred_pose_msg)
        tf_msg = make_tf(ts_int, frame_id, args.child_frame_id, x, y, z, yaw)
        writer.write("/odom_smooth", serialize_message(odom), ts_int)
        writer.write("/tf", serialize_message(tf_msg), ts_int)
        writer.write("/demo_markers", serialize_message(make_pose_markers(ts_int, frame_id, start_pose, goal_pose, args.marker_yaw_offset)), ts_int)
        writer.write("/goal_pred_markers", serialize_message(make_goal_pred_markers(ts_int, frame_id, pred_pose_tuple, args.marker_yaw_offset)), ts_int)
        is_last = idx == (len(out_ts) - 1)
        should_write_path = is_last
        if path_period_ns > 0 and ts_int >= next_path_write_ns:
            should_write_path = True
            while next_path_write_ns <= ts_int:
                next_path_write_ns += path_period_ns
        if should_write_path:
            writer.write("/odom_path_smooth", serialize_message(path_msg), ts_int)
            if pred_out is not None:
                writer.write("/goal_pred_path_smooth", serialize_message(pred_path_msg), ts_int)
            path_writes += 1

    duration_s = (out_ts[-1] - out_ts[0]) / 1e9
    print(f"[smooth-traj] input: {input_bag}")
    print(f"[smooth-traj] output: {output_bag}")
    print(f"[smooth-traj] odom_topic={odom_topic} nav_topic={','.join(crop_topics) if nav_times else 'none'}")
    print(f"[smooth-traj] crop: {(crop_start_ns - bag_start_ns)/1e9:.3f}s -> {(crop_end_ns - bag_start_ns)/1e9:.3f}s")
    print(f"[smooth-traj] odom samples: raw={len(odom_samples)} selected={selected_raw_count} kept={len(selected)} output={len(out_ts)}")
    print(f"[smooth-traj] manual repairs: {manual_replaced} ranges={','.join(f'{a}:{b}' for a, b in applied_manual_ranges) if applied_manual_ranges else 'none'}")
    print(f"[smooth-traj] spawn_map: x={start_pose[0]:.3f} y={start_pose[1]:.3f} yaw={start_pose[3] + args.marker_yaw_offset:.3f}")
    print(f"[smooth-traj] goal_pred samples: raw={len(goal_pred_samples)} selected={len(selected_goal_pred)} map_used={len(pred_map_samples)} path={'yes' if pred_out is not None else 'no'}")
    if goal_pose is not None:
        print(f"[smooth-traj] goal_rel: forward={args.goal_rel[0]:.3f} left={args.goal_rel[1]:.3f} yaw={args.goal_rel[2]:.3f}")
        print(f"[smooth-traj] goal_map: x={goal_pose[0]:.3f} y={goal_pose[1]:.3f} yaw={goal_pose[3]:.3f}")
    if goal_error_xy:
        e = np.array(goal_error_xy, dtype=np.float64)
        eyaw = np.array(goal_error_yaw, dtype=np.float64)
        print(f"[smooth-traj] goal_pred_error_xy: count={e.size} mean={float(e.mean()):.3f} p95={float(np.percentile(e, 95)):.3f} max={float(e.max()):.3f}")
        print(f"[smooth-traj] goal_pred_error_yaw_abs: mean={float(eyaw.mean()):.3f} p95={float(np.percentile(eyaw, 95)):.3f} max={float(eyaw.max()):.3f}")
    print(f"[smooth-traj] outliers interpolated: {replaced_outliers} truncated={1 if truncated_on_outlier_run else 0} (max_step_dist={args.max_step_dist:.3f}m max_step_yaw={args.max_step_yaw:.3f}rad max_outlier_run={args.max_outlier_run} max_interp_speed={args.max_interp_speed:.3f}m/s)")
    print(f"[smooth-traj] output duration={duration_s:.3f}s odom_hz={args.output_hz:.2f} path_hz={args.path_hz:.2f} path_writes={path_writes} smooth_window={args.smooth_window:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
