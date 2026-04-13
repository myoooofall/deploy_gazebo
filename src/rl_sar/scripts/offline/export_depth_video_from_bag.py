#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import time
from typing import Optional

import cv2
import numpy as np
import rosbag2_py
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image


def colormap_from_name(name: str) -> int:
    mapping = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    if name not in mapping:
        raise ValueError(f"unsupported colormap: {name}")
    return mapping[name]


def try_open_reader(uri: str, storage_id: str) -> rosbag2_py.SequentialReader:
    if storage_id:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader.open(storage_options, converter_options)
        return reader

    last_exc = None
    for sid in ("sqlite3", "mcap"):
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=sid)
            converter_options = rosbag2_py.ConverterOptions("", "")
            reader.open(storage_options, converter_options)
            return reader
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"failed to open bag '{uri}' as sqlite3/mcap: {last_exc}")


class DepthFrameConverter:
    def __init__(self, min_percentile: float, max_percentile: float, colormap: str):
        self._bridge = CvBridge()
        self._min_p = min_percentile
        self._max_p = max_percentile
        self._colormap = colormap

    def normalize_to_u8(self, frame: np.ndarray) -> np.ndarray:
        arr = frame.astype(np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.uint8)

        finite_vals = arr[finite]
        arr_min = float(np.min(finite_vals))
        arr_max = float(np.max(finite_vals))
        # Auto-detect normalized depth convention, e.g. depth_norm = depth/5 - 1 in [-1, 0].
        # In this case, 0 is a valid "far" value and must NOT be dropped.
        normalized_neg_to_zero = (arr_min < -0.05) and (arr_max <= 0.05)
        if normalized_neg_to_zero:
            valid = finite
        else:
            # For raw depth images, exact zeros are often invalid fill values.
            non_zero_valid = np.logical_and(finite, arr != 0.0)
            valid = non_zero_valid if np.any(non_zero_valid) else finite
        if not np.any(valid):
            return np.zeros(arr.shape, dtype=np.uint8)

        lo = np.percentile(arr[valid], self._min_p)
        hi = np.percentile(arr[valid], self._max_p)
        if hi <= lo:
            hi = lo + 1e-6
        arr = np.clip(arr, lo, hi)
        out = ((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)
        out[~valid] = 0
        return out

    def to_bgr_u8(self, img_msg: Image) -> np.ndarray:
        frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        arr = np.asarray(frame)

        if arr.ndim == 2:
            if arr.dtype == np.uint8:
                gray = arr
            else:
                gray = self.normalize_to_u8(arr)
            if self._colormap == "none":
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return cv2.applyColorMap(gray, colormap_from_name(self._colormap))

        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                arr = self.normalize_to_u8(arr)
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if arr.shape[2] == 3:
                return arr

        raise RuntimeError(f"unsupported image shape/dtype: shape={arr.shape}, dtype={arr.dtype}")


class DepthVideoRecorder(Node):
    def __init__(
        self,
        topic: str,
        output_path: str,
        fps: float,
        converter: DepthFrameConverter,
    ):
        super().__init__("depth_video_recorder")
        self._topic = topic
        self._output_path = output_path
        self._fps = fps
        self._converter = converter
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames = 0
        self._last_frame_ts: Optional[float] = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Image, topic, self._on_image, qos)
        self.get_logger().info(f"subscribed to topic: {topic}")

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def last_frame_ts(self) -> Optional[float]:
        return self._last_frame_ts

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _init_writer(self, width: int, height: int):
        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {self._output_path}")
        self.get_logger().info(
            f"writing video: {self._output_path} ({width}x{height}, {self._fps:.2f} fps)"
        )

    def _on_image(self, msg: Image):
        bgr = self._converter.to_bgr_u8(msg)
        h, w = bgr.shape[:2]
        if self._writer is None:
            self._init_writer(w, h)
        self._writer.write(bgr)
        self._frames += 1
        self._last_frame_ts = time.monotonic()
        if self._frames % 100 == 0:
            self.get_logger().info(f"recorded {self._frames} frames")


def init_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")
    return writer


def export_depth_video_direct(
    bag: str,
    topic: str,
    out: str,
    fps: float,
    converter: DepthFrameConverter,
    storage_id: str,
    max_frames: int,
):
    reader = try_open_reader(bag, storage_id)
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    if topic not in type_map:
        image_topics = [name for name, typ in type_map.items() if typ == "sensor_msgs/msg/Image"]
        raise RuntimeError(
            f"topic not found in bag: {topic}. image topics in bag: {image_topics}"
        )

    msg_cls = get_message(type_map[topic])
    writer: Optional[cv2.VideoWriter] = None
    frames = 0

    try:
        while reader.has_next():
            topic_name, data, _ = reader.read_next()
            if topic_name != topic:
                continue
            msg = deserialize_message(data, msg_cls)
            bgr = converter.to_bgr_u8(msg)
            h, w = bgr.shape[:2]
            if writer is None:
                writer = init_video_writer(out, fps, w, h)
                print(f"[depth-video] writing video: {out} ({w}x{h}, {fps:.2f} fps)")
            writer.write(bgr)
            frames += 1
            if frames % 100 == 0:
                print(f"[depth-video] recorded {frames} frames")
            if max_frames > 0 and frames >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()

    if frames <= 0:
        raise RuntimeError(f"export failed: no frame written, topic={topic}")
    print(f"[depth-video] done, wrote {frames} frames to {out}")


def export_depth_video_play(
    bag: str,
    topic: str,
    out: str,
    fps: float,
    converter: DepthFrameConverter,
    play_rate: float,
    post_roll_sec: float,
    wait_first_frame_sec: float,
    qos_overrides_path: str,
):
    env = os.environ.copy()
    if not env.get("ROS_LOG_DIR"):
        env["ROS_LOG_DIR"] = "/tmp/ros_logs"
    os.makedirs(env["ROS_LOG_DIR"], exist_ok=True)

    play_cmd = [
        "ros2",
        "bag",
        "play",
        bag,
        "--clock",
        "--rate",
        str(play_rate),
        "--topics",
        topic,
    ]
    if qos_overrides_path:
        play_cmd.extend(["--qos-profile-overrides-path", qos_overrides_path])

    print(f"[depth-video] play cmd: {' '.join(play_cmd)}")

    rclpy.init()
    node = DepthVideoRecorder(
        topic=topic,
        output_path=out,
        fps=fps,
        converter=converter,
    )

    player = None
    t0 = time.monotonic()
    try:
        player = subprocess.Popen(play_cmd, env=env)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if node.frames == 0 and (now - t0) > wait_first_frame_sec:
                raise RuntimeError(
                    f"no frames received from topic {topic} within {wait_first_frame_sec:.1f}s"
                )

            if player.poll() is not None:
                last = node.last_frame_ts
                if last is None or (now - last) > post_roll_sec:
                    break
    finally:
        if player is not None and player.poll() is None:
            player.send_signal(signal.SIGINT)
            try:
                player.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                player.kill()
        node.close()
        node.destroy_node()
        rclpy.shutdown()

    if node.frames <= 0:
        raise RuntimeError(f"export failed: no frame written, topic={topic}")
    print(f"[depth-video] done, wrote {node.frames} frames to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Export depth topic from rosbag2 to MP4 without ffmpeg"
    )
    parser.add_argument("bag", help="rosbag2 directory path")
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "play"],
        help="direct: read bag with rosbag2_py (recommended); play: use ros2 bag play + subscription",
    )
    parser.add_argument(
        "--topic",
        default="/camera/depth/processed_norm",
        help="depth image topic to export",
    )
    parser.add_argument(
        "--out",
        default="depth_preview.mp4",
        help="output video path (default: ./depth_preview.mp4)",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="output video fps")
    parser.add_argument("--play-rate", type=float, default=1.0, help="ros2 bag play --rate")
    parser.add_argument(
        "--storage-id",
        default="",
        help="force rosbag storage id in direct mode: sqlite3 or mcap",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="max frames to export in direct mode; 0 means all",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=2.0,
        help="lower percentile for depth normalization",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=98.0,
        help="upper percentile for depth normalization",
    )
    parser.add_argument(
        "--colormap",
        default="turbo",
        choices=["none", "turbo", "jet", "inferno", "magma", "viridis"],
        help="colormap for single-channel depth image",
    )
    parser.add_argument(
        "--post-roll-sec",
        type=float,
        default=1.0,
        help="wait time after bag play exits and no new frames",
    )
    parser.add_argument(
        "--wait-first-frame-sec",
        type=float,
        default=20.0,
        help="max wait time for first frame before exiting",
    )
    parser.add_argument(
        "--qos-overrides-path",
        default="",
        help="optional qos overrides yaml for ros2 bag play",
    )
    args = parser.parse_args()

    bag = os.path.abspath(args.bag)
    out = os.path.abspath(args.out)
    if not os.path.isdir(bag):
        raise RuntimeError(f"bag path not found: {bag}")
    if args.fps <= 0.0:
        raise RuntimeError("--fps must be > 0")
    if args.min_percentile < 0.0 or args.max_percentile > 100.0 or args.min_percentile >= args.max_percentile:
        raise RuntimeError("percentiles must satisfy: 0 <= min < max <= 100")
    if args.max_frames < 0:
        raise RuntimeError("--max-frames must be >= 0")

    print(f"[depth-video] bag: {bag}")
    print(f"[depth-video] mode: {args.mode}")
    print(f"[depth-video] topic: {args.topic}")
    print(f"[depth-video] out: {out}")
    converter = DepthFrameConverter(
        min_percentile=args.min_percentile,
        max_percentile=args.max_percentile,
        colormap=args.colormap,
    )
    if args.mode == "direct":
        export_depth_video_direct(
            bag=bag,
            topic=args.topic,
            out=out,
            fps=args.fps,
            converter=converter,
            storage_id=args.storage_id,
            max_frames=args.max_frames,
        )
    else:
        export_depth_video_play(
            bag=bag,
            topic=args.topic,
            out=out,
            fps=args.fps,
            converter=converter,
            play_rate=args.play_rate,
            post_roll_sec=args.post_roll_sec,
            wait_first_frame_sec=args.wait_first_frame_sec,
            qos_overrides_path=args.qos_overrides_path,
        )


if __name__ == "__main__":
    main()
