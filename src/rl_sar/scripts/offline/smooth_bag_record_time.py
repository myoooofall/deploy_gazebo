#!/usr/bin/env python3
"""Create a rosbag2 copy with smoother storage timestamps.

This script rewrites only rosbag2 SQLite `messages.timestamp` values. It does
not modify serialized message contents, so `header.stamp` remains the original
sensor/navigation timestamp used by localization.
"""

import argparse
import fnmatch
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is present on normal ROS installs.
    yaml = None


HEADER_FIRST_TYPES = {
    "sensor_msgs/msg/Imu",
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/LaserScan",
    "nav_msgs/msg/Odometry",
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/PoseWithCovarianceStamped",
    "geometry_msgs/msg/PointStamped",
    "geometry_msgs/msg/Vector3Stamped",
    "geometry_msgs/msg/QuaternionStamped",
    "geometry_msgs/msg/TwistStamped",
    "geometry_msgs/msg/WrenchStamped",
}

DEFAULT_HEADER_TOPICS = {
    "/rslidar_points",
    "/imu/data",
}


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def parse_header_stamp_ns(data: bytes) -> Optional[int]:
    if len(data) < 12:
        return None

    # ROS2 CDR payload starts with a 4-byte encapsulation header. Header-first
    # messages then store builtin_interfaces/Time as int32 sec + uint32 nanosec.
    for endian in ("<", ">"):
        try:
            sec, nsec = struct.unpack_from(f"{endian}iI", data, 4)
        except struct.error:
            continue
        if 0 <= nsec < 1_000_000_000 and -4_000_000_000 < sec < 4_000_000_000:
            stamp_ns = int(sec) * 1_000_000_000 + int(nsec)
            if stamp_ns == 0:
                return None
            return stamp_ns
    return None


def find_bag_dirs(path: Path) -> List[Path]:
    if (path / "metadata.yaml").is_file():
        return [path]
    bag_dirs = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "metadata.yaml").is_file():
            bag_dirs.append(child)
    return bag_dirs


def find_db_files(bag_dir: Path) -> List[Path]:
    return sorted(p for p in bag_dir.glob("*.db3") if p.is_file())


def load_topics(conn: sqlite3.Connection) -> Dict[int, Tuple[str, str]]:
    rows = conn.execute("SELECT id, name, type FROM topics").fetchall()
    return {int(topic_id): (name, msg_type) for topic_id, name, msg_type in rows}


def iter_messages(conn: sqlite3.Connection):
    yield from conn.execute("SELECT id, topic_id, timestamp, data FROM messages ORDER BY timestamp, id")


def should_use_header(topic_name: str, msg_type: str, patterns: Iterable[str]) -> bool:
    if msg_type not in HEADER_FIRST_TYPES:
        return False
    return any(fnmatch.fnmatch(topic_name, pattern) for pattern in patterns)


def collect_global_reference(
    bag_dirs: List[Path],
    header_topic_patterns: Iterable[str],
) -> Tuple[int, int]:
    min_record_ts: Optional[int] = None
    min_header_ts: Optional[int] = None

    for bag_dir in bag_dirs:
        for db_file in find_db_files(bag_dir):
            with sqlite3.connect(str(db_file)) as conn:
                topics = load_topics(conn)
                for _msg_id, topic_id, record_ts, data in iter_messages(conn):
                    min_record_ts = int(record_ts) if min_record_ts is None else min(min_record_ts, int(record_ts))
                    topic_name, msg_type = topics[int(topic_id)]
                    if not should_use_header(topic_name, msg_type, header_topic_patterns):
                        continue
                    header_ts = parse_header_stamp_ns(bytes(data))
                    if header_ts is None:
                        continue
                    min_header_ts = header_ts if min_header_ts is None else min(min_header_ts, header_ts)

    if min_record_ts is None:
        raise RuntimeError("No messages found in input bag.")
    if min_header_ts is None:
        raise RuntimeError("No usable header stamps found. Check topic names or message types.")
    return min_record_ts, min_header_ts


def summarize_topic(timestamps: List[int]) -> str:
    if len(timestamps) < 2:
        return "n<2"
    gaps_ms = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    duration_s = max((timestamps[-1] - timestamps[0]) / 1e9, 1e-9)
    hz = (len(timestamps) - 1) / duration_s
    return (
        f"n={len(timestamps)} hz={hz:.2f} "
        f"p50={percentile(gaps_ms, 50):.3f}ms "
        f"p99={percentile(gaps_ms, 99):.3f}ms "
        f"max={max(gaps_ms):.3f}ms"
    )


def rewrite_bag_dir(
    bag_dir: Path,
    global_record_start: int,
    global_header_start: int,
    header_topic_patterns: Iterable[str],
) -> Dict[str, Tuple[List[int], List[int]]]:
    topic_times: Dict[str, Tuple[List[int], List[int]]] = {}
    bag_min_ts: Optional[int] = None
    bag_max_ts: Optional[int] = None

    for db_file in find_db_files(bag_dir):
        conn = sqlite3.connect(str(db_file))
        try:
            topics = load_topics(conn)
            updates = []
            rows = list(iter_messages(conn))
            for msg_id, topic_id, record_ts, data in rows:
                topic_name, msg_type = topics[int(topic_id)]
                old_ts = int(record_ts)
                new_ts = old_ts
                if should_use_header(topic_name, msg_type, header_topic_patterns):
                    header_ts = parse_header_stamp_ns(bytes(data))
                    if header_ts is not None:
                        new_ts = global_record_start + (header_ts - global_header_start)

                bag_min_ts = new_ts if bag_min_ts is None else min(bag_min_ts, new_ts)
                bag_max_ts = new_ts if bag_max_ts is None else max(bag_max_ts, new_ts)
                old_list, new_list = topic_times.setdefault(topic_name, ([], []))
                old_list.append(old_ts)
                new_list.append(new_ts)
                if new_ts != old_ts:
                    updates.append((int(new_ts), int(msg_id)))

            if updates:
                conn.executemany("UPDATE messages SET timestamp=? WHERE id=?", updates)
                conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
            except sqlite3.DatabaseError:
                pass
        finally:
            conn.close()

        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_file) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    if bag_min_ts is not None and bag_max_ts is not None:
        update_metadata_yaml(bag_dir / "metadata.yaml", bag_min_ts, bag_max_ts - bag_min_ts)

    return topic_times


def update_metadata_yaml(metadata_path: Path, start_ns: int, duration_ns: int) -> None:
    text = metadata_path.read_text()
    lines = text.splitlines()
    out = []
    in_duration = False
    in_starting_time = False
    for line in lines:
        stripped = line.strip()
        if stripped == "duration:":
            in_duration = True
            in_starting_time = False
            out.append(line)
            continue
        if stripped == "starting_time:":
            in_duration = False
            in_starting_time = True
            out.append(line)
            continue
        if in_duration and stripped.startswith("nanoseconds:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}nanoseconds: {int(duration_ns)}")
            in_duration = False
            continue
        if in_starting_time and stripped.startswith("nanoseconds_since_epoch:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}nanoseconds_since_epoch: {int(start_ns)}")
            in_starting_time = False
            continue
        out.append(line)
    metadata_path.write_text("\n".join(out) + "\n")


def copy_input(input_path: Path, output_path: Path, force: bool) -> None:
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_path}")
        shutil.rmtree(output_path)
    shutil.copytree(input_path, output_path)


def merge_bag_dirs(bag_dirs: List[Path], merged_output: Path, force: bool) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write merged metadata.yaml")
    if merged_output.exists():
        if not force:
            raise FileExistsError(f"Merged output already exists: {merged_output}")
        shutil.rmtree(merged_output)
    merged_output.mkdir(parents=True)

    merged_db = merged_output / "merged_0.db3"
    out = sqlite3.connect(str(merged_db))
    topic_map: Dict[Tuple[str, str, str, str], int] = {}
    topic_counts: Dict[int, int] = {}
    topic_meta_by_id: Dict[int, Tuple[str, str, str, str]] = {}
    all_messages = []
    try:
        out.execute(
            "CREATE TABLE topics("
            "id INTEGER PRIMARY KEY,"
            "name TEXT NOT NULL,"
            "type TEXT NOT NULL,"
            "serialization_format TEXT NOT NULL,"
            "offered_qos_profiles TEXT NOT NULL)"
        )
        out.execute(
            "CREATE TABLE messages("
            "id INTEGER PRIMARY KEY,"
            "topic_id INTEGER NOT NULL,"
            "timestamp INTEGER NOT NULL,"
            "data BLOB NOT NULL)"
        )

        next_topic_id = 1
        for bag_dir in bag_dirs:
            for db_file in find_db_files(bag_dir):
                with sqlite3.connect(str(db_file)) as conn:
                    in_topics = {
                        int(row[0]): (row[1], row[2], row[3], row[4])
                        for row in conn.execute(
                            "SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics"
                        )
                    }
                    local_to_merged: Dict[int, int] = {}
                    for local_id, meta in in_topics.items():
                        if meta not in topic_map:
                            topic_map[meta] = next_topic_id
                            topic_meta_by_id[next_topic_id] = meta
                            out.execute(
                                "INSERT INTO topics(id, name, type, serialization_format, offered_qos_profiles) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (next_topic_id, *meta),
                            )
                            next_topic_id += 1
                        local_to_merged[local_id] = topic_map[meta]

                    for _msg_id, topic_id, timestamp, data in conn.execute(
                        "SELECT id, topic_id, timestamp, data FROM messages"
                    ):
                        merged_topic_id = local_to_merged[int(topic_id)]
                        all_messages.append((int(timestamp), merged_topic_id, bytes(data)))
                        topic_counts[merged_topic_id] = topic_counts.get(merged_topic_id, 0) + 1

        all_messages.sort(key=lambda item: item[0])
        out.executemany(
            "INSERT INTO messages(id, topic_id, timestamp, data) VALUES (?, ?, ?, ?)",
            ((idx, topic_id, ts, data) for idx, (ts, topic_id, data) in enumerate(all_messages, start=1)),
        )
        out.commit()
    finally:
        out.close()

    if all_messages:
        start_ns = all_messages[0][0]
        duration_ns = all_messages[-1][0] - all_messages[0][0]
    else:
        start_ns = 0
        duration_ns = 0

    topics_with_count = []
    for topic_id in sorted(topic_meta_by_id):
        name, msg_type, serialization_format, offered_qos_profiles = topic_meta_by_id[topic_id]
        topics_with_count.append(
            {
                "topic_metadata": {
                    "name": name,
                    "type": msg_type,
                    "serialization_format": serialization_format,
                    "offered_qos_profiles": offered_qos_profiles,
                },
                "message_count": topic_counts.get(topic_id, 0),
            }
        )

    metadata = {
        "rosbag2_bagfile_information": {
            "version": 4,
            "storage_identifier": "sqlite3",
            "relative_file_paths": ["merged_0.db3"],
            "duration": {"nanoseconds": int(duration_ns)},
            "starting_time": {"nanoseconds_since_epoch": int(start_ns)},
            "message_count": len(all_messages),
            "topics_with_message_count": topics_with_count,
            "compression_format": "",
            "compression_mode": "",
        }
    }
    (merged_output / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bag", type=Path, help="Bag dir or parent dir containing split bag dirs.")
    parser.add_argument("output_bag", type=Path, nargs="?", help="Output bag dir. Default: <input>_smooth")
    parser.add_argument(
        "--header-topic",
        action="append",
        dest="header_topics",
        help="Topic or fnmatch pattern whose first Header stamp should drive storage time. Can repeat.",
    )
    parser.add_argument(
        "--merge-split",
        action="store_true",
        help="If input/output contains split bag dirs, also write one merged playable bag.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        help="Merged output dir. Default: <output_bag>_merged",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists.")
    args = parser.parse_args()

    input_path = args.input_bag.expanduser().resolve()
    output_path = (args.output_bag or input_path.with_name(input_path.name + "_smooth")).expanduser().resolve()
    if not input_path.exists():
        print(f"[smooth-bag] input not found: {input_path}", file=sys.stderr)
        return 2

    header_topic_patterns = args.header_topics or sorted(DEFAULT_HEADER_TOPICS)
    input_bag_dirs = find_bag_dirs(input_path)
    if not input_bag_dirs:
        print(f"[smooth-bag] no metadata.yaml found under: {input_path}", file=sys.stderr)
        return 3

    global_record_start, global_header_start = collect_global_reference(input_bag_dirs, header_topic_patterns)
    print(f"[smooth-bag] input:  {input_path}")
    print(f"[smooth-bag] output: {output_path}")
    print(f"[smooth-bag] header topics: {', '.join(header_topic_patterns)}")
    print("[smooth-bag] copying input bag ...")
    copy_input(input_path, output_path, args.force)

    output_bag_dirs = find_bag_dirs(output_path)
    for bag_dir in output_bag_dirs:
        print(f"\n[smooth-bag] rewriting: {bag_dir.relative_to(output_path)}")
        stats = rewrite_bag_dir(bag_dir, global_record_start, global_header_start, header_topic_patterns)
        for topic_name in sorted(stats):
            old_times, new_times = stats[topic_name]
            if topic_name not in DEFAULT_HEADER_TOPICS and not any(fnmatch.fnmatch(topic_name, p) for p in header_topic_patterns):
                continue
            print(f"  {topic_name}")
            print(f"    before: {summarize_topic(old_times)}")
            print(f"    after : {summarize_topic(new_times)}")

    if args.merge_split and len(output_bag_dirs) > 1:
        merged_output = (
            args.merged_output.expanduser().resolve()
            if args.merged_output
            else output_path.with_name(output_path.name + "_merged")
        )
        print(f"\n[smooth-bag] merging split bags into: {merged_output}")
        merge_bag_dirs(output_bag_dirs, merged_output, args.force)

    print("\n[smooth-bag] done. Original message header.stamp values were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
