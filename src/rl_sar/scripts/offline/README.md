# Lite3 Offline Goal Evaluation Workflow

This directory provides a minimal workflow for **NX recording** and **host-side offline localization/evaluation**.

## 1) NX: record a raw bag

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
./record_lite3_raw_bag.sh run_maze_01
```

Use `--lite` to reduce bag size (skip depth topics):

```bash
./record_lite3_raw_bag.sh --lite run_maze_01
```

Use `--full` for full recording (default):

```bash
./record_lite3_raw_bag.sh --full run_maze_01
```

Default output:
- `$HOME/bags/lite3/raw/run_maze_01`

`--full` recorded topics include:
- `/rslidar_points`, `/imu/data`, `/tf`, `/tf_static`
- `/Odometry` (if available)
- `/nav/goal_actual_map`, `/nav/goal_pred_map`, `/nav/goal_error_body`, `/nav_goal_body`
- `/camera/depth/processed`, `/camera/depth/processed_norm`

`--lite` excludes:
- `/camera/depth/processed`, `/camera/depth/processed_norm`

## 2) NX push bag to Mac

Option A: direct rsync command from NX

```bash
rsync -avP ~/bags/lite3/raw/run_maze_01 <mac_user>@<mac_ip>:~/bags/lite3/raw/
```

Option B: helper script (run on NX)

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
MAC_USER=<your_mac_user> MAC_HOST=<your_mac_ip> ./pull_bag_from_nx.sh run_maze_01
```

You can also override destination path:

```bash
MAC_USER=<your_mac_user> MAC_HOST=<your_mac_ip> MAC_BASE_DIR=~/bags/lite3/raw ./pull_bag_from_nx.sh run_maze_01
```

## 3) Host: replay raw bag + run localization + record enriched bag

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
LOCALIZATION_CMD='ros2 launch hdl_localization lite_localization.launch.py enable_nav2:=false use_sim_time:=true map_path:=/ABS/PATH/TO/map.pcd' \
./replay_localize_and_record.sh ~/bags/lite3/raw/run_maze_01
```

Default enriched output:
- `$HOME/bags/lite3/enriched/<run_name>_enriched_YYYYmmdd_HHMMSS`

## 3.5) Optional: smooth rosbag storage time for RViz replay

This only rewrites rosbag2 SQLite `messages.timestamp` values. It does not
change serialized message contents, so lidar/IMU `header.stamp` stays original.

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
./smooth_bag_record_time.py --merge-split ~/liangwang_ws/rl_sar_new/bag/5-19-2
```

Outputs:
- `~/liangwang_ws/rl_sar_new/bag/5-19-2_smooth`
- `~/liangwang_ws/rl_sar_new/bag/5-19-2_smooth_merged`

For split bags, use the `_smooth_merged` directory with `ros2 bag play`.

## 4) Export CSV metrics from bag

```bash
./export_goal_metrics_from_bag.py ~/bags/lite3/enriched/<your_enriched_bag_dir>
```

Output directory:
- `<bag_dir>/analysis/`

Generated files:
- `odometry.csv`
- `goal_actual.csv`
- `goal_pred.csv`
- `goal_error_body.csv`
- `goal_compare_aligned.csv`
- `goal_error_offline_odom.csv`
- `summary.txt`

## 5) Export depth topic to MP4 (no ffmpeg required)

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
./export_depth_video_from_bag.py \
  ~/liangwang_ws/rl_sar_new/bag/3_25/run_20260401_170924 \
  --topic /camera/depth/processed_norm \
  --out ~/liangwang_ws/rl_sar_new/bag/3_25/run_20260401_170924/depth_processed_norm.mp4
```

Notes:
- Default mode is `--mode direct` (reads bag file directly), which avoids ROS QoS/playback issues.
- `processed_norm` usually looks better directly for preview videos.
- If you want raw depth, switch topic to `/camera/depth/processed`.
- Use `--colormap none` for grayscale, or keep default `turbo` for heatmap style.
- Optional fallback playback mode:
  - `--mode play`
- In `--mode play`, optional QoS override:
  - `--qos-overrides-path ./qos_overrides.yaml`

## Notes

- `rl_real_lite3` now falls back to publishing **body-frame goal values on existing goal topics** when `/Odometry` is unavailable.
- In fallback mode, goal topics use frame `base_link`; with `/Odometry` they use frame `map`.
- `summary.txt` includes sample counts and simple mean/p95/max error statistics.
