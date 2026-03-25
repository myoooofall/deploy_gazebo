# Lite3 Offline Goal Evaluation Workflow

This directory provides a minimal workflow for **NX recording** and **host-side offline localization/evaluation**.

## 1) NX: record a raw bag

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
./record_lite3_raw_bag.sh run_maze_01
```

Default output:
- `$HOME/bags/lite3/raw/run_maze_01`

Recorded topics include:
- `/rslidar_points`, `/imu/data`, `/tf`, `/tf_static`
- `/Odometry` (if available)
- `/nav/goal_actual_map`, `/nav/goal_pred_map`, `/nav/goal_error_body`, `/nav_goal_body`
- `/camera/depth/processed`, `/camera/depth/processed_norm`

## 2) Copy bag to host

```bash
rsync -avP <nx_user>@<nx_ip>:~/bags/lite3/raw/run_maze_01 ~/bags/lite3/raw/
```

## 3) Host: replay raw bag + run localization + record enriched bag

```bash
cd ~/liangwang_ws/rl_sar_new/rl_sar/src/rl_sar/scripts/offline
LOCALIZATION_CMD='ros2 launch hdl_localization lite_localization.launch.py enable_nav2:=false use_sim_time:=true map_path:=/ABS/PATH/TO/map.pcd' \
./replay_localize_and_record.sh ~/bags/lite3/raw/run_maze_01
```

Default enriched output:
- `$HOME/bags/lite3/enriched/<run_name>_enriched_YYYYmmdd_HHMMSS`

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

## Notes

- `rl_real_lite3` now falls back to publishing **body-frame goal values on existing goal topics** when `/Odometry` is unavailable.
- In fallback mode, goal topics use frame `base_link`; with `/Odometry` they use frame `map`.
- `summary.txt` includes sample counts and simple mean/p95/max error statistics.
