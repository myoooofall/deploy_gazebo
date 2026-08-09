# Offline Localization To Smooth Demo Bag

这个目录里的脚本用于把原始导航 bag 变成“展示用平滑轨迹 bag”。

## 目标流程
```text
raw split bag
  -> 平滑写盘时间，生成 <run>_smooth
  -> hdl_localization 离线定位并录制 /odom
  -> 裁剪导航开启到导航结束的时间段
  -> 剔除 /odom 跳变，平滑轨迹并降频
  -> 生成只用于 RViz 展示的新 bag
```

```bash
./smooth_bag_record_time.py /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3
./localize_record_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
./make_latest_demo_bag.sh easy_3_smooth 7 0 0
./play_demo_bag.sh easy_3_smooth_demo
```

第三步最后的 `7 0 0` 是目标相对出生点的 `forward left yaw`。如果不想画红色目标点，可以不加这三个数。


原始 bag 不会被修改或删除。

## 实物录制

推荐用 split 录制，避免 lidar 点云把 imu/nav 小话题挤在同一个 rosbag writer 里：

```bash
cd /home/teleai/Desktop/liangwang_ws/rl_sar_new/rl_sar/scripts
./record_nav_raw_bag_nx.sh easy_3
```

默认会生成：

```text
bag/easy_3/lidar
bag/easy_3/small
bag/easy_3/depth_norm
```

其中：

```text
lidar      只录 /rslidar_points
small      录 /imu/data、/tf、/odom、导航目标和指令等小话题
depth_norm 只录 /camera/depth/processed_norm
```

录 `depth_norm` 前，需要在 `rl_real_lite3` 的配置中打开：

```yaml
depth_debug_publish_enable: true
```

如果某次不想录深度图：

```bash
DEPTH_BAG_ENABLE=false ./record_nav_raw_bag_nx.sh easy_3
```

## 一条命令完成

只需要改第一行的原始 bag 路径：

```bash
cd /home/teleai/Desktop/liangwang_ws/rl_sar_new/rl_sar/scripts

./smooth_bag_record_time.py /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3
./localize_record_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
./make_latest_demo_bag.sh easy_3_smooth 7 0 0
./play_demo_bag.sh easy_3_smooth_demo
```

脚本会生成这些新 bag：

```text
easy_3_smooth
easy_3_smooth_localized
easy_3_smooth_demo
```

最后播放展示 bag：

```bash
./play_demo_bag.sh easy_3_smooth_demo
```

这个脚本会自动：

```text
发布 /map
打开 RViz
播放 demo bag
```

RViz 默认显示 `/map`、`/odom_smooth`、`/odom_path_smooth`。
如果 demo bag 里有 goal 信息，还会显示 `/demo_markers`、`/goal_pred_markers` 和 `/goal_pred_path_smooth`。

## 分步运行

### 1. 先平滑原始 bag 写盘时间

```bash
cd /home/teleai/Desktop/liangwang_ws/rl_sar_new/rl_sar/scripts
./smooth_bag_record_time.py /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3
```

默认会覆盖生成：

```text
/home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
```

不会生成 `_smooth_merged`，也不会修改原始 `easy_3`。

### 2. 平滑后的 bag 跑离线定位

```bash
./localize_record_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
```

默认会：

- 启动 hdl_localization
- 复用 `lite_cog_ros2/system/scripts/slam/start_localization_only.sh`
- 打开 RViz，并发布 `/map` 和 `/odom_path`
- 等待 8 秒，方便你用 `2D Pose Estimate` 设置初始位姿
- replay split bag
- 录制新的 localized bag，默认输出为 `<run>_localized`

如果出生点偏移明显，必须在 replay 前用 RViz 点一下 `2D Pose Estimate`。

### 3. 从 localized bag 生成平滑展示 bag

推荐短命令：自动寻找 `easy_3_smooth_localized`，并输出到 `easy_3_smooth_demo`。
如果希望画出红色目标点，后面加目标相对出生点的 `forward left yaw`。

```bash
./make_latest_demo_bag.sh easy_3_smooth 7 0 0
```

如果只想生成平滑轨迹和出生点，不画手动目标点，也可以不加这三个数：

```bash
./make_latest_demo_bag.sh easy_3_smooth
```

如果你已经知道 localized bag 的完整路径，也可以直接传路径：

```bash
./make_latest_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth_localized
```

底层完整命令是：

```bash
./make_smooth_trajectory_bag.py \
  /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth_localized \
  /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_demo_smooth \
  --force
```

默认会：

- 自动优先读取 `/odom`，没有则读 `/Odometry`
- 用 `/nav/goal_pred_map` 的第一帧和最后一帧裁剪导航有效时间段
- 默认不做跳变剔除，只做重采样和时间窗口平滑
- 对 `clutter_7_0_0`，`make_latest_demo_bag.sh` 默认只手动修复裁剪后 odom 的 `74:77` 点，也就是当前这条轨迹中间那次 ICP 尖峰
- `/demo_markers`：绿色出生点；如果传了 `goal_forward goal_left goal_yaw`，还会写入红色目标点
- `/goal_pred_markers` 和 `/goal_pred_path_smooth`：如果 localized bag 里有 `/nav/goal_pred_map`，会把预测目标单独画出来
- 输出 20Hz 平滑轨迹
- `/odom_path_smooth` 默认 2Hz 发布，避免 RViz 中后段越来越卡
- 使用 0.8 秒滑动窗口平滑 x/y/z/yaw

### 4. 播放展示 bag

```bash
./play_demo_bag.sh easy_3_smooth_demo
```

这个脚本会发布 `/map`、打开 RViz，并播放 demo bag。

## 常用参数

降低播放轨迹频率：

```bash
OUTPUT_HZ=10 ./make_offline_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
```

加大平滑程度：

```bash
SMOOTH_WINDOW=1.2 ./make_offline_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
```

如果要手动启用短尖峰剔除：

```bash
MAX_STEP_DIST=1.0 MAX_INTERP_SPEED=1.5 ./make_latest_demo_bag.sh easy_3_smooth
```

这会把短时间跳出去、但前后轨迹能合理连起来的 `/odom` 点用前后点插值替换，不会 hold 上一帧。

如果只想针对某条展示轨迹手动修复指定 odom 点，不启用自动判断：

```bash
REPAIR_INDEX_RANGE=74:77 ./make_latest_demo_bag.sh clutter_7_0_0
```

这里的索引是裁剪出导航时间段后的 `/odom` 序号。`clutter_7_0_0` 已经内置这个默认值，直接运行即可。

启用连续跳变截断：

```bash
MAX_OUTLIER_RUN=20 ./make_latest_demo_bag.sh easy_3_smooth
```

修改 replay 速度：

```bash
PLAY_RATE=0.3 ./make_offline_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
```

增加设置初始位姿的等待时间：

```bash
INIT_POSE_WAIT_SEC=15 ./make_offline_demo_bag.sh /home/teleai/Desktop/liangwang_ws/rl_sar_new/bag/easy_3_smooth
```

手动指定裁剪时间，而不用 `/nav/goal_pred_map`：

```bash
START_SEC=5.59 END_SEC=46.0 ./make_latest_demo_bag.sh clutter_straight_easy 7 0 0
```

`clutter_straight_easy` 已经在脚本里内置了这个裁剪范围，因为它的 `/nav/*` 话题提前结束，但 `/odom` 后半段仍然有效。

## 脚本说明

```text
smooth_bag_record_time.py
  只平滑原始 bag 中 /rslidar_points 和 /imu/data 的写盘 timestamp，让 RViz replay 不那么卡。
  不会修改消息内部 header.stamp。

replay_nav_split_bag.sh
  播放 lidar/small 分开录制的 split bag。

localize_record_bag.sh
  跑 hdl_localization，并录制带 /odom 的 localized bag。

make_smooth_trajectory_bag.py
  从 localized bag 里提取 /odom，裁剪导航时段，生成平滑展示 bag。

make_latest_demo_bag.sh
  自动找到 <run>_localized，生成 <run>_demo；也兼容旧的 <run>_localized_*。

play_demo_bag.sh
  发布地图、打开 RViz、播放 <run>_demo。

make_offline_demo_bag.sh
  串联 localize_record_bag.sh 和 make_smooth_trajectory_bag.py。
```

## 注意

- 原始 bag 没有 `/odom`，必须先跑离线定位。
- 展示 bag 只用于画图和视频叠加，不要拿它再做定位评估。
- 如果只想让原始 replay 更顺，先用 `smooth_bag_record_time.py` 生成 `_smooth` bag，再把 `_smooth` bag 传给本流程。
