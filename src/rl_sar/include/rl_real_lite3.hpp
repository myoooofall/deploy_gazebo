/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_LITE3_HPP
#define RL_REAL_LITE3_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "lite3/fsm.hpp"

// Lite3 SDK
#include "sender.h"
#include "receiver.h"
#include "robot_types.h"
#include <cmath>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

//Retroid Gamepad
#include "gamepad.h"
#include "retroid_gamepad.h"
#include "gamepad_keys.h"

#if defined(USE_ROS1)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#elif defined(USE_ROS2)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class RL_Real : public RL
#if defined(USE_ROS2)
    , public rclcpp::Node
#endif
{
public:
    RL_Real();
    ~RL_Real();

private:
    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_udpRecv;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_navi;
    std::shared_ptr<LoopFunc> loop_vision;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // Lite3 SDK interface
    Sender* sender_ = nullptr;
    Receiver* receiver_ = nullptr;
    RobotCmd robot_joint_cmd_{};
    RobotData* robot_data_=nullptr;
    void UDPRecv();
    void EulerToQuaternion(float roll, float pitch, float yaw, float q[4]);
    void HandleKeyboard();
#if defined(USE_ROS2)
    void PublishSlamImuFromSdk(const RobotState<double> &state);
#endif

    //Retroid Gamepad
    std::shared_ptr<RetroidGamepad> gamepad_ptr_;
    RetroidKeys rt_keys_record_, rt_keys_;
    bool first_flag_;
    bool nav_gamepad_enable_ = false;
    bool joystick_override_active_ = false;

    // hierarchical navigation (high-level policy @ 10Hz)
    void RunHighLevel();
    bool InitHierarchicalNav();
    void UpdateHighFrequencyObs();
    void StartNavObsLogIfNeeded(uint64_t goal_seq);
    void StopNavObsLogIfNeeded();
    void WriteNavObsSemanticLog(
        uint64_t goal_seq,
        bool new_goal,
        double time_io,
        double timer_norm,
        const torch::Tensor &goal_init_body,
        const torch::Tensor &pred_target_body,
        const torch::Tensor &cmd_raw,
        const torch::Tensor &cmd_filtered,
        const torch::Tensor &prev_high_cmd_scaled,
        const torch::Tensor &base_ang_vel_scaled,
        const torch::Tensor &projected_gravity,
        const torch::Tensor &dof_pos_raw,
        const torch::Tensor &dof_pos_term,
        const torch::Tensor &dof_vel_raw,
        const torch::Tensor &dof_vel_term,
        const torch::Tensor &actions);
    double GetNavEpisodeElapsedSec() const;
    void ResetNavEpisodeClock();
    void ResetNavModelStates();
    void WarmupNavModels(int obs_dim, int obs_io_dim, int hf_dim);
    void WarmupNavObservationBuffers(int obs_dim, int obs_io_dim, int hf_dim);
    void PrimeNavRuntimeOnce();
    void StartNavGoalInput();
    void SetNavGoalBody(double goal_x, double goal_y, double goal_yaw, const char *source);
    [[noreturn]] void DisableNavigationWithError(const std::string &stage, const std::string &detail);
    void StartPerfCsvIfNeeded();
    void StopPerfCsvIfNeeded();
    void WritePerfCsvRow(
        const char *event,
        double dt_ms = std::numeric_limits<double>::quiet_NaN(),
        double exec_ms = std::numeric_limits<double>::quiet_NaN(),
        double sdk_imu_stamp_ms = std::numeric_limits<double>::quiet_NaN(),
        double sdk_imu_stamp_delta_ms = std::numeric_limits<double>::quiet_NaN(),
        double imu_pub_dt_ms = std::numeric_limits<double>::quiet_NaN(),
        double imu_header_stamp_ms = std::numeric_limits<double>::quiet_NaN(),
        double imu_header_delta_ms = std::numeric_limits<double>::quiet_NaN(),
        double rl_tick_ms = std::numeric_limits<double>::quiet_NaN(),
        double rl_infer_ms = std::numeric_limits<double>::quiet_NaN(),
        double rl_total_ms = std::numeric_limits<double>::quiet_NaN(),
        double vision_tick_ms = std::numeric_limits<double>::quiet_NaN(),
        double vision_process_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_tick_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_vision_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_high_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_reset_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_model_reset_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_obs_build_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_state_copy_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_tensor_create_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_cat_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_hist_reset_insert_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_hist_get_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_to_device_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_to_cpu_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_post_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_log_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_publish_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_total_ms = std::numeric_limits<double>::quiet_NaN(),
        double udp_get_state_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_obs_hist_reset_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_obs_io_hist_reset_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_hf_reset_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_obs_hist_insert_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_obs_io_hist_insert_ms = std::numeric_limits<double>::quiet_NaN(),
        double nav_hf_insert_ms = std::numeric_limits<double>::quiet_NaN());

    // nav state shared across loops
    std::atomic<bool> nav_enabled_{false};
    std::atomic<uint64_t> nav_goal_seq_{0};

    std::atomic<double> nav_goal_body_x_{0.0};
    std::atomic<double> nav_goal_body_y_{0.0};
    std::atomic<double> nav_goal_body_yaw_{0.0};

    std::atomic<double> nav_cmd_x_{0.0};
    std::atomic<double> nav_cmd_y_{0.0};
    std::atomic<double> nav_cmd_yaw_{0.0};
    std::atomic<bool> nav_goal_input_active_{false};
    // buffers and models (guarded as needed)
    torch::jit::script::Module nav_high_model_;
    torch::jit::script::Module nav_vision_model_;
    torch::Device nav_infer_device_ = torch::Device(torch::kCPU);
    std::string nav_config_path_;
    std::string nav_high_model_path_;
    std::string nav_vision_model_path_;

    int nav_obs_hist_len_ = 10;
    int nav_obs_io_hist_len_ = 10;
    int nav_highfreq_hist_len_ = 20;
    int nav_obs_dim_ = 0;
    int nav_obs_io_dim_ = 0;
    int nav_hf_dim_ = 0;
    int nav_vision_channels_ = 1;      // expected channels for nav_vision_model input
    double nav_dt_ = 0.1;               // 10Hz
    double nav_episode_length_s_ = 30;  // default if not specified
    double nav_goal_stop_radius_ = 0.5; // stop nav when sqrt(pred_x^2 + pred_y^2) <= this
    // Training-aligned high command shaping params:
    // filtered = alpha * last + (1-alpha) * raw
    // delta-limited by high_command_max_step_*
    // then clipped by clip_commands_vx/vy/w
    double nav_command_clip_x_ = 1.0;
    double nav_command_clip_y_ = 0.5;
    double nav_command_clip_yaw_ = 0.6;
    double nav_high_command_filter_alpha_ = 0.8;
    double nav_high_command_max_step_x_ = 0.1;
    double nav_high_command_max_step_y_ = 0.05;
    double nav_high_command_max_step_yaw_ = 0.12;
    int nav_watchdog_timeout_ms_ = 1200; // stale high-level heartbeat timeout
    bool nav_perf_log_enable_ = false;   // controlled by nav_debug_enable_
    double nav_perf_log_interval_s_ = 1.0;
    // Runtime diagnostics policy: keep stdout quiet unless nav_debug_enable_ is on.
    bool nav_console_info_enable_ = false;
    bool nav_loop_overrun_log_enable_ = false;
    bool nav_loop_lifecycle_log_enable_ = false;
    bool nav_depth_console_log_enable_ = false;
    bool nav_depth_debug_publish_enable_ = false; // publishes extra image topics, keep separate from logs
    bool nav_debug_enable_ = false;      // master switch for terminal diagnostics
    double nav_debug_log_interval_s_ = 1.0;
    bool nav_perf_csv_enable_ = false;   // structured timing CSV for NX A/B tests
    double nav_perf_csv_flush_interval_s_ = 1.0;
    std::string nav_perf_csv_dir_;
    std::string nav_perf_csv_path_;
    std::ofstream nav_perf_csv_stream_;
    std::mutex nav_perf_csv_mutex_;
    std::chrono::steady_clock::time_point nav_perf_csv_start_tp_;
    std::chrono::steady_clock::time_point nav_perf_csv_last_flush_tp_;
    bool nav_obs_log_enable_ = false;    // structured semantic-observation csv logger (for sim-vs-real comparison)
    double nav_obs_log_interval_s_ = 0.1;
    std::string nav_obs_log_dir_;
    std::string nav_obs_log_path_;
    std::ofstream nav_obs_log_stream_;
    bool nav_obs_log_active_ = false;
    uint64_t nav_obs_log_goal_seq_ = 0;
    double nav_obs_log_last_time_io_ = -1.0;
    // Lite3 SDK IMU angular_velocity_* fields are named as roll/pitch/yaw rates.
    // When true, interpret them as Euler angle rates and convert to body-frame
    // angular velocity [wx, wy, wz] before feeding policies.
    bool nav_sdk_gyro_is_euler_rate_ = true;

    std::mutex nav_highfreq_mutex_;
    std::mutex nav_state_mutex_;
    ObservationBuffer nav_highfreq_buf_;
    ObservationBuffer nav_obs_hist_buf_;
    ObservationBuffer nav_obs_io_hist_buf_;

    torch::Tensor nav_position_targets_body_initial_;
    torch::Tensor nav_spawn_positions_body_initial_;
    torch::Tensor nav_high_command_;

    std::atomic<uint64_t> nav_active_goal_seq_{0};
    std::atomic<int64_t> nav_episode_start_ns_{0};
    // Cached elapsed time samples for debugging/inspection. Both are derived from the same episode clock.
    std::atomic<double> nav_time_io_{0.0};
    std::atomic<double> nav_time_io_hf_{0.0};
    std::atomic<double> nav_timer_left_{0.0};
    std::atomic<uint64_t> nav_hl_beat_seq_{0};
    std::atomic<bool> nav_runtime_prime_done_{false};

    std::mutex nav_last_actions_mutex_;
    std::vector<float> nav_last_actions_;

    // others
    int motiontime = 0;
    std::vector<double> mapped_joint_positions;
    std::vector<double> mapped_joint_velocities;

#if defined(USE_ROS2)
    // depth
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_subscriber;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_depth_publisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_depth_norm_publisher;
    void DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void RunVision();
    std::mutex depth_raw_msg_mutex_;
    sensor_msgs::msg::Image::SharedPtr depth_raw_latest_msg_;
    std::mutex depth_buffer_mutex_;

    // nav interface
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr nav_goal_body_subscriber;
    void NavGoalBodyCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg);

    // map-frame target comparison interface
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr nav_goal_actual_map_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr nav_goal_pred_map_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nav_goal_compare_markers_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr nav_goal_error_body_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr nav_cmd_high_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr nav_cmd_applied_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr sdk_imu_publisher_;
    bool sdk_imu_pub_started_ = false;

    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void PublishNavGoalComparison(
        const torch::Tensor &pred_target_body,
        uint64_t goal_seq,
        bool new_goal);
    bool TryProjectBodyTargetToMap(
        double body_x,
        double body_y,
        double body_yaw,
        double base_x,
        double base_y,
        double base_yaw,
        geometry_msgs::msg::PoseStamped *pose_out) const;

    // latest localization pose (map -> base_link)
    mutable std::mutex odom_pose_mutex_;
    bool odom_pose_received_ = false;
    double odom_map_x_ = 0.0;
    double odom_map_y_ = 0.0;
    double odom_map_yaw_ = 0.0;
    rclcpp::Time odom_stamp_{0, 0, RCL_ROS_TIME};

    // goal projected in map frame and latched per goal sequence
    bool nav_goal_actual_map_valid_ = false;
    uint64_t nav_goal_actual_map_goal_seq_ = 0;
    geometry_msgs::msg::PoseStamped nav_goal_actual_map_pose_;
#endif

#if defined(USE_ROS1)
    geometry_msgs::Twist cmd_vel;
    ros::Subscriber cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
#elif defined(USE_ROS2)
    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
#endif
};

#endif // RL_REAL_LITE3_HPP
