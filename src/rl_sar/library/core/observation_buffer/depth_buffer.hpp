#ifndef DEPTH_BUFFER_HPP
#define DEPTH_BUFFER_HPP

#include <torch/torch.h>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
class DepthBuffer
{
public:
    static void SetConsoleLogEnabled(bool enabled);
    static bool IsConsoleLogEnabled();

    DepthBuffer(int num_envs, int height, int width, int include_history_steps);
    DepthBuffer();

    void reset(std::vector<int> reset_idxs, torch::Tensor new_depth);
    void insert(torch::Tensor new_depth);
    torch::Tensor get_depth_vec();  // 返回前 include_history_steps-1 帧用于推理（保留一帧延迟）
    torch::Tensor process_depth_image(const sensor_msgs::msg::Image::SharedPtr msg,
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher,
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_norm_publisher = nullptr);

private:
    int num_envs;
    int height;
    int width;
    int include_history_steps;
    torch::Tensor depth_buf;
    bool initialized = false;  // 标记是否已经初始化（第一次插入）
};

#endif // DEPTH_BUFFER_HPP
