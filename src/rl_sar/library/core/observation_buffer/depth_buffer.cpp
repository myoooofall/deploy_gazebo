#include "depth_buffer.hpp"
#include <opencv2/highgui.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <iomanip>
#include <iostream>
// DepthBuffer implementation
DepthBuffer::DepthBuffer() {}

DepthBuffer::DepthBuffer(int num_envs,
                        int height,
                        int width,
                        int include_history_steps)
    : num_envs(num_envs),
      height(height),
      width(width),
      include_history_steps(include_history_steps)
{
    depth_buf = torch::zeros({num_envs, include_history_steps, height, width}, torch::dtype(torch::kFloat32));
}

void DepthBuffer::reset(std::vector<int> reset_idxs, torch::Tensor new_depth)
{
    std::vector<torch::indexing::TensorIndex> indices;
    for (int idx : reset_idxs)
    {
        indices.push_back(torch::indexing::Slice(idx));
    }
    depth_buf.index_put_(indices, new_depth.repeat({1, include_history_steps, 1, 1}));
    // reset后，如果所有环境都被reset，则重置initialized标志
    // 这里简化处理：reset后保持initialized状态，因为buffer已经被填充了
}

void DepthBuffer::insert(torch::Tensor new_depth)
{

    
    if (!initialized) {
        // 第一次插入：用第一帧复制满整个buffer
        for (int i = 0; i < include_history_steps; ++i)
        {
            depth_buf.index({0, i, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
        }
        initialized = true;
    } else {
        // 后续插入：FIFO队列逻辑（索引越大越新）
        if (include_history_steps <= 1)
        {
            depth_buf.index({0, 0, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
            return;
        }
        for (int i = 0; i < include_history_steps - 1; ++i)
        {
            depth_buf.index({0, i, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) =
                depth_buf.index({0, i + 1, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
        }
        depth_buf.index({0, include_history_steps - 1, torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
    }
}

torch::Tensor DepthBuffer::get_depth_vec()
{
    // Return history window with one-frame delay (drop newest frame):
    // - include_history_steps=9 -> return [0..7] (8 frames)
    // - include_history_steps=2 -> return [0]    (1 frame)
    // - include_history_steps=1 -> return [0]
    if (include_history_steps <= 1)
    {
        return depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 1),
                                torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
    }
    return depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, include_history_steps - 1),
                            torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)});
}

torch::Tensor DepthBuffer::process_depth_image(const sensor_msgs::msg::Image::SharedPtr msg,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_norm_publisher)
{
    torch::Tensor depth_tensor;
    try
    {
        if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
            msg->encoding == sensor_msgs::image_encodings::MONO16)
        {
            // 16UC1: millimeters -> meters
            auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            const cv::Mat &depth_u16 = cv_ptr->image;
            cv::Mat depth_f32;
            depth_u16.convertTo(depth_f32, CV_32FC1, 1.0 / 1000.0);
            depth_tensor = torch::from_blob(
                depth_f32.data,
                {depth_f32.rows, depth_f32.cols},
                torch::kFloat32).clone();
        }
        else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
        {
            // 32FC1: already in meters
            auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            const cv::Mat &depth_f32 = cv_ptr->image;
            depth_tensor = torch::from_blob(
                depth_f32.data,
                {depth_f32.rows, depth_f32.cols},
                torch::kFloat32).clone();
        }
        else
        {
            throw std::runtime_error("Unsupported depth encoding: " + msg->encoding);
        }
    }
    catch (const cv_bridge::Exception &e)
    {
        throw std::runtime_error(std::string("Depth decode failed: ") + e.what());
    }

    // Invalid-depth handling for deployment debugging: invalid -> nearest range (0.1m)
    torch::Tensor valid_mask = torch::isfinite(depth_tensor) & (depth_tensor > 0.0);
    depth_tensor = torch::where(valid_mask, depth_tensor, torch::full_like(depth_tensor, 0.1));
    
    // First resize to intermediate size (60, 106): height=60, width=106
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // Add batch and channel dims for interpolate
    depth_tensor = torch::nn::functional::interpolate(
        depth_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{60, 106})
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze(0).squeeze(0);  // Remove dims back to [60, 106]
    
    // Crop width: remove first 10 and last 10 columns [10:-10], keeping height unchanged
    // Result: [60, 86]
    depth_tensor = depth_tensor.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(10, -10)});  // [60, 86]
    
    // // 打印裁剪后的范围
    // std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 将深度值裁剪到0.05-5.0米范围
    depth_tensor = torch::clamp(depth_tensor, 0.1, 5.0);
    
    // 发布用于可视化的深度图（在归一化之前保存原始值）
    if (processed_publisher) {
        // depth_tensor shape is [60, 86] at this point
        torch::Tensor processed_tensor = depth_tensor.contiguous();
        
        // Get dimensions (height=60, width=86)
        int h = processed_tensor.size(0);
        int w = processed_tensor.size(1);
        
        
        cv::Mat depth_mat(h, w, CV_32FC1, processed_tensor.data_ptr<float>());
        
        // 使用32FC1编码发布，单位是米，rqt可以正确处理
        auto image_msg = cv_bridge::CvImage(msg->header, "32FC1", depth_mat).toImageMsg();
        processed_publisher->publish(*image_msg);
    }
    
    const float raw_depth_min_m = depth_tensor.min().item<float>();
    const float raw_depth_max_m = depth_tensor.max().item<float>();
    const float raw_depth_mean_m = depth_tensor.mean().item<float>();

    // 归一化到[-1, 0]范围 (用于推理)
    depth_tensor = depth_tensor / 5 - 0.5;
    depth_tensor = torch::nn::functional::avg_pool2d(
        depth_tensor.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::AvgPool2dFuncOptions({2, 2}).stride({2, 2})
    ).squeeze(0).squeeze(0);

    // Publish normalized depth used by model inference.
    if (processed_norm_publisher) {
        torch::Tensor norm_tensor = depth_tensor.contiguous();
        const int h_norm = norm_tensor.size(0);
        const int w_norm = norm_tensor.size(1);
        cv::Mat norm_mat(h_norm, w_norm, CV_32FC1, norm_tensor.data_ptr<float>());
        auto norm_msg = cv_bridge::CvImage(msg->header, "32FC1", norm_mat).toImageMsg();
        processed_norm_publisher->publish(*norm_msg);
    }

    const float model_depth_min = depth_tensor.min().item<float>();
    const float model_depth_max = depth_tensor.max().item<float>();
    const float model_depth_mean = depth_tensor.mean().item<float>();
    static bool depth_stats_logged_once = false;
    if (!depth_stats_logged_once)
    {
        std::cout << std::fixed << std::setprecision(4)
                  << "[NAV][DEPTH] raw_m[min=" << raw_depth_min_m
                  << ", max=" << raw_depth_max_m
                  << ", mean=" << raw_depth_mean_m
                  << "] model[min=" << model_depth_min
                  << ", max=" << model_depth_max
                  << ", mean=" << model_depth_mean
                  << "]" << std::endl;
        depth_stats_logged_once = true;
    }

    return depth_tensor;  // Shape: [30, 43]
}
