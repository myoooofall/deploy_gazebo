#include "depth_buffer.hpp"
#include <opencv2/highgui.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

static std::atomic<bool> g_depth_console_log_enabled{true};

void DepthBuffer::SetConsoleLogEnabled(bool enabled)
{
    g_depth_console_log_enabled.store(enabled, std::memory_order_relaxed);
}

bool DepthBuffer::IsConsoleLogEnabled()
{
    return g_depth_console_log_enabled.load(std::memory_order_relaxed);
}

static float Percentile(std::vector<float> values, double q)
{
    if (values.empty())
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (q <= 0.0) q = 0.0;
    if (q >= 1.0) q = 1.0;
    const size_t k = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
    return values[k];
}
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

    // Invalid-depth handling: invalid -> far range (5.0m).
    // This is safer for navigation than treating invalid as near obstacle.
    torch::Tensor valid_mask = torch::isfinite(depth_tensor) & (depth_tensor > 0.0);
    torch::Tensor invalid_mask = (~valid_mask).to(torch::kFloat32);
    const float invalid_ratio_raw = invalid_mask.mean().item<float>();
    depth_tensor = torch::where(valid_mask, depth_tensor, torch::full_like(depth_tensor, 5.0));
    
    // First resize to intermediate size (60, 106): height=60, width=106.
    // Use nearest for depth to avoid creating interpolated "fake" depths.
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // Add batch and channel dims for interpolate
    depth_tensor = torch::nn::functional::interpolate(
        depth_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{60, 106})
            .mode(torch::kNearest)
    ).squeeze(0).squeeze(0);  // Remove dims back to [60, 106]
    torch::Tensor invalid_mask_resized = torch::nn::functional::interpolate(
        invalid_mask.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{60, 106})
            .mode(torch::kNearest)
    ).squeeze(0).squeeze(0);
    
    // Crop width: remove first 10 and last 10 columns [10:-10], keeping height unchanged
    // Result: [60, 86]
    depth_tensor = depth_tensor.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(10, -10)});  // [60, 86]
    invalid_mask_resized = invalid_mask_resized.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(10, -10)});
    
    // // 打印裁剪后的范围
    // std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 将深度值裁剪到0.05-5.0米范围
    depth_tensor = torch::clamp(depth_tensor, 0.1, 5.0);
    const float invalid_ratio_model_input = invalid_mask_resized.mean().item<float>();
    const float far_ratio_model_input = (depth_tensor >= 4.9).to(torch::kFloat32).mean().item<float>();

    torch::Tensor depth_for_stats = depth_tensor.contiguous();
    const float *depth_ptr = depth_for_stats.data_ptr<float>();
    const int64_t depth_numel = depth_for_stats.numel();
    std::vector<float> depth_values;
    depth_values.reserve(static_cast<size_t>(depth_numel));
    for (int64_t i = 0; i < depth_numel; ++i)
    {
        depth_values.push_back(depth_ptr[i]);
    }
    const float p50_m = Percentile(depth_values, 0.50);
    const float p90_m = Percentile(depth_values, 0.90);
    const float p99_m = Percentile(depth_values, 0.99);
    
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
    if (DepthBuffer::IsConsoleLogEnabled())
    {
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
                      << "]\n";
            depth_stats_logged_once = true;
        }
    }

    if (DepthBuffer::IsConsoleLogEnabled())
    {
        static bool depth_health_inited = false;
        static auto depth_health_last_tp = std::chrono::steady_clock::now();
        static uint64_t depth_health_samples = 0;
        static double depth_health_sum_invalid_raw = 0.0;
        static double depth_health_sum_invalid_model = 0.0;
        static double depth_health_sum_far_model = 0.0;
        static double depth_health_max_raw = 0.0;
        static double depth_health_last_p50 = 0.0;
        static double depth_health_last_p90 = 0.0;
        static double depth_health_last_p99 = 0.0;

        if (!depth_health_inited)
        {
            depth_health_last_tp = std::chrono::steady_clock::now();
            depth_health_inited = true;
        }
        depth_health_samples += 1;
        depth_health_sum_invalid_raw += static_cast<double>(invalid_ratio_raw);
        depth_health_sum_invalid_model += static_cast<double>(invalid_ratio_model_input);
        depth_health_sum_far_model += static_cast<double>(far_ratio_model_input);
        depth_health_max_raw = std::max(depth_health_max_raw, static_cast<double>(raw_depth_max_m));
        depth_health_last_p50 = static_cast<double>(p50_m);
        depth_health_last_p90 = static_cast<double>(p90_m);
        depth_health_last_p99 = static_cast<double>(p99_m);

        const auto now_tp = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now_tp - depth_health_last_tp).count() >= 1.0 && depth_health_samples > 0)
        {
            const double avg_invalid_raw = depth_health_sum_invalid_raw / static_cast<double>(depth_health_samples);
            const double avg_invalid_model = depth_health_sum_invalid_model / static_cast<double>(depth_health_samples);
            const double avg_far_model = depth_health_sum_far_model / static_cast<double>(depth_health_samples);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << "[NAV][DEPTH][HEALTH] samples=" << depth_health_samples
                << " invalid_raw=" << avg_invalid_raw
                << " invalid_model=" << avg_invalid_model
                << " far_model=" << avg_far_model
                << " p50/p90/p99_m=[" << depth_health_last_p50 << "/" << depth_health_last_p90 << "/" << depth_health_last_p99 << "]"
                << " raw_max_m=" << depth_health_max_raw;
            std::cout << oss.str() << '\n';

            depth_health_samples = 0;
            depth_health_sum_invalid_raw = 0.0;
            depth_health_sum_invalid_model = 0.0;
            depth_health_sum_far_model = 0.0;
            depth_health_max_raw = 0.0;
            depth_health_last_tp = now_tp;
        }
    }

    return depth_tensor;  // Shape: [30, 43]
}
