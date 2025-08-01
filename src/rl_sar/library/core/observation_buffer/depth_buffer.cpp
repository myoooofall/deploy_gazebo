#include "depth_buffer.hpp"
#include <opencv2/highgui.hpp>
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
}

void DepthBuffer::insert(torch::Tensor new_depth)
{
    // Shift observations back.
    torch::Tensor shifted_depth = depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(include_history_steps - 1, torch::indexing::None), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}).clone();
    depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, include_history_steps - 1), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = shifted_depth;

    // Add new observation.
    depth_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-1), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None)}) = new_depth;
}

torch::Tensor DepthBuffer::get_depth_vec()
{
    return depth_buf;
}

torch::Tensor DepthBuffer::process_depth_image(const sensor_msgs::msg::Image::SharedPtr msg,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher)  
{
    // // 打印消息格式信息
    // std::cout << "深度图编码格式: " << msg->encoding << std::endl;
    // std::cout << "每个像素的字节数: " << msg->step / msg->width << std::endl;
    // std::cout << "是否大端序: " << msg->is_bigendian << std::endl;
    
    // // 打印原始图像尺寸
    // std::cout << "原始图像尺寸: " << msg->width << "x" << msg->height << std::endl;
    // std::cout << "目标图像尺寸: " << width << "x" << height << std::endl;
    
    // 正确读取16位深度数据
    std::vector<uint16_t> depth_data;
    depth_data.reserve(msg->width * msg->height);
    const uint8_t* data_ptr = msg->data.data();
    
    for (size_t i = 0; i < msg->data.size(); i += 2) {
        uint16_t depth;
        if (msg->is_bigendian) {
            depth = (static_cast<uint16_t>(data_ptr[i]) << 8) | static_cast<uint16_t>(data_ptr[i + 1]);
        } else {
            depth = (static_cast<uint16_t>(data_ptr[i + 1]) << 8) | static_cast<uint16_t>(data_ptr[i]);
        }
        depth_data.push_back(depth);
    }
    
    torch::Tensor depth_tensor = torch::from_blob(depth_data.data(), 
        {msg->height, msg->width}, torch::kInt16).clone();
    
    // // 打印原始深度值范围
    // std::cout << "原始深度值范围: [" << depth_tensor.min().item<int16_t>() << ", " << depth_tensor.max().item<int16_t>() << "]" << std::endl;
    
    // 转换为float类型并转换为米
    depth_tensor = depth_tensor.to(torch::kFloat32) / 1000.0;  // 转换为米
    
    // // 打印转换为米后的范围
    // std::cout << "转换为米后的范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // // 将深度值裁剪到0.2-2.0米范围
    depth_tensor = torch::clamp(depth_tensor, 0.2, 2.0);
    
    // // 打印裁剪后的范围
    // std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 归一化到-0.5到0.5范围
    depth_tensor = (depth_tensor - 1) / 2;  // (x - 1.1) / 1.8 将0.2-2.0映射到-0.5-0.5
    
    // // 打印归一化后的范围
    // std::cout << "归一化后的范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    
    // 调整大小到目标尺寸
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // 添加batch和channel维度
    // std::cout<<"123"<<std::endl;
    depth_tensor = torch::nn::functional::interpolate(
        depth_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear)
            .align_corners(false)
    );
    
    // 打印调整后的深度值范围
    // std::cout << "调整后的深度值范围: [" << depth_tensor.min().item<float>() << ", " << depth_tensor.max().item<float>() << "]" << std::endl;
    if (processed_publisher) {
        // Get dimensions
        int height = depth_tensor.sizes()[2]; // Assuming HxW
        int width = depth_tensor.sizes()[3]; // Assuming HxW
        cv::Mat depth_mat(height, width, CV_32FC1, depth_tensor.data_ptr<float>());
        // cv::imshow("Processed Depth Image", depth_mat);
        // cv::waitKey(1);  // Wait for a short time to display the image

        cv::Mat depth_uint16_mat;
        // 将浮点深度值转换为毫米，并转换为 CV_16UC1 (unsigned short)
        // 假设最大深度为 10米，即 10000毫米
        // 可以根据实际传感器范围调整缩放因子
        depth_mat.convertTo(depth_uint16_mat, CV_16UC1, 1000.0); // 乘以1000将米转换为毫米
        auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", depth_uint16_mat).toImageMsg();
        processed_publisher->publish(*image_msg);
    }
    return depth_tensor.squeeze(0).squeeze(0);  // 移除batch和channel维度
}