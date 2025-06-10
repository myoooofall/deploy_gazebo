#include <librealsense2/rs.hpp>
#include <torch/torch.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

class DepthBuffer {
public:
    DepthBuffer(int num_envs, int height, int width, int include_history_steps)
        : num_envs(num_envs), height(height), width(width), 
          include_history_steps(include_history_steps) {
        depth_buf = torch::zeros({num_envs, include_history_steps, height, width}, 
                                torch::dtype(torch::kFloat32));
    }

    // Add move constructor and move assignment operator
    DepthBuffer(DepthBuffer&& other) noexcept
        : num_envs(other.num_envs), height(other.height), width(other.width),
          include_history_steps(other.include_history_steps),
          depth_buf(std::move(other.depth_buf)) {}

    DepthBuffer& operator=(DepthBuffer&& other) noexcept {
        if (this != &other) {
            num_envs = other.num_envs;
            height = other.height;
            width = other.width;
            include_history_steps = other.include_history_steps;
            depth_buf = std::move(other.depth_buf);
        }
        return *this;
    }

    // Delete copy constructor and copy assignment operator
    DepthBuffer(const DepthBuffer&) = delete;
    DepthBuffer& operator=(const DepthBuffer&) = delete;

    void insert(torch::Tensor new_depth) {
        std::lock_guard<std::mutex> lock(mutex);
        // Shift observations back
        torch::Tensor shifted_depth = depth_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(include_history_steps - 1, torch::indexing::None),
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(torch::indexing::None)
        }).clone();
        
        depth_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(0, include_history_steps - 1),
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(torch::indexing::None)
        }) = shifted_depth;

        // Add new observation
        depth_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(-1),
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(torch::indexing::None)
        }) = new_depth;
    }

    torch::Tensor get_depth_vec() {
        return depth_buf;
    }

private:
    int num_envs;
    int height;
    int width;
    int include_history_steps;
    torch::Tensor depth_buf;
    std::mutex mutex;
};

class DepthTest {
public:
    DepthTest() 
        : depth_buffer(1, 60, 86, 2) {  // Initialize in constructor's initialization list
        // 初始化 RealSense
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);
    }

    ~DepthTest() {
        pipe.stop();
    }

    void run() {
        int frame_count = 0;
        while (frame_count < 100) {  // 测试100帧
            try {
                rs2::frameset frames = pipe.wait_for_frames(100);
                if (frames) {
                    rs2::depth_frame depth = frames.get_depth_frame();
                    
                    // 打印原始深度图信息
                    std::cout << "\n帧 " << frame_count << " 信息:" << std::endl;
                    std::cout << "原始深度图尺寸: " << depth.get_width() << "x" << depth.get_height() << std::endl;
                    
                    // 处理深度图数据
                    torch::Tensor processed_depth = torch::zeros({60, 86});
                    
                    // 获取原始深度数据
                    std::vector<uint16_t> depth_data;
                    depth_data.reserve(depth.get_width() * depth.get_height());
                    const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(depth.get_data());
                    
                    for (int i = 0; i < depth.get_width() * depth.get_height(); i++) {
                        depth_data.push_back(data_ptr[i]);
                    }
                    
                    // 创建深度tensor
                    torch::Tensor depth_tensor = torch::from_blob(depth_data.data(), 
                        {depth.get_height(), depth.get_width()}, torch::kInt16).clone();
                    
                    // 打印原始深度值范围
                    std::cout << "原始深度值范围: [" << depth_tensor.min().item<int16_t>() << ", " 
                              << depth_tensor.max().item<int16_t>() << "]" << std::endl;
                    
                    // 转换为float类型并转换为米
                    depth_tensor = depth_tensor.to(torch::kFloat32) / 1000.0;  // 转换为米
                    
                    // 打印转换为米后的范围
                    std::cout << "转换为米后的范围: [" << depth_tensor.min().item<float>() << ", " 
                              << depth_tensor.max().item<float>() << "]" << std::endl;
                    
                    // 将深度值裁剪到0.2-2.0米范围
                    depth_tensor = torch::clamp(depth_tensor, 0.2, 2.0);
                    
                    // 打印裁剪后的范围
                    std::cout << "裁剪后的深度范围(米): [" << depth_tensor.min().item<float>() << ", " 
                              << depth_tensor.max().item<float>() << "]" << std::endl;
                    
                    // 归一化到-0.5到0.5范围
                    depth_tensor = (depth_tensor - 1) / 2;  // (x - 1.1) / 1.8 将0.2-2.0映射到-0.5-0.5
                    
                    // 打印归一化后的范围
                    std::cout << "归一化后的范围: [" << depth_tensor.min().item<float>() << ", " 
                              << depth_tensor.max().item<float>() << "]" << std::endl;
                    
                    // 调整大小到目标尺寸
                    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0);  // 添加batch和channel维度
                    depth_tensor = torch::nn::functional::interpolate(
                        depth_tensor,
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{60, 86})
                            .mode(torch::kBilinear)
                            .align_corners(false)
                    );
                    
                    // 打印调整后的深度值范围
                    std::cout << "调整后的深度值范围: [" << depth_tensor.min().item<float>() << ", " 
                              << depth_tensor.max().item<float>() << "]" << std::endl;
                    
                    processed_depth = depth_tensor.squeeze(0).squeeze(0);  // 移除batch和channel维度
                    
                    // 打印处理后的深度图信息
                    std::cout << "处理后的深度图尺寸: " << processed_depth.sizes() << std::endl;
                    std::cout << "处理后的深度值范围: [" << processed_depth.min().item<float>() 
                              << ", " << processed_depth.max().item<float>() << "] 米" << std::endl;
                    
                    // 打印中心点及其周围的值
                    int center_i = 30;
                    int center_j = 43;
                    std::cout << "中心区域深度值:" << std::endl;
                    for(int i = center_i-1; i <= center_i+1; i++) {
                        for(int j = center_j-1; j <= center_j+1; j++) {
                            std::cout << processed_depth[i][j].item<float>() << " ";
                        }
                        std::cout << std::endl;
                    }
                    
                    depth_buffer.insert(processed_depth.unsqueeze(0));
                    
                    // 获取并打印buffer中的深度图
                    torch::Tensor depth_vec = depth_buffer.get_depth_vec();
                    std::cout << "Buffer中的深度图形状: " << depth_vec.sizes() << std::endl;
                    std::cout << "Buffer中的深度值范围: [" << depth_vec.min().item<float>() 
                              << ", " << depth_vec.max().item<float>() << "] 米" << std::endl;
                    
                    frame_count++;
                }
            }
            catch (const rs2::error& e) {
                std::cerr << "RealSense error: " << e.what() << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

private:
    rs2::pipeline pipe;
    rs2::config cfg;
    DepthBuffer depth_buffer;
};

int main() {
    try {
        DepthTest test;
        test.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}