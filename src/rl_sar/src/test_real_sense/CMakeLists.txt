cmake_minimum_required(VERSION 3.10)
project(depth_test)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(realsense2 REQUIRED)
set(Torch_DIR /home/nvidia/.local/lib/python3.10/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

# 添加可执行文件
add_executable(test_depth test_realsense.cpp)

# 链接库
target_link_libraries(test_depth
    PRIVATE
    realsense2::realsense2
    ${TORCH_LIBRARIES}
)

# 包含头文件目录
target_include_directories(test_depth
    PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

# 设置编译选项
if(MSVC)
    target_compile_options(test_depth PRIVATE /W4)
else()
    target_compile_options(test_depth PRIVATE -Wall -Wextra)
endif()
