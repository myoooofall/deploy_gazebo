cmake_minimum_required(VERSION 3.5)
project(rl_sar VERSION 3.0.0 LANGUAGES CXX)

set(USE_CMAKE OFF CACHE BOOL "Use Cmake build system")
message(STATUS "USE_CMAKE: ${USE_CMAKE}")
if(USE_CMAKE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

if(NOT USE_CMAKE)
    if($ENV{ROS_DISTRO} MATCHES "noetic")
        add_compile_definitions(USE_ROS1)
    elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")
        add_compile_definitions(USE_ROS2)
    endif()
else()
    message(WARNING "ROS_DISTRO not set, assuming non-ROS mode")
    add_compile_definitions(USE_CMAKE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
if(UPPER_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_options(-g)
endif()

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT USE_CMAKE)
    if($ENV{ROS_DISTRO} MATCHES "noetic")
        find_package(catkin REQUIRED COMPONENTS
            controller_manager genmsg joint_state_controller robot_state_publisher roscpp gazebo_ros std_msgs tf geometry_msgs robot_msgs robot_joint_controller rospy)
        catkin_package(CATKIN_DEPENDS robot_joint_controller rospy)
    elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")
        find_package(ament_cmake REQUIRED)
        find_package(joint_state_broadcaster REQUIRED)
        find_package(robot_state_publisher REQUIRED)
        find_package(rclcpp REQUIRED)
        find_package(gazebo_ros REQUIRED)
        find_package(std_msgs REQUIRED)
        find_package(robot_msgs REQUIRED)
        find_package(robot_joint_controller REQUIRED)
        find_package(rclpy REQUIRED)
        find_package(gazebo_msgs REQUIRED)
        find_package(std_srvs REQUIRED)
        find_package(geometry_msgs REQUIRED)
        find_package(sensor_msgs REQUIRED)
        find_package(nav_msgs REQUIRED)
        find_package(visualization_msgs REQUIRED)
        find_package(cv_bridge REQUIRED)
        ament_package()
    endif()
    add_compile_options(${GAZEBO_CXX_FLAGS})
    find_package(gazebo REQUIRED)
endif()

# LibTorch & Python (NX Fixed Path)
set(Torch_DIR /home/ysc/.local/lib/python3.8/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

link_directories(/usr/local/lib)
include_directories(${YAML_CPP_INCLUDE_DIR})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ARCH_DIR "aarch64")
    set(UNITREE_LIB "libunitree_legged_sdk_arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(ARCH_DIR "x86_64")
    set(UNITREE_LIB "libunitree_legged_sdk_amd64")
endif()

# --- SDK1 (A1/Aliengo) - 保持导入状态 ---
add_library(unitree_legged_sdk SHARED IMPORTED)
set_target_properties(unitree_legged_sdk PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/unitree_legged_sdk-3.2/include"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/unitree_legged_sdk-3.2/lib/${UNITREE_LIB}.so"
)

# --- SDK2 (Go2/G1) - 彻底屏蔽，防止 DDS 冲突 ---
# set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples build for unitree_sdk2" FORCE)
# add_subdirectory(library/thirdparty/unitree_sdk2-2.0.0)

# --- Lite3_MotionSDK ---
add_library(lite3_motionsdk SHARED IMPORTED)
set_target_properties(lite3_motionsdk PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/Lite3_MotionSDK/include"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/Lite3_MotionSDK/lib/libdeeprobotics_legged_sdk_${ARCH_DIR}.so"
)
set(LITE3_REAL_LIBS Threads::Threads lite3_motionsdk)

# Gamepad Source
set(GAMEPAD_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/gamepad/src/gamepad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/gamepad/src/retroid_gamepad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/gamepad/src/udp_receiver.cpp
)

include_directories(
    include library/core/matplotlibcpp library/core/observation_buffer library/core/rl_sdk library/core/loop library/core/fsm policy
    ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/gamepad/include
)

# Core Library: rl_sdk
add_library(rl_sdk library/core/rl_sdk/rl_sdk.cpp)
target_link_libraries(rl_sdk PUBLIC "${TORCH_LIBRARIES}" Python3::Python TBB::tbb)
ament_target_dependencies(rl_sdk PUBLIC cv_bridge rclcpp sensor_msgs)

# Core Library: observation_buffer
add_library(observation_buffer library/core/observation_buffer/observation_buffer.cpp library/core/observation_buffer/depth_buffer.cpp)
target_link_libraries(observation_buffer "${TORCH_LIBRARIES}")
ament_target_dependencies(observation_buffer rclcpp sensor_msgs cv_bridge)

# --- Target 1: Simulation (已修复 std_srvs 链接) ---
if(NOT USE_CMAKE)
    add_executable(rl_sim src/rl_sim.cpp)
    target_link_libraries(rl_sim rl_sdk observation_buffer yaml-cpp Threads::Threads
                          std_srvs::std_srvs__rosidl_typesupport_cpp 
                          std_srvs::std_srvs__rosidl_generator_cpp)
    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")
        ament_target_dependencies(rl_sim 
            joint_state_broadcaster robot_state_publisher rclcpp gazebo_ros 
            std_msgs robot_msgs robot_joint_controller gazebo_msgs geometry_msgs nav_msgs std_srvs)
        install(TARGETS rl_sim DESTINATION lib/${PROJECT_NAME})
    endif()
endif()

# --- Target 2: Lite3 Real ---
add_executable(rl_real_lite3 src/rl_real_lite3.cpp ${GAMEPAD_SRC})
target_link_libraries(rl_real_lite3 ${LITE3_REAL_LIBS} rl_sdk observation_buffer yaml-cpp)
if($ENV{ROS_DISTRO} MATCHES "foxy|humble")
    ament_target_dependencies(rl_real_lite3 rclcpp geometry_msgs sensor_msgs nav_msgs visualization_msgs)
    install(TARGETS rl_real_lite3 DESTINATION lib/${PROJECT_NAME})
endif()

# --- 下面是非 Lite3 实体部分，已全部注释 ---
# add_executable(rl_real_a1 src/rl_real_a1.cpp)
# add_executable(rl_real_go2 src/rl_real_go2.cpp)
# add_executable(rl_real_g1 src/rl_real_g1.cpp)

# --- 脚本和资源安装 (已补全 generate_terrain_world.py) ---
if(NOT USE_CMAKE AND $ENV{ROS_DISTRO} MATCHES "foxy|humble")
    install(PROGRAMS 
        scripts/actuator_net.py 
        scripts/nav_goal_dialog.py 
        scripts/generate_terrain_world.py # <--- 补上了这个
        scripts/offline/record_lite3_raw_bag.sh
        scripts/offline/replay_localize_and_record.sh
        scripts/offline/export_goal_metrics_from_bag.py
        scripts/offline/pull_bag_from_nx.sh
        scripts/benchmark/record_nx_power.py
        scripts/benchmark/summarize_guide_latency.py
        DESTINATION lib/${PROJECT_NAME}
    )
    install(DIRECTORY launch worlds policy DESTINATION share/${PROJECT_NAME})
endif()
