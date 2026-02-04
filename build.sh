# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -e

# ========================
# Configuration
# ========================

# Color definitions
COLOR_ERROR='\033[0;31m'     # Red
COLOR_SUCCESS='\033[0;32m'   # Green
COLOR_WARNING='\033[1;33m'   # Yellow
COLOR_INFO='\033[0;34m'      # Blue
COLOR_DEBUG='\033[0;36m'     # Cyan
COLOR_RESET='\033[0m'        # Reset

# ========================
# Helper Functions
# ========================

print_separator() {
    echo -e "${COLOR_INFO}-------------------------------------------------------------------${COLOR_RESET}"
}

print_header() {
    print_separator
    echo -e "${COLOR_INFO}$1${COLOR_RESET}"
}

print_success() {
    echo -e "${COLOR_SUCCESS}$1${COLOR_RESET}"
}

print_warning() {
    echo -e "${COLOR_WARNING}$1${COLOR_RESET}"
}

print_error() {
    echo -e "${COLOR_ERROR}$1${COLOR_RESET}"
}

print_info() {
    echo -e "${COLOR_INFO}$1${COLOR_RESET}"
}

ask_confirmation() {
    local message="$1"
    echo -e -n "${COLOR_WARNING}$message (y/n): ${COLOR_RESET}"
    read -r response
    case "$response" in
        [yY]) return 0 ;;
        [nN]) return 1 ;;
        *)
            print_error "Please enter 'y' or 'n'"
            ask_confirmation "$message"
            ;;
    esac
}

# ========================
# Build Functions
# ========================

run_cmake_build() {
    print_header "[Running CMake Build]"
    print_warning "NOTE: CMake build is for hardware deployment only, not for simulation."
    print_separator

    cmake src/rl_sar/ -B cmake_build \
        -DUSE_CMAKE=ON \
        -DTorch_DIR=/home/ysc/.local/lib/python3.8/site-packages/torch/share/cmake/Torch \
        -DPython3_EXECUTABLE=/usr/bin/python3.8 \
        -DPython3_INCLUDE_DIR=/usr/include/python3.8 \
        -DPython3_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.8.so
    
    cmake --build cmake_build -j4

    print_success "CMake build completed!"
}

run_ros_build() {
    local packages=("$@")
    local package_list=$(IFS=' '; echo "${packages[*]}")

    print_header "[Running ROS Build]"

    # Clean existing symlinks
    clean_existing_symlinks "${packages[@]}"

    # Detect incompatible artifacts
    detect_incompatible_build_artifacts

    # Create appropriate symlinks
    if [ ${#packages[@]} -eq 0 ]; then
        create_symlinks_for_all_packages
    else
        create_symlinks_for_specific_packages "${packages[@]}"
    fi

    # 路径硬编码修复
    local EXTRA_CMAKE_ARGS="-DTorch_DIR=/home/ysc/.local/lib/python3.8/site-packages/torch/share/cmake/Torch -DPython3_EXECUTABLE=/usr/bin/python3.8 -DPython3_INCLUDE_DIR=/usr/include/python3.8 -DPython3_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.8.so"

    # Execute build
    if [ ${#packages[@]} -eq 0 ]; then
        if [[ "$ROS_DISTRO" == "noetic" ]]; then
            print_header "[Using catkin build]"
            print_info "Building all packages..."
            catkin build
        else
            print_header "[Using colcon build]"
            print_info "Building all packages..."
            colcon build --merge-install --symlink-install --cmake-args $EXTRA_CMAKE_ARGS
        fi
    else
        if [[ "$ROS_DISTRO" == "noetic" ]]; then
            print_header "[Using catkin build]"
            print_info "Building specific packages: $package_list"
            catkin build $package_list
        else
            print_header "[Using colcon build]"
            print_info "Building specific packages: $package_list"
            colcon build --merge-install --symlink-install --packages-select $package_list --cmake-args $EXTRA_CMAKE_ARGS
        fi
    fi

    print_success "ROS build completed!"
}

# ========================
# Clean Functions
# ========================

clean_workspace() {
    local packages=("$@")

    print_header "[Cleaning Workspace]"

    print_info "Cleaning build artifacts..."
    rm -rf build/ devel/ install/ log/ logs/ .catkin_tools/

    print_success "Clean completed!"
}

clean_existing_symlinks() {
    local packages=("$@")

    print_header "[Cleaning Existing Symlinks]"

    if [ ${#packages[@]} -eq 0 ]; then
        print_info "Removing all existing package.xml symlinks..."
        find src -name "package.xml" -type l -delete
        print_success "Removed all existing symlinks"
    else
        print_info "Removing existing symlinks for specified packages..."
        removed_packages=()
        for package_name in "${packages[@]}"; do
            package_dir=$(find src -name "$package_name" -type d | head -n 1)
            if [ -n "$package_dir" ] && [ -L "$package_dir/package.xml" ]; then
                rm -f "$package_dir/package.xml"
                removed_packages+=("$package_name")
            fi
        done
        print_success "Removed existing symlinks from: ${removed_packages[*]}"
    fi
}

# ========================
# ROS Specific Functions
# ========================

detect_incompatible_build_artifacts() {
    print_header "[Checking for Incompatible Build Artifacts]"
    local needs_cleanup=false
    if [[ "$ROS_DISTRO" != "noetic" ]]; then
        if [ -d "devel" ] || [ -d ".catkin_tools" ]; then
            needs_cleanup=true
        fi
    fi
    if [ "$needs_cleanup" = true ]; then
        clean_workspace
    else
        print_success "No incompatible build artifacts found"
    fi
}

create_symlinks_for_package() {
    local package_dir="$1"
    if [ -d "$package_dir" ]; then
        if [ -f "$package_dir/package.ros1.xml" ] && [ -f "$package_dir/package.ros2.xml" ]; then
            [ -e "$package_dir/package.xml" ] && rm -f "$package_dir/package.xml"
            if [[ "$ROS_DISTRO" == "noetic" ]]; then
                ln -s package.ros1.xml "$package_dir/package.xml"
                return 0
            elif [[ "$ROS_DISTRO" == "foxy" || "$ROS_DISTRO" == "humble" ]]; then
                ln -s package.ros2.xml "$package_dir/package.xml"
                return 0
            fi
        fi
    fi
    return 1
}

create_symlinks_for_all_packages() {
    print_header "[Creating Symlinks for All Packages]"
    created_packages=()
    while IFS= read -r -d '' package_dir; do
        package_dir=$(dirname "$package_dir")
        package_name=$(basename "$package_dir")
        if create_symlinks_for_package "$package_dir"; then
            created_packages+=("$package_name")
        fi
    done < <(find src -name "package.ros1.xml" -print0)
    print_success "Created symlinks for: ${created_packages[*]}"
}

create_symlinks_for_specific_packages() {
    local packages=("$@")
    print_header "[Creating Symlinks for Specific Packages]"
    created_packages=()
    for package_name in "${packages[@]}"; do
        package_dir=$(find src -name "$package_name" -type d | head -n 1)
        if [ -n "$package_dir" ] && create_symlinks_for_package "$package_dir"; then
            created_packages+=("$package_name")
        fi
    done
}

show_usage() {
    echo "Usage: $0 [OPTIONS] [PACKAGE_NAMES...]"
    echo "  -c, --clean    Clean workspace"
    echo "  -m, --cmake    Build using CMake"
}

main() {
    local packages=()
    local clean_mode=false
    local cmake_mode=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--clean) clean_mode=true; shift ;;
            -m|--cmake) cmake_mode=true; shift ;;
            -h|--help) show_usage; exit 0 ;;
            *) packages+=("$1"); shift ;;
        esac
    done
    if [ "$cmake_mode" = true ]; then run_cmake_build; exit 0; fi
    if [ "$clean_mode" = true ]; then clean_workspace "${packages[@]}"; exit 0; fi
    if [ -z "$ROS_DISTRO" ]; then print_error "Source ROS first!"; exit 1; fi
    run_ros_build "${packages[@]}"
}

main "$@"
