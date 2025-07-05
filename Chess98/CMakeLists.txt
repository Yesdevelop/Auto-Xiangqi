# 指定最小 CMake 版本
cmake_minimum_required(VERSION 3.14)

# 设置项目名称
project(Chess98 LANGUAGES CXX)

# 启用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 仅在 Release 构建时启用 O3 优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

# 只收集当前目录下的 .cpp 和 .hpp 文件（不递归子目录）
file(GLOB SOURCE_CPP_FILES "*.cpp")
file(GLOB SOURCE_HPP_FILES "*.hpp")

# 添加可执行文件
add_executable(Chess98 ${SOURCE_CPP_FILES} ${SOURCE_HPP_FILES})

# 添加当前目录为头文件搜索路径
target_include_directories(Chess98 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})