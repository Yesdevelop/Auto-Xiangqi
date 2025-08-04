# 指定最小 CMake 版本
cmake_minimum_required(VERSION 3.14)

# 设置项目名称
project(Chess98 LANGUAGES CXX)

# 启用 C++17 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 仅在 Release 构建时启用 O3 优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

# 只收集当前目录下的 .cpp 和 .hpp 文件（不递归子目录）
file(GLOB SOURCE_CPP_FILES "Chess98/*.cpp")
file(GLOB SOURCE_HPP_FILES "Chess98/*.hpp")

# 添加可执行文件
add_executable(Chess98 ${SOURCE_CPP_FILES} ${SOURCE_HPP_FILES})

# 添加当前目录为头文件搜索路径
target_include_directories(Chess98 PRIVATE Chess98)

# 要求如果有动态链接库, 用static方式编译
# 使用静态链接库
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

# 如果你知道具体需要静态链接的库, 也可以单独指定它们。
# 例如, 如果需要静态链接 pthread 库：
# find_library(PTHREAD_LIB pthread REQUIRED)
# target_link_libraries(Chess98 PRIVATE ${PTHREAD_LIB})
