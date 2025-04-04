# Minimum CMake version required
cmake_minimum_required(VERSION 3.16)

# Project name and language
project(cpp_imageNN LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the Eigen library include path
include_directories("C:/VsLibs/eigen-3.4.0")

# Add the Qt library include path (if needed)
include_directories("C:/VsLibs/qt-everywhere-src-6.9.0/include")

# Add the source files
add_executable(cpp_imageNN main.cpp)

# Link Qt libraries (if needed)
#find_package(Qt6 REQUIRED COMPONENTS Widgets)
#target_link_libraries(cpp_imageNN Qt6::Widgets)