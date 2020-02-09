cmake_minimum_required(VERSION 2.8.3)
project(poseEstimation)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(poseEstimation2d2d poseEstimation2d2d.cpp)
target_link_libraries(poseEstimation2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})