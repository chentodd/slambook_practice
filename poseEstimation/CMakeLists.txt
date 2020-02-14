cmake_minimum_required(VERSION 2.8.3)
project(poseEstimation)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)

find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
)

add_executable(poseEstimation2d2d poseEstimation2d2d.cpp)
target_link_libraries(poseEstimation2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(poseEstimation3d2d poseEstimation3d2d.cpp)
target_link_libraries(poseEstimation3d2d g2o_core g2o_stuff ${OpenCV_LIBS})

add_executable(poseEstimation3d3d poseEstimation3d3d.cpp)
target_link_libraries(poseEstimation3d3d g2o_core g2o_stuff ${OpenCV_LIBS})