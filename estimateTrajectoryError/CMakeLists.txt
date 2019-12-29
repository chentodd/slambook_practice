cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

find_package(GLEW REQUIRED Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
include_directories(
    "/usr/include/eigen3"
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)
add_executable(estimateTrajectoryError estimateTrajectoryError.cpp)
target_link_libraries(estimateTrajectoryError ${Pangolin_LIBRARIES})