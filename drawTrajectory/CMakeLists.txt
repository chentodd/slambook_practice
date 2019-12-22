cmake_minimum_required(VERSION 2.8.30)
add_compile_options(-std=c++11)

find_package(GLEW REQUIRED) 
find_package(Pangolin REQUIRED)
include_directories(
    "/usr/include/eigen3"
    ${Pangolin_INCLUDE_DIRS}
)
add_executable(drawTrajectory drawTrajectory.cpp)
target_link_libraries(drawTrajectory ${Pangolin_LIBRARIES})