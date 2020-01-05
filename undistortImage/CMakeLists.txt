cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(undistortImage undistortImage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})