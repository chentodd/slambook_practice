cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

add_executable(rgbdVision rgbdVision.cpp)
target_link_libraries(rgbdVision 
    ${OpenCV_LIBS} 
    ${Pangolin_LIBRARIES}
)