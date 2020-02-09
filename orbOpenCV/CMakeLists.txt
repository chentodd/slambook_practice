cmake_minimum_required(VERSION 2.8.3)
project(orbOpenCV)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(orbOpenCV orbOpenCV.cpp)
target_link_libraries(orbOpenCV ${OpenCV_LIBS})

add_executable(orbSelf orbSelf.cpp)
target_link_libraries(orbSelf ${OpenCV_LIBS})