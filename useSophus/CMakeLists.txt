cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

find_package(Sophus REQUIRED) 
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(useSophus useSophus.cpp)
# target_link_libraries(drawTrajectory ${Pangolin_LIBRARIES})