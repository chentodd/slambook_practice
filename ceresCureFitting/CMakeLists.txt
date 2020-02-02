cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

# OpenCV
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

add_executable(
    ceresCurveFitting
    ceresCurveFitting.cpp
)

target_link_libraries(
    ceresCurveFitting
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)