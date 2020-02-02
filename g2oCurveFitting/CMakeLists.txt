cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
)

add_executable(g2oCurveFitting
    g2oCurveFitting.cpp
)

target_link_libraries(g2oCurveFitting
    ${OpenCV_LIBS}
    g2o_core
    g2o_stuff
)