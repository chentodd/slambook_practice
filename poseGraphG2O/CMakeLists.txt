cmake_minimum_required(VERSION 2.8.3)
project(poseGraphG2O)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

# Eigen
include_directories("/usr/include/eigen3")

# sophus 
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o 
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(poseGraphG2OSE3 poseGraphG2OSE3.cpp)
target_link_libraries(poseGraphG2OSE3
    g2o_core g2o_stuff 
    g2o_types_slam3d 
    ${CHOLMOD_LIBRARIES}
)

add_executable(poseGraphG2OLieAlgebra poseGraphG2OLieAlgebra.cpp)
target_link_libraries(poseGraphG2OLieAlgebra
    g2o_core g2o_stuff
    ${CHOLMOD_LIBRARIES}
    ${Sophus_LIBRARIES}
)

