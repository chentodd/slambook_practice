cmake_minimum_required(VERSION 2.8.3)
project(dense_mapping)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})

add_executable(denseMono denseMono.cpp)
target_link_libraries(denseMono ${THIRD_PARTY_LIBS})

add_executable(pointCloudMapping pointCloudMapping.cpp)
target_link_libraries(pointCloudMapping ${OpenCV_LIBS} ${PCL_LIBRARIES})