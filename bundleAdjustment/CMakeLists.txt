cmake_minimum_required(VERSION 2.8.3)
project(bundle_adjustment)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_library(balCommon common.cpp)
# add_executable(bundleAdjustmentG2o bundleAdjustmentG2o.cpp)
add_executable(bundleAdjustmentCeres bundleAdjustmentCeres.cpp)

target_link_libraries(bundleAdjustmentCeres ${CERES_LIBRARIES} balCommon)
#target_link_libraries(bundleAdjustmentG2o ${G2O_LIBS} bal_common)
