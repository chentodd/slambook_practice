cmake_minimum_required(VERSION 2.8.3)
project(loop_closure)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

# opencv 
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dbow3  
set(DBoW3_INCLUDE_DIRS "/usr/local/include")

add_executable(trainVocabulary trainVocabulary.cpp)
target_link_libraries(trainVocabulary ${OpenCV_LIBS} DBoW3)

add_executable(calculateDifference calculateDifference.cpp )
target_link_libraries(calculateDifference ${OpenCV_LIBS} DBoW3)

# add_executable( gen_vocab gen_vocab_large.cpp )
# target_link_libraries( gen_vocab ${OpenCV_LIBS} ${DBoW3_LIBS} )