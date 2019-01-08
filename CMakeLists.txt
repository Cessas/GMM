cmake_minimum_required(VERSION 3.9)
project(untitled2)

set(CMAKE_CXX_STANDARD 11)
set(EIGEN_DIR "C:\\Eigen3")
include_directories(${EIGEN_DIR})
add_executable(untitled2 main.cpp)
target_link_libraries(untitled2 ${EIGEN_DIR})