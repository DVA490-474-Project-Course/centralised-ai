#===============================================================================
# Setup
cmake_minimum_required(VERSION 3.20)
# Requirement to use 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

project(robot_controller_interface)

#===============================================================================
# find_package, include etc. Lastly add_subdirectory.
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#===============================================================================
# add_subdirectories must be last

add_subdirectory(src)
add_subdirectory(test)

#find_package(Torch REQUIRED)
#===============================================================================