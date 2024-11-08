#===============================================================================
# Setup

cmake_minimum_required(VERSION 3.20)
# Requirement to use 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

project(centralized_ai)

#===============================================================================
# find_package, include etc. Lastly add_subdirectory.

# Set to the absolute path of .../libtorch/share/cmake/Torch in your libtorch directory
set(CMAKE_PREFIX_PATH "/home/viktor/PRO/libtorch/share/cmake/Torch")

find_package(Protobuf REQUIRED)
find_package(GTest REQUIRED)
find_package(Torch REQUIRED)

#===============================================================================
# add_subdirectories must be last

add_subdirectory(src)
#add_subdirectory(test)

#===============================================================================
