#===============================================================================
# Includes

# Fetch and include LibTorch
include(FetchContent)
FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
)
FetchContent_MakeAvailable(libtorch)

set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#===============================================================================
# Add which modules/directories contains stuff that should be compiled

add_executable(main_exe main.cc)

# Where to find source code for libraries etc
add_subdirectory(collective-robot-behaviour)

#===============================================================================
# Dependencies

# Libraries used by main
target_link_libraries(main_exe
    world_lib
    mlp_lib
    utils_lib
)


target_include_directories(main_exe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(main_exe PRIVATE "${TORCH_INCLUDE_DIRS}")

#===============================================================================