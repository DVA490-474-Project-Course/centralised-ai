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
    "${TORCH_LIBRARIES}"
)

target_include_directories(main_exe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

#===============================================================================