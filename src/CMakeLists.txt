#===============================================================================
# Includes

#===============================================================================
# Add which modules/directories contains stuff that should be compiled

add_executable(main_exe main.cc)

# Where to find source code for libraries etc
add_subdirectory(collective-robot-behaviour)
add_subdirectory(ssl-interface)
add_subdirectory(ai-module)

#===============================================================================
# Dependencies

# Libraries used by main
target_link_libraries(main_exe
    world_lib
    mlp_lib
    utils_lib
)

#===============================================================================
