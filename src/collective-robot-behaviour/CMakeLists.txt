#===============================================================================
# World

add_library(world_lib world.cc)
target_include_directories(world_lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(world_lib "${TORCH_LIBRARIES}" c10 torch_cpu torch)
#===============================================================================
# Utils

add_library(utils_lib utils.cc)
target_link_libraries(utils_lib "${TORCH_LIBRARIES}" c10 torch_cpu torch)
#===============================================================================