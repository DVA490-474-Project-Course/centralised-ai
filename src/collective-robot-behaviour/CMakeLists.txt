#===============================================================================
# World

add_library(world_lib world.cc)
target_link_libraries(world_lib "${TORCH_LIBRARIES}")
#===============================================================================
# Utils

add_library(utils_lib utils.cc)
target_link_libraries(utils_lib "${TORCH_LIBRARIES}")
#===============================================================================