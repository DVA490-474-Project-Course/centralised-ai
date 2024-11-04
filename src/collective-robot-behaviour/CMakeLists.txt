#===============================================================================
# World

add_library(world_lib world.cc)
target_link_libraries(world_lib "${TORCH_LIBRARIES}")
#===============================================================================
# Utils

add_library(utils_lib utils.cc)
target_link_libraries(utils_lib "${TORCH_LIBRARIES}")
#===============================================================================
add_library(mappo_lib  network.cc communication.cc mappo.cc)
target_link_libraries(mappo_lib "${TORCH_LIBRARIES}")
