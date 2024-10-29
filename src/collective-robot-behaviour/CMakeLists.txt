#===============================================================================
# World

add_library(world_lib world.cc)
#===============================================================================
# MLP

add_library(mlp_lib mlp.cc)
#===============================================================================
# Utils

add_library(utils_lib utils.cc)
target_link_libraries(utils_lib "${TORCH_LIBRARIES}")
#===============================================================================
add_library(mappo_lib  network.cc communication.cc mappo.cc)
target_link_libraries(mappo_lib "${TORCH_LIBRARIES}")
