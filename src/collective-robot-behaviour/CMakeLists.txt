#===============================================================================
# World

add_library(world_lib world.cc)
target_include_directories(world_lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

#===============================================================================
# MLP

add_library(mlp_lib mlp.cc)
target_include_directories(mlp_lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
#===============================================================================
# Utils

add_library(utils_lib utils.cc)
target_link_libraries(utils_lib ${TORCH_LIBRARIES})
target_include_directories(mlp_lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
#===============================================================================