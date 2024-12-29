#===============================================================================

add_library(mappo_lib network.cc communication.cc mappo.cc utils.cc run_state.cc reward.cc evaluation.cc)
target_link_libraries(mappo_lib "${TORCH_LIBRARIES}" simulation_interface_lib Python3::Python)

include_directories(../../external)

target_include_directories(mappo_lib PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

#===============================================================================