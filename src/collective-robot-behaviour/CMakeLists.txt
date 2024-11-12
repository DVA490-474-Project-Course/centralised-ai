#===============================================================================
add_library(mappo_lib  network.cc communication.cc  mappo.cc utils.cc world.cc run_state.cc)
target_link_libraries(mappo_lib "${TORCH_LIBRARIES}" simulation_interface_lib Python3::Python)

target_include_directories(mappo_lib PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})