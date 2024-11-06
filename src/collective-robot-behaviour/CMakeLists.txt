#===============================================================================
add_library(mappo_lib  network.cc communication.cc  mappo.cc utils.cc run_state.cc world.cc)
target_link_libraries(mappo_lib "${TORCH_LIBRARIES}")
