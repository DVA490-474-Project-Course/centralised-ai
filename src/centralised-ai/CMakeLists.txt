
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PROJECT_HEADERS
        communication.h
        network.h
)
add_executable(AI_model demo_run.cc)

add_library(mappo_lib mappo.cc)
add_library(network_lib network.cc)
add_library(communication_lib communication.cc mappo.cc)

target_link_libraries(communication_lib network_lib "${TORCH_LIBRARIES}")
target_link_libraries(network_lib communication_lib "${TORCH_LIBRARIES}")
target_link_libraries(mappo_lib communication_lib network_lib "${TORCH_LIBRARIES}")
target_link_libraries(AI_model communication_lib network_lib "${TORCH_LIBRARIES}")

# Include source files
target_sources(AI_model PRIVATE demo_run.cc)

