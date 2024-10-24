
find_package(Torch REQUIRED) #move in root cmakelist
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PROJECT_HEADERS
        Communication.h
        network.h
)
add_executable(AI_model
        main.cc)

add_library(network_lib network.cc)

add_library(communication_lib Communication.cc
        MAPPO.cc
        MAPPO.h)
target_link_libraries(communication_lib network_lib "${TORCH_LIBRARIES}")
target_link_libraries(network_lib communication_lib "${TORCH_LIBRARIES}")

target_link_libraries(AI_model communication_lib network_lib "${TORCH_LIBRARIES}")
set_property(TARGET AI_model PROPERTY CXX_STANDARD 17)

# Include source files
target_sources(AI_model PRIVATE main.cc)

