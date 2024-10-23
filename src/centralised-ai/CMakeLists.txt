
find_package(Torch REQUIRED) #move in root cmakelist
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PROJECT_HEADERS
        Communication.h
        network.h
)
add_executable(example-app
        main.cc)

add_library(network_lib network.cc)

add_library(communication_lib Communication.cc
        MAPPO.cc
        MAPPO.h)
target_link_libraries(communication_lib network_lib "${TORCH_LIBRARIES}")
target_link_libraries(network_lib communication_lib "${TORCH_LIBRARIES}")

target_link_libraries(example-app communication_lib network_lib "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# Include source files
target_sources(example-app PRIVATE main.cc)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)