cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(AI-Model)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(AI-Model main.cc)

target_link_libraries(AI-Model "${TORCH_LIBRARIES}")
set_property(TARGET AI-Model PROPERTY CXX_STANDARD 17)
# Include source files
target_sources(AI-Model PRIVATE
        main.cc
        Communication.h
        networks.h
)
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET AI-Model
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)