cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

project(centralized_ai)

add_subdirectory(src)

# add .cc files to executable
file(GLOB_RECURSE SRC_FILES src/ssl-interface/*.cc)
add_executable(ssl_vision_client_demo ${SRC_FILES} src/main.cc)

# include header files
target_include_directories(ssl_vision_client_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ssl-interface)

# link libraries
target_link_libraries(ssl_vision_client_demo vision_client_lib)
