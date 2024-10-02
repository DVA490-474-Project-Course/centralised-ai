cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

project(centralized_ai)

add_subdirectory(src)

# add .cc files to executable
add_executable(ssl_vision_client_demo src/ssl-interface/ssl_vision_client_demo.cc)
add_executable(ssl_game_controller_client_demo src/ssl-interface/ssl_game_controller_client_demo.cc)

# include header files
target_include_directories(ssl_vision_client_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ssl-interface)
target_include_directories(ssl_game_controller_client_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ssl-interface)

# link libraries
target_link_libraries(ssl_vision_client_demo ssl_interface_lib)
target_link_libraries(ssl_game_controller_client_demo ssl_interface_lib)

#____________________________________________________________
#googleTest
#_____________________________________________________________
# Add GoogleTest manually
#add_subdirectory(/home/aaiza/googletest ${CMAKE_BINARY_DIR}/googletest-build)


# Fetch and Add GoogleTest
include(FetchContent)

# Declare GoogleTest as a dependency and download it automatically
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)

# Make GoogleTest available
FetchContent_MakeAvailable(googletest)

# Add your test executable
add_executable(ssl_vision_client_test src/ssl-interface/ssl_vision_client.cc test/ssl-interface/ssl_vision_client_test.cc
)

# Include GoogleTest headers
target_include_directories(ssl_vision_client_test PRIVATE 
    ${gtest_SOURCE_DIR}/include
    ${gmock_SOURCE_DIR}/include
)

target_include_directories(ssl_vision_client_test PUBLIC src/ssl-interface/generated src/ssl-interface/)

# Link GoogleTest and GoogleTest main library
target_link_libraries(ssl_vision_client_test gtest_main gmock ssl_interface_lib)
# Enable testing
enable_testing()

# Add a test to be run with `ctest`
add_test(NAME ssl_vision_client_ctest COMMAND ssl_vision_client_test)
