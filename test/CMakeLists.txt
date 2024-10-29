 #===============================================================================
# Includes

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

enable_testing()

#===============================================================================
# Setup gtest with main_test

add_executable(main_test_exe main_test.cc)

# !!! Add any libraries (MADE BY YOU!) that your tests use here !!!
target_link_libraries(
  main_test_exe
  GTest::gmock_main
  ssl_interface_lib
  world_lib
  mlp_lib
  utils_lib
)

#===============================================================================
# Dependencies

# Test source files
target_sources(main_test_exe PRIVATE
  main_test.cc
  collective-robot-behaviour-test/world_test.cc
  collective-robot-behaviour-test/mlp_test.cc
  collective-robot-behaviour-test/utils_test.cc
  ssl-interface-test/ssl_game_controller_client_test.cc
  ssl-interface-test/ssl_vision_client_test.cc
)

#===============================================================================
