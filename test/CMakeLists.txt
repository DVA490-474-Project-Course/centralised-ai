#===============================================================================
# Includes

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.4  # You can specify the latest version here
)
FetchContent_MakeAvailable(pybind11)

# Fetch googletest
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

# Libraries used by tests
target_link_libraries(
  main_test_exe
  GTest::gtest_main
  world_lib
  mlp_lib
)

#===============================================================================
# Dependencies

# Test source files
target_sources(main_test_exe PRIVATE
main_test
collective-robot-behaviour-test/world_test.cc
collective-robot-behaviour-test/mlp_test.cc
)

# Packages that must be included
target_include_directories(main_test_exe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test")

#===============================================================================
gtest_discover_tests(main_test_exe)

#===============================================================================