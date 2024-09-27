cmake_minimum_required(VERSION 3.10)
project(WorldTest)

# Enable testing
enable_testing()

# Find GTest
find_package(GTest REQUIRED)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.4  # You can specify the latest version here
)
FetchContent_MakeAvailable(pybind11)

# Include directories
include_directories(../src/collective-robot-behaviour)

# Add your source files
set(SOURCE_FILES
../src/collective-robot-behaviour/world.cc)


# Add your test files
set(TEST_FILES
   collective-robot-behaviour-test/world_test.cc
)

# Create an executable for the tests
add_executable(world_test ${SOURCE_FILES} ${TEST_FILES})

# Link against GTest, Pybind11, and pthread
target_link_libraries(world_test GTest::GTest GTest::Main pybind11::module pthread)

# Add a test
add_test(NAME WorldTest COMMAND world_test)

