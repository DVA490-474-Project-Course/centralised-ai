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

add_executable(main_test_exe 
${CMAKE_CURRENT_SOURCE_DIR}/../src/ssl-interface/ssl_vision_client.cc 
${CMAKE_CURRENT_SOURCE_DIR}/main_test.cc
${CMAKE_CURRENT_SOURCE_DIR}/ssl-interface/ssl_vision_client_test.cc
)

# !!! Add any libraries (MADE BY YOU!) that your tests use here !!!
target_link_libraries(
  main_test_exe
  GTest::gtest_main
  GTest::gmock   # Link against Google Mock
  ssl_interface_lib
)


#===============================================================================
# Dependencies

target_include_directories(main_test_exe PRIVATE 
  "${CMAKE_CURRENT_SOURCE_DIR}/test" 
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/ssl-interface"
  "${gtest_SOURCE_DIR}/include"   # Include the gtest directory
  "${gmock_SOURCE_DIR}/include"   # Include the gmock directory
)


#===============================================================================
# Must be last (I think)

gtest_discover_tests(main_test_exe)
