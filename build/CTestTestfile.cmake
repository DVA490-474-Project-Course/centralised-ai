# CMake generated Testfile for 
# Source directory: /home/jacob/centralised-ai
# Build directory: /home/jacob/centralised-ai/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[collective_robot_behaviour_test]=] "/home/jacob/centralised-ai/bin/collective_robot_behaviour_test")
set_tests_properties([=[collective_robot_behaviour_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/jacob/centralised-ai/CMakeLists.txt;54;add_test;/home/jacob/centralised-ai/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("src")
subdirs("test")
