# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jacob/centralised-ai

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jacob/centralised-ai/build

# Include any dependencies generated for this target.
include test/CMakeFiles/main_test_exe.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/main_test_exe.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/main_test_exe.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/main_test_exe.dir/flags.make

test/CMakeFiles/main_test_exe.dir/main_test.cc.o: test/CMakeFiles/main_test_exe.dir/flags.make
test/CMakeFiles/main_test_exe.dir/main_test.cc.o: ../test/main_test.cc
test/CMakeFiles/main_test_exe.dir/main_test.cc.o: test/CMakeFiles/main_test_exe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/main_test_exe.dir/main_test.cc.o"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/main_test_exe.dir/main_test.cc.o -MF CMakeFiles/main_test_exe.dir/main_test.cc.o.d -o CMakeFiles/main_test_exe.dir/main_test.cc.o -c /home/jacob/centralised-ai/test/main_test.cc

test/CMakeFiles/main_test_exe.dir/main_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_test_exe.dir/main_test.cc.i"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jacob/centralised-ai/test/main_test.cc > CMakeFiles/main_test_exe.dir/main_test.cc.i

test/CMakeFiles/main_test_exe.dir/main_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_test_exe.dir/main_test.cc.s"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jacob/centralised-ai/test/main_test.cc -o CMakeFiles/main_test_exe.dir/main_test.cc.s

test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o: test/CMakeFiles/main_test_exe.dir/flags.make
test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o: ../test/collective-robot-behaviour-test/world_test.cc
test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o: test/CMakeFiles/main_test_exe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o -MF CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o.d -o CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o -c /home/jacob/centralised-ai/test/collective-robot-behaviour-test/world_test.cc

test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.i"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jacob/centralised-ai/test/collective-robot-behaviour-test/world_test.cc > CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.i

test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.s"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jacob/centralised-ai/test/collective-robot-behaviour-test/world_test.cc -o CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.s

test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o: test/CMakeFiles/main_test_exe.dir/flags.make
test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o: ../test/collective-robot-behaviour-test/mlp_test.cc
test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o: test/CMakeFiles/main_test_exe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o -MF CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o.d -o CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o -c /home/jacob/centralised-ai/test/collective-robot-behaviour-test/mlp_test.cc

test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.i"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jacob/centralised-ai/test/collective-robot-behaviour-test/mlp_test.cc > CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.i

test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.s"
	cd /home/jacob/centralised-ai/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jacob/centralised-ai/test/collective-robot-behaviour-test/mlp_test.cc -o CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.s

# Object files for target main_test_exe
main_test_exe_OBJECTS = \
"CMakeFiles/main_test_exe.dir/main_test.cc.o" \
"CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o" \
"CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o"

# External object files for target main_test_exe
main_test_exe_EXTERNAL_OBJECTS =

../bin/main_test_exe: test/CMakeFiles/main_test_exe.dir/main_test.cc.o
../bin/main_test_exe: test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/world_test.cc.o
../bin/main_test_exe: test/CMakeFiles/main_test_exe.dir/collective-robot-behaviour-test/mlp_test.cc.o
../bin/main_test_exe: test/CMakeFiles/main_test_exe.dir/build.make
../bin/main_test_exe: /usr/lib/x86_64-linux-gnu/libgtest_main.a
../bin/main_test_exe: src/collective-robot-behaviour/libworld_lib.a
../bin/main_test_exe: src/collective-robot-behaviour/libmlp_lib.a
../bin/main_test_exe: /usr/lib/x86_64-linux-gnu/libgtest.a
../bin/main_test_exe: test/CMakeFiles/main_test_exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../../bin/main_test_exe"
	cd /home/jacob/centralised-ai/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main_test_exe.dir/link.txt --verbose=$(VERBOSE)
	cd /home/jacob/centralised-ai/build/test && /usr/bin/cmake -D TEST_TARGET=main_test_exe -D TEST_EXECUTABLE=/home/jacob/centralised-ai/bin/main_test_exe -D TEST_EXECUTOR= -D TEST_WORKING_DIR=/home/jacob/centralised-ai/build/test -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D TEST_FILTER= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=main_test_exe_TESTS -D CTEST_FILE=/home/jacob/centralised-ai/build/test/main_test_exe[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=5 -D TEST_XML_OUTPUT_DIR= -P /usr/share/cmake-3.22/Modules/GoogleTestAddTests.cmake

# Rule to build all files generated by this target.
test/CMakeFiles/main_test_exe.dir/build: ../bin/main_test_exe
.PHONY : test/CMakeFiles/main_test_exe.dir/build

test/CMakeFiles/main_test_exe.dir/clean:
	cd /home/jacob/centralised-ai/build/test && $(CMAKE_COMMAND) -P CMakeFiles/main_test_exe.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/main_test_exe.dir/clean

test/CMakeFiles/main_test_exe.dir/depend:
	cd /home/jacob/centralised-ai/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jacob/centralised-ai /home/jacob/centralised-ai/test /home/jacob/centralised-ai/build /home/jacob/centralised-ai/build/test /home/jacob/centralised-ai/build/test/CMakeFiles/main_test_exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/main_test_exe.dir/depend

