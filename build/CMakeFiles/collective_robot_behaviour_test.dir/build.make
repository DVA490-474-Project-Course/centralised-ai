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
include CMakeFiles/collective_robot_behaviour_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/collective_robot_behaviour_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/collective_robot_behaviour_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/collective_robot_behaviour_test.dir/flags.make

CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o: CMakeFiles/collective_robot_behaviour_test.dir/flags.make
CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o: ../test/collective-robot-behaviour-test/mlp_test.cc
CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o: CMakeFiles/collective_robot_behaviour_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o -MF CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o.d -o CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o -c /home/jacob/centralised-ai/test/collective-robot-behaviour-test/mlp_test.cc

CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jacob/centralised-ai/test/collective-robot-behaviour-test/mlp_test.cc > CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.i

CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jacob/centralised-ai/test/collective-robot-behaviour-test/mlp_test.cc -o CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.s

CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o: CMakeFiles/collective_robot_behaviour_test.dir/flags.make
CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o: ../test/collective-robot-behaviour-test/world_test.cc
CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o: CMakeFiles/collective_robot_behaviour_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o -MF CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o.d -o CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o -c /home/jacob/centralised-ai/test/collective-robot-behaviour-test/world_test.cc

CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jacob/centralised-ai/test/collective-robot-behaviour-test/world_test.cc > CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.i

CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jacob/centralised-ai/test/collective-robot-behaviour-test/world_test.cc -o CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.s

# Object files for target collective_robot_behaviour_test
collective_robot_behaviour_test_OBJECTS = \
"CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o" \
"CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o"

# External object files for target collective_robot_behaviour_test
collective_robot_behaviour_test_EXTERNAL_OBJECTS =

../bin/collective_robot_behaviour_test: CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/mlp_test.cc.o
../bin/collective_robot_behaviour_test: CMakeFiles/collective_robot_behaviour_test.dir/test/collective-robot-behaviour-test/world_test.cc.o
../bin/collective_robot_behaviour_test: CMakeFiles/collective_robot_behaviour_test.dir/build.make
../bin/collective_robot_behaviour_test: CMakeFiles/collective_robot_behaviour_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jacob/centralised-ai/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/collective_robot_behaviour_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/collective_robot_behaviour_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/collective_robot_behaviour_test.dir/build: ../bin/collective_robot_behaviour_test
.PHONY : CMakeFiles/collective_robot_behaviour_test.dir/build

CMakeFiles/collective_robot_behaviour_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/collective_robot_behaviour_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/collective_robot_behaviour_test.dir/clean

CMakeFiles/collective_robot_behaviour_test.dir/depend:
	cd /home/jacob/centralised-ai/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jacob/centralised-ai /home/jacob/centralised-ai /home/jacob/centralised-ai/build /home/jacob/centralised-ai/build /home/jacob/centralised-ai/build/CMakeFiles/collective_robot_behaviour_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/collective_robot_behaviour_test.dir/depend

