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
CMAKE_SOURCE_DIR = /home/aaiza/Desktop/centralised-ai/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aaiza/Desktop/centralised-ai/src/build

# Include any dependencies generated for this target.
include ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include ssl-interface/CMakeFiles/ssl_interface_lib.dir/progress.make

# Include the compile flags for this target's objects.
include ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make

../ssl-interface/generated/messages_robocup_ssl_detection.pb.h: ../ssl-interface/proto/messages_robocup_ssl_detection.proto
../ssl-interface/generated/messages_robocup_ssl_detection.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/messages_robocup_ssl_detection.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/messages_robocup_ssl_detection.proto

../ssl-interface/generated/messages_robocup_ssl_detection.pb.cc: ../ssl-interface/generated/messages_robocup_ssl_detection.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/messages_robocup_ssl_detection.pb.cc

../ssl-interface/generated/messages_robocup_ssl_geometry.pb.h: ../ssl-interface/proto/messages_robocup_ssl_geometry.proto
../ssl-interface/generated/messages_robocup_ssl_geometry.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/messages_robocup_ssl_geometry.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/messages_robocup_ssl_geometry.proto

../ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc: ../ssl-interface/generated/messages_robocup_ssl_geometry.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc

../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.h: ../ssl-interface/proto/messages_robocup_ssl_wrapper.proto
../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/messages_robocup_ssl_wrapper.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/messages_robocup_ssl_wrapper.proto

../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc: ../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc

../ssl-interface/generated/ssl_gc_common.pb.h: ../ssl-interface/proto/ssl_gc_common.proto
../ssl-interface/generated/ssl_gc_common.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_common.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_common.proto

../ssl-interface/generated/ssl_gc_common.pb.cc: ../ssl-interface/generated/ssl_gc_common.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/ssl_gc_common.pb.cc

../ssl-interface/generated/ssl_gc_game_event.pb.h: ../ssl-interface/proto/ssl_gc_game_event.proto
../ssl-interface/generated/ssl_gc_game_event.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_game_event.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_game_event.proto

../ssl-interface/generated/ssl_gc_game_event.pb.cc: ../ssl-interface/generated/ssl_gc_game_event.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/ssl_gc_game_event.pb.cc

../ssl-interface/generated/ssl_gc_geometry.pb.h: ../ssl-interface/proto/ssl_gc_geometry.proto
../ssl-interface/generated/ssl_gc_geometry.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_geometry.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_geometry.proto

../ssl-interface/generated/ssl_gc_geometry.pb.cc: ../ssl-interface/generated/ssl_gc_geometry.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/ssl_gc_geometry.pb.cc

../ssl-interface/generated/ssl_gc_referee_message.pb.h: ../ssl-interface/proto/ssl_gc_referee_message.proto
../ssl-interface/generated/ssl_gc_referee_message.pb.h: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Running cpp protocol buffer compiler on /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_referee_message.proto"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/protoc --cpp_out /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated -I /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto /home/aaiza/Desktop/centralised-ai/src/ssl-interface/proto/ssl_gc_referee_message.proto

../ssl-interface/generated/ssl_gc_referee_message.pb.cc: ../ssl-interface/generated/ssl_gc_referee_message.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate ../ssl-interface/generated/ssl_gc_referee_message.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o: ../ssl-interface/ssl_vision_client.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o -MF CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o.d -o CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_vision_client.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_vision_client.cc > CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_vision_client.cc -o CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o: ../ssl-interface/ssl_game_controller_client.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o -MF CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o.d -o CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_game_controller_client.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_game_controller_client.cc > CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_game_controller_client.cc -o CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o: ../ssl-interface/generated/messages_robocup_ssl_detection.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_detection.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_detection.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_detection.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o: ../ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o: ../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o: ../ssl-interface/generated/ssl_gc_common.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_common.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_common.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_common.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o: ../ssl-interface/generated/ssl_gc_game_event.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_game_event.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_game_event.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_game_event.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o: ../ssl-interface/generated/ssl_gc_geometry.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_geometry.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_geometry.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_geometry.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o: ../ssl-interface/generated/ssl_gc_referee_message.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o -MF CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o.d -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_referee_message.pb.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_referee_message.pb.cc > CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/generated/ssl_gc_referee_message.pb.cc -o CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.s

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/flags.make
ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o: ../ssl-interface/ssl_automated_referee.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o: ssl-interface/CMakeFiles/ssl_interface_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o -MF CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o.d -o CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o -c /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_automated_referee.cc

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.i"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_automated_referee.cc > CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.i

ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.s"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aaiza/Desktop/centralised-ai/src/ssl-interface/ssl_automated_referee.cc -o CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.s

# Object files for target ssl_interface_lib
ssl_interface_lib_OBJECTS = \
"CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o" \
"CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o" \
"CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o"

# External object files for target ssl_interface_lib
ssl_interface_lib_EXTERNAL_OBJECTS =

ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_vision_client.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_game_controller_client.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_detection.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_geometry.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/messages_robocup_ssl_wrapper.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_common.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_game_event.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_geometry.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/generated/ssl_gc_referee_message.pb.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/ssl_automated_referee.o
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/build.make
ssl-interface/libssl_interface_lib.a: ssl-interface/CMakeFiles/ssl_interface_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aaiza/Desktop/centralised-ai/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Linking CXX static library libssl_interface_lib.a"
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && $(CMAKE_COMMAND) -P CMakeFiles/ssl_interface_lib.dir/cmake_clean_target.cmake
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ssl_interface_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ssl-interface/CMakeFiles/ssl_interface_lib.dir/build: ssl-interface/libssl_interface_lib.a
.PHONY : ssl-interface/CMakeFiles/ssl_interface_lib.dir/build

ssl-interface/CMakeFiles/ssl_interface_lib.dir/clean:
	cd /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface && $(CMAKE_COMMAND) -P CMakeFiles/ssl_interface_lib.dir/cmake_clean.cmake
.PHONY : ssl-interface/CMakeFiles/ssl_interface_lib.dir/clean

ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/messages_robocup_ssl_detection.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/messages_robocup_ssl_detection.pb.h
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/messages_robocup_ssl_geometry.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/messages_robocup_ssl_geometry.pb.h
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/messages_robocup_ssl_wrapper.pb.h
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_common.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_common.pb.h
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_game_event.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_game_event.pb.h
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_geometry.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_geometry.pb.h
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_referee_message.pb.cc
ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend: ../ssl-interface/generated/ssl_gc_referee_message.pb.h
	cd /home/aaiza/Desktop/centralised-ai/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aaiza/Desktop/centralised-ai/src /home/aaiza/Desktop/centralised-ai/src/ssl-interface /home/aaiza/Desktop/centralised-ai/src/build /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface /home/aaiza/Desktop/centralised-ai/src/build/ssl-interface/CMakeFiles/ssl_interface_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ssl-interface/CMakeFiles/ssl_interface_lib.dir/depend

