set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Generate Protobuf files
file(GLOB_RECURSE PROTO_FILES proto/*.proto)
protobuf_generate_cpp(PROTO_CPP PROTO_H ${PROTO_FILES} PROTOC_OUT_DIR ${PROTO_GENERATED_DIR})

add_library(simulation_interface_lib simulation_interface.cc ${PROTO_CPP} ${PROTO_FILES})

# add executable for simulation interface demo
add_executable(simulation_interface_demo simulation_interface_demo.cc)

# include header directories
target_include_directories(simulation_interface_lib PUBLIC ${PROTO_GENERATED_DIR})
target_include_directories(simulation_interface_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation-interface)

# link libraries
target_link_libraries(simulation_interface_lib ${Protobuf_LIBRARIES})
target_link_libraries(simulation_interface_demo simulation_interface_lib)
