set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Generate Protobuf files
file(GLOB_RECURSE PROTO_FILES proto/*.proto)
protobuf_generate_cpp(PROTO_CPP PROTO_H ${PROTO_FILES} PROTOC_OUT_DIR ${PROTO_GENERATED_DIR})

add_library(ssl_interface_lib
  ${PROTO_CPP}
  ssl_vision_client.cc
  ssl_game_controller_client.cc
  automated_referee.cc
  simulation_reset.cc)

# add executables for demos
add_executable(ssl_vision_client_demo ssl_vision_client_demo.cc)
add_executable(ssl_game_controller_client_demo ssl_game_controller_client_demo.cc)
add_executable(automated_referee_demo automated_referee_demo.cc)

# link libraries
target_link_libraries(ssl_interface_lib ${Protobuf_LIBRARIES})
target_link_libraries(ssl_vision_client_demo ssl_interface_lib)
target_link_libraries(ssl_game_controller_client_demo ssl_interface_lib)
target_link_libraries(automated_referee_demo ssl_interface_lib)
