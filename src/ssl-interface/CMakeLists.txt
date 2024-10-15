set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

find_package(Protobuf REQUIRED)

file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Generate Protobuf files
file(GLOB_RECURSE PROTO_FILES proto/*.proto)
protobuf_generate_cpp(PROTO_CPP PROTO_H ${PROTO_FILES} PROTOC_OUT_DIR ${PROTO_GENERATED_DIR})

add_library(ssl_interface_lib ssl_vision_client.cc ssl_game_controller_client.cc ${PROTO_CPP} ${PROTO_FILES} ssl_automated_referee.cc)

# add executables for demos
add_executable(ssl_vision_client_demo ssl_vision_client_demo.cc)
add_executable(ssl_game_controller_client_demo ssl_game_controller_client_demo.cc)
add_executable(ssl_automated_referee_demo ssl_automated_referee_demo.cc)

# include header directories
target_include_directories(ssl_interface_lib PUBLIC ${PROTO_GENERATED_DIR})
target_include_directories(ssl_vision_client_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ssl_game_controller_client_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ssl_automated_referee_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# link libraries
target_link_libraries(ssl_interface_lib ${Protobuf_LIBRARIES})
target_link_libraries(ssl_vision_client_demo ssl_interface_lib)
target_link_libraries(ssl_game_controller_client_demo ssl_interface_lib)
target_link_libraries(ssl_automated_referee_demo ssl_interface_lib)
