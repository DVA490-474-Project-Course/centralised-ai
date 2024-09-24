set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

find_package(Protobuf REQUIRED)

file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Generate Protobuf files
file(GLOB_RECURSE PROTO_FILES proto/*.proto)
protobuf_generate_cpp(PROTO_CPP PROTO_H ${PROTO_FILES} PROTOC_OUT_DIR ${PROTO_GENERATED_DIR})

add_library(vision_client_lib ssl_vision_client.cc ${PROTO_CPP} ${PROTO_FILES})

target_include_directories(vision_client_lib PUBLIC ${PROTO_GENERATED_DIR})

target_link_libraries(vision_client_lib ${Protobuf_LIBRARIES})
