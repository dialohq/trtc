cmake_minimum_required(VERSION 3.25)
project(trtc-server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRTC_VERSION "dev" CACHE STRING "version the binary reports (/info)")
add_compile_definitions(TRTC_VERSION="${TRTC_VERSION}")

find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)
find_package(LibArchive REQUIRED)
find_path(HTTPLIB_INCLUDE_DIR httplib.h REQUIRED)

# One binary: `serve` (the broker) always; `build` when linked with TensorRT.
# OFF builds a broker-only binary for tests (they stub the build subprocess —
# no GPU or TensorRT needed).
option(TRTC_WITH_TENSORRT "link the build subcommand against TensorRT" ON)

# The OpenAPI contract travels inside the binary (GET /openapi.json);
# builder_config option enums are injected at startup from the tables in
# builder_config.hpp, so the served contract cannot drift from the code.
file(READ openapi.json TRTC_OPENAPI_JSON)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS openapi.json)
configure_file(openapi_embedded.hpp.in openapi_embedded.hpp @ONLY)

add_executable(trtc-server main.cpp server.cpp)
target_include_directories(trtc-server PRIVATE ${HTTPLIB_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(trtc-server PRIVATE OpenSSL::SSL OpenSSL::Crypto nlohmann_json::nlohmann_json
                                          LibArchive::LibArchive Threads::Threads ${CMAKE_DL_LIBS})

if(TRTC_WITH_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
  find_library(TENSORRT_NVINFER nvinfer REQUIRED)
  find_library(TENSORRT_NVONNXPARSER nvonnxparser REQUIRED)
  target_sources(trtc-server PRIVATE build.cpp)
  target_compile_definitions(trtc-server PRIVATE TRTC_WITH_TENSORRT)
  target_include_directories(trtc-server PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(trtc-server PRIVATE ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER})

  # The same build, as its own standalone binary.
  add_executable(trtc-build build_standalone.cpp build.cpp)
  target_include_directories(trtc-build PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(trtc-build PRIVATE OpenSSL::Crypto nlohmann_json::nlohmann_json
                                           ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER} ${CMAKE_DL_LIBS})
  install(TARGETS trtc-build)
endif()

install(TARGETS trtc-server)
