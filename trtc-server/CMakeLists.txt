cmake_minimum_required(VERSION 3.25)
project(trtc-server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)
find_path(HTTPLIB_INCLUDE_DIR httplib.h REQUIRED)

# The HTTP broker: no TensorRT anywhere near it.
add_executable(trtc-server server.cpp)
target_include_directories(trtc-server PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(trtc-server PRIVATE OpenSSL::SSL OpenSSL::Crypto nlohmann_json::nlohmann_json
                                          Threads::Threads ${CMAKE_DL_LIBS})

# The build tool: links the one TensorRT this image ships. OFF builds just the
# broker (tests stub the build step, no GPU or TensorRT needed).
option(TRTC_WITH_TENSORRT "build trtc-build against TensorRT" ON)
if(TRTC_WITH_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
  find_library(TENSORRT_NVINFER nvinfer REQUIRED)
  find_library(TENSORRT_NVONNXPARSER nvonnxparser REQUIRED)
  add_executable(trtc-build build.cpp)
  target_include_directories(trtc-build PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(trtc-build PRIVATE ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER} OpenSSL::Crypto
                                           nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
  install(TARGETS trtc-build)
endif()

install(TARGETS trtc-server)
