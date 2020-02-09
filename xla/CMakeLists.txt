cmake_minimum_required(VERSION 3.15)
project(xla)

set(CMAKE_CXX_STANDARD 14)

include_directories(~/torch_xla/third_party/tensorflow)
include_directories(~/torch_xla/third_party/tensorflow/bazel-bin)
include_directories(~/torch_xla/third_party/tensorflow/bazel-tensorflow/external/eigen_archive)
include_directories(~/torch_xla/third_party/tensorflow/bazel-tensorflow/external/com_google_protobuf/src)
include_directories(~/torch_xla/third_party/tensorflow/bazel-tensorflow/external/com_google_absl)

add_executable(xla main.cpp)
target_link_libraries(xla
        ~/torch_xla/third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so)
