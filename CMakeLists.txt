cmake_minimum_required(VERSION 3.12)

project(onnx_bench CXX CUDA)

include(cmake/third_party.cmake)

# find_package(CUDA REQUIRED)

set(ORT_DIR "/usr" CACHE PATH "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

include_directories("${ORT_DIR}/include")
link_directories("${ORT_DIR}/lib")

add_executable(onnx_bench main.cc)
set(DEPS glog gflags pthread cudart nvToolsExt cnpy jsoncpp z)
target_link_libraries(onnx_bench onnxruntime)
target_link_libraries(onnx_bench ${DEPS})

add_executable(test_dataset dataset_test.cc)
target_link_libraries(test_dataset jsoncpp)
target_link_libraries(test_dataset ${DEPS})
