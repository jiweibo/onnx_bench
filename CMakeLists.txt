cmake_minimum_required(VERSION 3.12)

project(onnx_bench CXX)

option(ORT_DIR "onnxruntime root dir" /usr)
option(JSONCPP_DIR "jsoncpp_dir" /usr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("/usr/include")
link_directories("/usr/lib")
link_directories("/usr/lib/x86_64-linux-gnu/")

include_directories("/usr/local/include")
link_directories("/usr/local/lib")

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

include_directories("${ORT_DIR}/include")
include_directories("${ORT_DIR}/include/onnxruntime")
include_directories("${ORT_DIR}/include/onnxruntime/core/session")
link_directories("${ORT_DIR}/lib")

include_directories("${JSONCPP_DIR}/include")
link_directories("${JSONCPP_DIR}/lib")

set(DEPS glog gflags cudart)

add_executable(onnx_bench main.cc)
target_link_libraries(onnx_bench onnxruntime)
target_link_libraries(onnx_bench jsoncpp)
target_link_libraries(onnx_bench ${DEPS})


add_executable(test_dataset dataset_test.cc)
target_link_libraries(test_dataset jsoncpp)
target_link_libraries(test_dataset ${DEPS})
