cmake_minimum_required(VERSION 3.12)

project(onnx_bench CXX CUDA)

include(cmake/third_party.cmake)

# find_package(CUDA REQUIRED)

set(ORT_DIR "/usr" CACHE PATH "onnxruntime root dir")

include_directories(${CMAKE_SOURCE_DIR})

include_directories(/usr/include/ifx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

include_directories("${ORT_DIR}/include")
link_directories("${ORT_DIR}/lib")

add_executable(onnx_bench main.cc)
set(DEPS glog gflags pthread cudart nvToolsExt cnpy jsoncpp z nvidia-ml)
target_link_libraries(onnx_bench onnxruntime)
target_link_libraries(onnx_bench ${DEPS})

add_executable(demo simple_demo.cc)
target_link_libraries(demo onnxruntime)
target_link_libraries(demo ${DEPS})

# add_executable(onnx_bench2 main2.cc)
# target_link_libraries(onnx_bench2 onnxruntime)
# target_link_libraries(onnx_bench2 ${DEPS})

# To use old nvml lib, we need the macro
add_definitions(-DNVML_NO_UNVERSIONED_FUNC_DEFS)

add_executable(ifx_bench ifx_main.cc)
target_link_libraries(ifx_bench ifx cublas)
target_link_libraries(ifx_bench ${DEPS})

add_executable(test_dataset dataset_test.cc)
target_link_libraries(test_dataset jsoncpp)
target_link_libraries(test_dataset ${DEPS})

add_executable(memcpy_test memcpy_test.cu)
target_link_libraries(memcpy_test ${DEPS})