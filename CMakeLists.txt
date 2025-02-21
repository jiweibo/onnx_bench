cmake_minimum_required(VERSION 3.12)

project(onnx_bench CXX CUDA)

include(cmake/third_party.cmake)

set(ORT_DIR "/usr" CACHE PATH "onnxruntime root dir")
set(IFX_DIR "/usr/local/ifx" CACHE PATH "ifxruntime dir")
set(TENSORRT_DIR "/usr/local/TensorRT" CACHE PATH "tensorrt root dir")

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

include_directories("${ORT_DIR}/include")
link_directories("${ORT_DIR}/lib")

include_directories("${IFX_DIR}/include")
include_directories("/usr/include/ifx")
link_directories("${IFX_DIR}/lib")

include_directories("${TENSORRT_DIR}/include")
link_directories("${TENSORRT_DIR}/lib")

add_executable(onnx_bench main.cc)
set(DEPS glog gflags pthread dl cudart nvToolsExt cnpy jsoncpp z nvidia-ml cupti unwind)
target_link_libraries(onnx_bench onnxruntime)
target_link_libraries(onnx_bench ${DEPS})

add_executable(demo simple_demo.cc)
target_link_libraries(demo onnxruntime)
target_link_libraries(demo ${DEPS})

# To use old nvml lib, we need the macro
add_definitions(-DNVML_NO_UNVERSIONED_FUNC_DEFS)
add_executable(ifx_bench ifx_main.cc)
target_link_libraries(ifx_bench ifx cublas cuda nvperf_host nvperf_target nvinfer)
target_link_libraries(ifx_bench ${DEPS})

#add_subdirectory(cuda_hook)

add_executable(ifx_run_npz ifx_run_npz.cc)
target_link_libraries(ifx_run_npz ifx)
target_link_libraries(ifx_run_npz ${DEPS})

add_executable(test_dataset dataset_test.cc)
target_link_libraries(test_dataset jsoncpp)
target_link_libraries(test_dataset ${DEPS})

add_executable(cuda_runtime_api_test cuda_runtime_api_test.cu)
target_link_libraries(cuda_runtime_api_test ${DEPS})

add_subdirectory(trt)

add_subdirectory(test)
